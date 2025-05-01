#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def load_config(path):
    print(f"[LOG] Loading config from {path}")
    with open(path, 'r') as f:
        config = json.load(f)
    print(
        f"[LOG] Config loaded: dataset={config.get('dataset')}, storage_dir={config.get('storage_dir')}")
    return config


class CsvNeo4jManager:
    def __init__(
        self,
        config_path,
        uri="neo4j://localhost:7687",
        auth=("neo4j", "password"),
        max_hops=5
    ):
        print(f"[LOG] Initializing with URI={uri}, max_hops={max_hops}")
        self.config = load_config(config_path)
        self.max_hops = max_hops
        self.driver = GraphDatabase.driver(uri, auth=auth)

        # Load edges CSV
        csv_file = os.path.join(
            self.config['storage_dir'], f"{self.config['dataset']}_edges.csv")
        print(f"[LOG] Loading edges DataFrame from {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"[LOG] Loaded {len(df)} rows from edges CSV")

        # Graph edges: positive label == 1
        pos_df = df[df['label'] == 1].copy()
        # Query edges: split != 'pre'
        qry_df = df[df['split'] != 'pre'].copy()

        # Prepare DataFrame for CSV export (graph structure)
        self.pos_df = pos_df[['u', 'v', 'ts', 'edge_type']].rename(
            columns={'u': ':START_ID', 'v': ':END_ID', 'ts': 'ts:int'}
        )
        print(f"[LOG] Positive edges for graph: {len(self.pos_df)}")

        # Prepare query edge list with edge_id
        self.query_edges = list(
            qry_df[['edge_id', 'u', 'v', 'ts']].itertuples(index=False, name=None))
        print(
            f"[LOG] Query edges for pathfinding: {len(self.query_edges)} entries")

    def create_csv(self, out_dir):
        print(f"[LOG] Creating CSVs in {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        # Nodes CSV
        nodes = pd.unique(
            self.pos_df[[':START_ID', ':END_ID']].values.ravel('K'))
        nodes_csv = os.path.join(out_dir, "nodes.csv")
        pd.DataFrame({"node_id:ID": nodes}).to_csv(nodes_csv, index=False)
        print(f"[LOG] Wrote nodes CSV: {nodes_csv}")

        # Edges CSV
        edges_csv = os.path.join(out_dir, "edges.csv")
        self.pos_df.to_csv(edges_csv, index=False)
        print(f"[LOG] Wrote edges CSV: {edges_csv}")

        return nodes_csv, edges_csv

    def import_csv(self, nodes_csv, edges_csv):
        print("[LOG] Importing CSVs into Neo4j")
        with self.driver.session() as sess:
            sess.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.node_id IS UNIQUE"
            )
            print("[LOG] Uniqueness constraint on Node.node_id")
            # Import nodes
            sess.run(
                "LOAD CSV WITH HEADERS FROM $file AS r "
                "MERGE (n:Node {node_id: toInteger(r['node_id:ID'])})",
                file=f"file:///{nodes_csv}"
            )
            print(f"[LOG] Imported nodes from {nodes_csv}")
            # Import edges
            sess.run(
                "LOAD CSV WITH HEADERS FROM $file AS r "
                "MATCH (a:Node {node_id: toInteger(r[':START_ID'])}),"
                "(b:Node {node_id: toInteger(r[':END_ID'])}) "
                "MERGE (a)-[l:LINK]->(b) "
                "SET l.ts = toInteger(r['ts:int']), l.edge_type = r.edge_type",
                file=f"file:///{edges_csv}"
            )
            print(f"[LOG] Imported edges from {edges_csv}")

    def find_shortest(self, u, v, ts):
        cypher = (
            "MATCH (a:Node {node_id:$u}), (b:Node {node_id:$v}) "
            "MATCH p=(a)-[rels:LINK*1..$h]-(b) "
            "WHERE ALL(r IN rels WHERE r.ts < $ts) "
            "RETURN [n IN nodes(p)| n.node_id] AS nodes, "
            "length(rels) AS hops, "
            "[r IN rels | r.edge_type] AS meta_path "
            "ORDER BY hops ASC LIMIT 1"
        )
        params = {'u': u, 'v': v, 'h': self.max_hops, 'ts': ts}
        with self.driver.session() as sess:
            rec = sess.run(cypher, **params).single()
            if rec:
                return {'nodes': rec['nodes'], 'hops': rec['hops'], 'meta_path': rec['meta_path']}
            return None

    def _compute_record(self, record):
        edge_id, u, v, ts = record
        print(f"[TRACE] computing edge_id={edge_id} ({u}->{v}) ts={ts}")
        path = self.find_shortest(u, v, ts)
        return {'edge_id': edge_id, 'path': path}

    def find_paths(self, out_dir):
        print("[LOG] Computing paths for query edges")
        os.makedirs(out_dir, exist_ok=True)
        total = len(self.query_edges)
        print(f"[LOG] Total query edges: {total}")
        results = []
        hops_list = []
        found = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            for res in tqdm(executor.map(self._compute_record, self.query_edges),
                            total=total, desc="Paths"):
                results.append(res)
                if res['path'] is not None:
                    found += 1
                    hops_list.append(res['path']['hops'])
        coverage = found / total if total else 0
        print(
            f"[LOG] Coverage: found paths for {found}/{total} edges ({coverage:.2%})")
        if hops_list:
            arr = np.array(hops_list)
            print(
                f"[LOG] Hops statistics: mean={arr.mean():.2f}, std={arr.std():.2f}, min={arr.min()}, max={arr.max()}")
            q1, q2, q3 = np.percentile(arr, [25, 50, 75])
            print(f"[LOG] Hops quartiles: 25%={q1}, 50%={q2}, 75%={q3}")
        mapping = {str(r['edge_id']): r['path'] for r in results}
        map_file = os.path.join(out_dir, "paths.json")
        with open(map_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"[LOG] Saved edge_id->path mapping to {map_file}")

    def close(self):
        print("[LOG] Closing Neo4j driver")
        self.driver.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export CSV graph & find paths in Neo4j")
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--uri', default='neo4j://localhost:7687')
    parser.add_argument('--user', default='neo4j')
    parser.add_argument('--password', required=True)
    parser.add_argument('--max_hops', type=int, default=5)
    parser.add_argument('--out', default='./neo4j_import')
    args = parser.parse_args()
    print(f"[LOG] args: {vars(args)}")
    mgr = CsvNeo4jManager(
        config_path=args.config,
        uri=args.uri,
        auth=(args.user, args.password),
        max_hops=args.max_hops
    )
    nodes_csv, edges_csv = mgr.create_csv(args.out)
    mgr.import_csv(nodes_csv, edges_csv)
    mgr.find_paths(args.out)
    mgr.close()
