// pathfinder.cpp
#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
using namespace std;

struct Row
{
    int edge_id, u, u_type, v, v_type, ts, label;
    string edge_type, split;
};

using Node = pair<int, int>; // (node_id, node_type)

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <csv_path> <max_hops> <thread_id> <num_threads>\n";
        return 1;
    }
    string csv_path = argv[1];
    int max_hops = stoi(argv[2]);
    int thread_id = stoi(argv[3]);
    int num_threads = stoi(argv[4]);

    // 1) Load CSV
    vector<Row> rows;
    {
        ifstream in(csv_path);
        string header;
        getline(in, header);
        string line;
        while (getline(in, line))
        {
            Row r;
            stringstream ss(line);
            getline(ss, line, ',');
            r.edge_id = stoi(line);
            getline(ss, line, ',');
            r.u = stoi(line);
            getline(ss, line, ',');
            r.v = stoi(line);
            getline(ss, line, ',');
            r.u_type = stoi(line);
            getline(ss, line, ',');
            r.v_type = stoi(line);
            getline(ss, line, ',');
            r.ts = stoi(line);
            getline(ss, r.split, ',');
            getline(ss, line, ',');
            r.label = stoi(line);
            getline(ss, r.edge_type, '\n');
            rows.push_back(r);
        }
    }

    // 2) Sort by ts asc, label desc
    sort(rows.begin(), rows.end(), [](auto &a, auto &b)
         {
        if (a.ts != b.ts) return a.ts < b.ts;
        return a.label > b.label; });

    // 3) Incremental graph storage
    unordered_map<int, vector<pair<Node, string>>> adj;
    int cur_ts = rows.empty() ? 0 : rows[0].ts;
    vector<Row> buffer;

    auto flush_buffer = [&]()
    {
        for (auto &r : buffer)
        {
            if (r.label != 1)
                continue; // only positive
            Node n1 = {r.u, r.u_type}, n2 = {r.v, r.v_type};
            adj[r.u].emplace_back(n2, r.edge_type);
            adj[r.v].emplace_back(n1, r.edge_type);
        }
        buffer.clear();
    };

    // 4) Process rows, and for query‐edges run BFS
    for (auto &r : rows)
    {
        if (r.ts != cur_ts)
        {
            flush_buffer();
            cur_ts = r.ts;
        }
        buffer.push_back(r);

        bool do_query = (r.split != "pre") && (r.edge_id % num_threads == thread_id);
        if (!do_query)
            continue;

        // BFS with depth‐limit = max_hops
        Node src = {r.u, r.u_type}, dst = {r.v, r.v_type};
        queue<vector<Node>> q;
        unordered_set<long long> seen;
        auto key = [&](const Node &n)
        {
            return (long long)n.first << 32 | (unsigned long long)n.second;
        };
        q.push({src});
        seen.insert(key(src));
        vector<Node> best;

        while (!q.empty())
        {
            auto path = q.front();
            q.pop();
            int depth = (int)path.size() - 1;
            if (depth > max_hops)
                continue;
            Node cur = path.back();
            if (cur == dst)
            {
                best = path;
                break;
            }
            if (depth == max_hops)
                continue; // do not expand further
            for (auto &pr : adj[cur.first])
            {
                Node nb = pr.first;
                long long k = key(nb);
                if (!seen.count(k))
                {
                    seen.insert(k);
                    auto np = path;
                    np.push_back(nb);
                    q.push(move(np));
                }
            }
        }

        // Only print when a path is found
        if (!best.empty())
        {
            // Collect meta-paths
            vector<string> edge_types, node_types;
            for (size_t i = 0; i + 1 < best.size(); ++i)
            {
                int u_id = best[i].first;
                Node nxt = best[i + 1];
                for (auto &pr : adj[u_id])
                {
                    if (pr.first == nxt)
                    {
                        edge_types.push_back(pr.second);
                        break;
                    }
                }
            }
            for (auto &n : best)
            {
                node_types.push_back(to_string(n.second));
            }

            // Print: edge_id \t hops \t nodes(id|type,...) \t node_types(...) \t edge_types(...)
            cout << r.edge_id << "\t";
            cout << best.size() - 1 << "\t";
            for (size_t i = 0; i < best.size(); ++i)
            {
                if (i)
                    cout << ",";
                cout << best[i].first << "|" << best[i].second;
            }
            cout << "\t";
            for (size_t i = 0; i < node_types.size(); ++i)
            {
                if (i)
                    cout << ",";
                cout << node_types[i];
            }
            cout << "\t";
            for (size_t i = 0; i < edge_types.size(); ++i)
            {
                if (i)
                    cout << ",";
                cout << edge_types[i];
            }
            cout << "\n";
        }
    }
    // flush any remaining edges
    flush_buffer();
    return 0;
}
