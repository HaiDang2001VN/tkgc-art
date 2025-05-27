// g++ -std=c++23 -O3 -pthread -o prepare prepare.cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <queue>
#include <map>
using namespace std;

vector<string> split(string s, const string& delimiter);

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
    int max_hops    = stoi(argv[2]);
    int thread_id   = stoi(argv[3]);
    int num_threads = stoi(argv[4]);

    ofstream log_stream("./logs.txt", ios_base::app);

    log_stream << "Processing thread " << thread_id << " of " << num_threads << endl;
    log_stream << "Max hops: " << max_hops << endl;
    log_stream << "CSV path: " << csv_path << endl;

    // 1) Load CSV
    vector<map<string, string>> csv_data;
    vector<string> col_names;
    vector<Row> rows;
    {
        ifstream in(csv_path);
        string header;
        getline(in, header);

        col_names = split(header, ",");

        string line;
        while (getline(in, line))
        {
            vector<string> row_data = split(line, ",");
            map<string, string> data_by_col;
            for (int i = 0; i < col_names.size(); ++i)
                data_by_col[col_names[i]] = row_data[i];
            csv_data.push_back(data_by_col);
        }
        
        for (map<string, string> row_data: csv_data) {
            try {
                Row row;
                row.edge_type = row_data["edge_type"];
                row.split     = row_data["split"];
                row.edge_id   = stoi(row_data["edge_id"]);
                row.u         = stoi(row_data["u"]);
                row.u_type    = stoi(row_data["u_type"]);
                row.v         = stoi(row_data["v"]);
                row.v_type    = stoi(row_data["v_type"]);
                row.ts        = stoi(row_data["ts"]);
                row.label     = stoi(row_data["label"]);
    
                rows.push_back(row);
            }
            catch (exception ex) {
                ofstream log_stream("./logs.txt", ios_base::app);
                
                for (auto pair: row_data)
                    log_stream << '(' << pair.first << ',' << pair.second << ')' << ' ';

                log_stream << '\n' << ex.what() << '\n';

                throw;
            }
        }
    }

    log_stream << "Loaded " << rows.size() << " rows\n";
    log_stream << "Columns: ";
    for (const auto &col : col_names)
        log_stream << col << " ";
    log_stream << "\n";

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
            cout << r.edge_id << ";";
            cout << best.size() - 1 << ";";

            // cout << "."; // Prefix with a dot to make sure there's always input
            for (size_t i = 0; i < best.size(); ++i)
            {
                if (i)
                    cout << ",";
                cout << best[i].first;// << "|" << best[i].second;
            }
            cout << ";";

            // cout << "."; // Prefix with a dot to make sure there's always input
            for (size_t i = 0; i < node_types.size(); ++i)
            {
                if (i)
                    cout << ",";
                cout << node_types[i];
            }
            cout << ";";

            // cout << "."; // Prefix with a dot to make sure there's always input
            for (size_t i = 0; i < edge_types.size(); ++i)
            {
                if (i)
                    cout << ",";
                cout << edge_types[i];
            }
            cout << endl;
        }
    }
    // flush any remaining edges
    flush_buffer();
    return 0;
}

vector<string> split(string s, const string& delimiter) {
    vector<string> tokens;
    size_t pos = 0;
    string token;
    while ((pos = s.find(delimiter)) != string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);

    return tokens;
}