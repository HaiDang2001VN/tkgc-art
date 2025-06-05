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
    int edge_id, ts, label;
    string u, v; // u_type and v_type removed
    string edge_type, split;
};

// using Node = pair<string, string>; // (node_id_string, node_type_string) // REMOVED

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
    // vector<map<string, string>> csv_data; // Removed
    vector<string> col_names;
    vector<Row> rows;
    unordered_map<string, string> node_id_to_type_map;   // Moved here: Map from node_id to node_type
    {
        ifstream in(csv_path);
        string header;
        getline(in, header);

        col_names = split(header, ",");

        string line;
        while (getline(in, line))
        {
            vector<string> row_data_vec = split(line, ",");
            unordered_map<string, string> row_map_data; // Changed to unordered_map
            for (int i = 0; i < col_names.size(); ++i)
                row_map_data[col_names[i]] = row_data_vec[i];

            try {
                Row row;
                row.edge_type = row_map_data["edge_type"];
                row.split     = row_map_data["split"];
                row.edge_id   = stoi(row_map_data["edge_id"]);
                row.u         = row_map_data["u"];
                // row.u_type    = row_map_data["u_type"]; // Removed
                row.v         = row_map_data["v"];
                // row.v_type    = row_map_data["v_type"]; // Removed
                row.ts        = stoi(row_map_data["ts"]);
                row.label     = stoi(row_map_data["label"]);
    
                rows.push_back(row);

                // Populate node type map during CSV loading
                node_id_to_type_map[row.u] = row_map_data["u_type"]; // Use row_map_data directly
                node_id_to_type_map[row.v] = row_map_data["v_type"]; // Use row_map_data directly

            }
            catch (exception ex) {
                ofstream log_stream("./logs.txt", ios_base::app);
                
                for (auto pair: row_map_data)
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

    // 2) Sort by ts desc, label desc
    sort(rows.begin(), rows.end(), [](auto &a, auto &b)
         {
        if (a.ts != b.ts) return a.ts > b.ts; // Sort in descending order of ts
        return a.label > b.label; });

    // 3) Incremental graph storage
    unordered_map<string, vector<pair<string, string>>> adj; // Key: node_id, Value: vector of (neighbor_id, edge_type)
    // unordered_map<string, string> node_id_to_type_map;   // Map from node_id to node_type // MOVED
    int cur_ts = rows.empty() ? 0 : rows.back().ts; // Initialize cur_ts to the last timestamp
    vector<Row> buffer;

    auto flush_buffer = [&]()
    {
        for (auto &buf_r : buffer)
        {
            if (buf_r.label != 1)
                continue; // only positive
            
            // Populate node type map - REMOVED FROM HERE
            // node_id_to_type_map[buf_r.u] = buf_r.u_type;
            // node_id_to_type_map[buf_r.v] = buf_r.v_type;

            // Add edges to adjacency list (neighbor_id, edge_type)
            adj[buf_r.u].emplace_back(buf_r.v, buf_r.edge_type); 
            adj[buf_r.v].emplace_back(buf_r.u, buf_r.edge_type); 
        }
        buffer.clear();
    };

    // 4) Process rows, and for query‐edges run BFS
    while (!rows.empty())
    {
        Row& r = rows.back(); // Get the last row
        if (r.ts != cur_ts)
        {
            flush_buffer();
            cur_ts = r.ts;
        }
        buffer.push_back(r);

        bool do_query = (r.split != "pre") && (r.edge_id % num_threads == thread_id);
        if (do_query)
        {
            // BFS with depth‐limit = max_hops
            string src_id = r.u;
            string dst_id = r.v;

            queue<string> q; // Queue stores node IDs
            unordered_map<string, string> parent_map; // Key: child_id, Value: parent_id
            unordered_map<string, int> node_depths;   // Key: node_id, Value: depth
            
            q.push(src_id);
            node_depths[src_id] = 0; 
            vector<string> best; // Stores path as a vector of node IDs
            // bool path_to_dst_found_bfs = false; // Removed

            while (!q.empty())
            {
                string cur_node_id = q.front();
                q.pop();

                int cur_depth = node_depths[cur_node_id];

                if (cur_depth >= max_hops) 
                {
                    continue;
                }

                // Check if cur_node_id exists in adj before iterating
                if (adj.count(cur_node_id)) {
                    for (auto &pr : adj.at(cur_node_id)) // pr is pair<string, string> (neighbor_id, edge_type)
                    {
                        string neighbor_node_id = pr.first;
                        if (!node_depths.count(neighbor_node_id)) 
                        {
                            parent_map[neighbor_node_id] = cur_node_id; 
                            node_depths[neighbor_node_id] = cur_depth + 1;

                            if (neighbor_node_id == dst_id) 
                            {
                                string temp_node_id = dst_id;
                                while (temp_node_id != src_id) 
                                {
                                    best.push_back(temp_node_id);
                                    temp_node_id = parent_map.at(temp_node_id); 
                                }
                                best.push_back(src_id);
                                reverse(best.begin(), best.end());
                                // path_to_dst_found_bfs = true; // Removed
                                break; 
                            }
                            q.push(neighbor_node_id); 
                        }
                    }
                }
                if (!best.empty()) // If path was found
                {
                    break; 
                }
            }

            // Only print when a path is found
            if (!best.empty()) // best is vector<string> (node IDs)
            {
                // Collect meta-paths
                vector<string> edge_types, node_types_in_path;
                for (size_t i = 0; i + 1 < best.size(); ++i)
                {
                    string current_path_node_id = best[i]; 
                    string next_path_node_id = best[i + 1];
                    if (adj.count(current_path_node_id)) {
                        for (auto &pr : adj.at(current_path_node_id)) // pr is (neighbor_id, edge_type)
                        {
                            if (pr.first == next_path_node_id)
                            {
                                edge_types.push_back(pr.second);
                                break;
                            }
                        }
                    }
                }
                for (const string& node_id_in_path : best)
                {
                    // Ensure node_id_to_type_map has the type, otherwise handle (e.g., "UNKNOWN")
                    if (node_id_to_type_map.count(node_id_in_path)) {
                        node_types_in_path.push_back(node_id_to_type_map.at(node_id_in_path));
                    } else {
                        node_types_in_path.push_back("UNKNOWN_TYPE"); // Fallback
                    }
                }

                // Print: edge_id ; hops ; node_ids ; node_types ; edge_types
                cout << r.edge_id << ";";
                cout << best.size() - 1 << ";";

                for (size_t i = 0; i < best.size(); ++i)
                {
                    if (i)
                        cout << ",";
                    cout << best[i]; // best[i] is the string ID
                }
                cout << ";";

                for (size_t i = 0; i < node_types_in_path.size(); ++i)
                {
                    if (i)
                        cout << ",";
                    cout << node_types_in_path[i]; 
                }
                cout << ";";

                for (size_t i = 0; i < edge_types.size(); ++i)
                {
                    if (i)
                        cout << ",";
                    cout << edge_types[i];
                }
                cout << endl;
            }
        }
        rows.pop_back(); // Remove the processed row
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