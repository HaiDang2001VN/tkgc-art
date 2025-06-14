// g++ -std=c++23 -O3 -pthread -o prepare.o prepare.cpp
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
#include <chrono>   // Added
#include <iomanip>  // Added

using namespace std;

// Helper function to get current timestamp
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

vector<string> split(const string& s, char delimiter); // Changed delimiter to char

struct Row
{
    int edge_id, ts, label;
    int u, v, edge_type; 
    string split;
};

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        // Initial error to std::cerr as log_stream is not yet set up.
        std::cerr << "[Error] " << getCurrentTimestamp() << " Usage: " << argv[0] << " <csv_path> <max_hops> <thread_id> <num_threads>\n";
        return 1;
    }
    string csv_path = argv[1];
    int max_hops    = stoi(argv[2]);
    int thread_id   = stoi(argv[3]);
    int num_threads = stoi(argv[4]);

    string prefix = csv_path;
    size_t suffix_pos = prefix.rfind("_edges.csv");
    if (suffix_pos != string::npos) {
        prefix.erase(suffix_pos);
    }
    // Adjusted log file name
    string log_file_name = prefix + "_prepare_logs_" + to_string(num_threads) + "_" + to_string(thread_id) + ".txt";

    ofstream log_stream(log_file_name, ios_base::app);
    if (!log_stream.is_open()) {
        std::cerr << "[Error] " << getCurrentTimestamp() << " Failed to open log file: " << log_file_name << std::endl;
        return 1; // Cannot proceed without log file
    }

    log_stream << "[Info] " << getCurrentTimestamp() << " Processing thread " << thread_id << " of " << num_threads << endl;
    log_stream << "[Info] " << getCurrentTimestamp() << " Max hops: " << max_hops << endl;
    log_stream << "[Info] " << getCurrentTimestamp() << " CSV path: " << csv_path << endl;

    // 1) Load CSV
    vector<string> col_names;
    vector<Row> rows;
    unordered_map<int, int> node_id_to_type_map;   // Map from node_id to node_type
    {
        ifstream in(csv_path);
        if (!in.is_open()) {
            log_stream << "[Error] " << getCurrentTimestamp() << " Failed to open CSV file: " << csv_path << endl;
            return 1;
        }
        string header;
        if (!getline(in, header)) {
            log_stream << "[Error] " << getCurrentTimestamp() << " Failed to read header from CSV file: " << csv_path << endl;
            return 1;
        }

        col_names = split(header, ','); 

        string line;
        int line_num = 1;
        while (getline(in, line))
        {
            line_num++;
            vector<string> row_data_vec = split(line, ','); 
            if (row_data_vec.size() != col_names.size()) {
                log_stream << "[Warning] " << getCurrentTimestamp() << " Line " << line_num << ": Skipping due to mismatched column count. Expected " << col_names.size() << ", got " << row_data_vec.size() << endl;
                continue;
            }
            unordered_map<string, string> row_map_data; 
            for (size_t i = 0; i < col_names.size(); ++i) // Use size_t for loop counter
                row_map_data[col_names[i]] = row_data_vec[i];

            try {
                Row row;
                row.edge_type = stoi(row_map_data.at("edge_type")); 
                row.split     = row_map_data.at("split");
                row.edge_id   = stoi(row_map_data.at("edge_id"));
                row.u         = stoi(row_map_data.at("u")); 
                row.v         = stoi(row_map_data.at("v")); 
                row.ts        = stoi(row_map_data.at("ts"));
                row.label     = stoi(row_map_data.at("label"));
    
                rows.push_back(row);

                node_id_to_type_map[row.u] = stoi(row_map_data.at("u_type")); 
                node_id_to_type_map[row.v] = stoi(row_map_data.at("v_type")); 

            }
            catch (const std::out_of_range& oor) {
                log_stream << "[Error] " << getCurrentTimestamp() << " Line " << line_num << ": Missing expected column. Details: " << oor.what() << ". Row data: ";
                for (auto const& [key, val] : row_map_data) {
                    log_stream << '(' << key << ',' << val << ')' << ' ';
                }
                log_stream << endl;
                // Decide if to throw or continue
            }
            catch (const std::invalid_argument& ia) {
                 log_stream << "[Error] " << getCurrentTimestamp() << " Line " << line_num << ": Invalid argument for conversion. Details: " << ia.what() << ". Row data: ";
                for (auto const& [key, val] : row_map_data) {
                    log_stream << '(' << key << ',' << val << ')' << ' ';
                }
                log_stream << endl;
                // Decide if to throw or continue
            }
            catch (const exception& ex) {
                log_stream << "[Error] " << getCurrentTimestamp() << " Line " << line_num << ": An unexpected error occurred. Details: " << ex.what() << ". Row data: ";
                for (auto const& [key, val] : row_map_data) {
                    log_stream << '(' << key << ',' << val << ')' << ' ';
                }
                log_stream << endl;
                throw; // Re-throw for unexpected critical errors
            }
        }
        in.close(); // Explicitly close though RAII handles it
    }

    log_stream << "[Info] " << getCurrentTimestamp() << " Loaded " << rows.size() << " rows" << endl;
    log_stream << "[Info] " << getCurrentTimestamp() << " Columns: ";
    for (const auto &col : col_names)
        log_stream << col << " ";
    log_stream << endl;

    // 2) Sort by ts desc, label desc
    sort(rows.begin(), rows.end(), [](auto &a, auto &b)
         {
        if (a.ts != b.ts) return a.ts > b.ts; 
        return a.label > b.label; });
    log_stream << "[Info] " << getCurrentTimestamp() << " Rows sorted." << endl;

    // 3) Incremental graph storage
    unordered_map<int, vector<pair<int, int>>> adj; 
    int cur_ts = rows.empty() ? 0 : rows.back().ts; 
    vector<Row> buffer;
    int paths_found_count = 0; // Counter for paths found

    auto flush_buffer = [&]()
    {
        for (auto &buf_r : buffer)
        {
            if (buf_r.label != 1)
                continue; 
            
            adj[buf_r.u].emplace_back(buf_r.v, buf_r.edge_type); 
            adj[buf_r.v].emplace_back(buf_r.u, buf_r.edge_type); 
        }
        buffer.clear();
    };

    // // Output total paths count first (placeholder, will be updated)
    // // This line will be overwritten later if paths are found.
    // // If no paths are found, it will remain 0.
    // long paths_count_pos = cout.tellp(); // Get current position
    // cout << "0" << endl; // Placeholder for total paths count


    // 4) Process rows, and for query‐edges run BFS
    log_stream << "[Info] " << getCurrentTimestamp() << " Starting BFS processing..." << endl;
    while (!rows.empty())
    {
        Row& r = rows.back(); 
        if (r.ts != cur_ts)
        {
            flush_buffer();
            cur_ts = r.ts;
        }
        buffer.push_back(r);

        bool do_query = (r.split != "pre") && (r.edge_id % num_threads == thread_id);
        if (do_query)
        {
            int src_id = r.u; 
            int dst_id = r.v; 

            queue<int> q; 
            unordered_map<int, int> parent_map; 
            unordered_map<int, int> node_depths;   
            
            q.push(src_id);
            node_depths[src_id] = 0; 
            vector<int> best; 
            
            while (!q.empty())
            {
                int cur_node_id = q.front(); 
                q.pop();
                int cur_depth = node_depths[cur_node_id];

                if (cur_depth >= max_hops) 
                {
                    continue;
                }

                if (adj.count(cur_node_id)) {
                    for (auto &pr : adj.at(cur_node_id)) 
                    {
                        int neighbor_node_id = pr.first; 
                        if (!node_depths.count(neighbor_node_id)) 
                        {
                            parent_map[neighbor_node_id] = cur_node_id; 
                            node_depths[neighbor_node_id] = cur_depth + 1;

                            if (neighbor_node_id == dst_id) 
                            {
                                int temp_node_id = dst_id; 
                                while (temp_node_id != src_id) 
                                {
                                    best.push_back(temp_node_id);
                                    temp_node_id = parent_map.at(temp_node_id); 
                                }
                                best.push_back(src_id);
                                reverse(best.begin(), best.end());
                                break; 
                            }
                            q.push(neighbor_node_id); 
                        }
                    }
                }
                if (!best.empty()) 
                {
                    break; 
                }
            }

            if (!best.empty()) 
            {
                paths_found_count++; // Increment counter
                vector<int> edge_types; 
                vector<int> node_types_in_path; 
                for (size_t i = 0; i + 1 < best.size(); ++i)
                {
                    int current_path_node_id = best[i]; 
                    int next_path_node_id = best[i + 1]; 
                    bool found_edge = false;
                    if (adj.count(current_path_node_id)) {
                        for (auto &pr : adj.at(current_path_node_id)) 
                        {
                            if (pr.first == next_path_node_id)
                            {
                                edge_types.push_back(pr.second);
                                found_edge = true;
                                break;
                            }
                        }
                    }
                    if (!found_edge) {
                        // This case should ideally not happen if BFS found a valid path
                        // through the existing adj list. Log if it does.
                        log_stream << "[Warning] " << getCurrentTimestamp() << " Edge not found in adj between "
                                   << current_path_node_id << " and " << next_path_node_id 
                                   << " for EID " << r.edge_id << " path reconstruction." << endl;
                        // Add a placeholder or handle error, e.g., edge_types.push_back(-1);
                    }
                }
                for (const int& node_id_in_path : best) 
                {
                    if (node_id_to_type_map.count(node_id_in_path)) {
                        node_types_in_path.push_back(node_id_to_type_map.at(node_id_in_path));
                    } else {
                        log_stream << "[Warning] " << getCurrentTimestamp() << " Node type not found for node " << node_id_in_path 
                                   << " in EID " << r.edge_id << ". Using fallback 0." << endl;
                        node_types_in_path.push_back(0); // Fallback
                    }
                }

                // Output format: eid;hops;nodes;node_types;edge_types
                cout << r.edge_id << ";";
                cout << best.size() - 1 << ";"; // hops

                for (size_t i = 0; i < best.size(); ++i)
                {
                    cout << best[i] << (i == best.size() - 1 ? "" : ",");
                }
                cout << ";";

                for (size_t i = 0; i < node_types_in_path.size(); ++i)
                {
                    cout << node_types_in_path[i] << (i == node_types_in_path.size() - 1 ? "" : ",");
                }
                cout << ";";

                for (size_t i = 0; i < edge_types.size(); ++i)
                {
                    cout << edge_types[i] << (i == edge_types.size() - 1 ? "" : ",");
                }
                cout << "\n"; // Newline after each complete path entry
            }
        }
        rows.pop_back(); 
    }
    flush_buffer();
    log_stream << "[Info] " << getCurrentTimestamp() << " BFS processing finished." << endl;

    // // Go back and write the actual total paths count
    // long end_pos = cout.tellp();
    // cout.seekp(paths_count_pos);
    // cout << paths_found_count; // Write actual count
    // cout.seekp(end_pos); // Return to the end to not mess up subsequent cout if any
    // cout.flush(); // Ensure it's written

    log_stream << "[Info] " << getCurrentTimestamp() << " Found and wrote " << paths_found_count << " paths for thread " << thread_id << "." << endl;
    log_stream.close();
    return 0;
}

vector<string> split(const string& s, char delimiter) { // Changed to char delimiter
    vector<string> tokens;
    string token;
    stringstream ss(s);
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}