// g++ -std=c++23 -O3 -pthread -o preprocess.o preprocess.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <numeric>
#include <functional>
#include <iterator>
#include <chrono>   // Added
#include <iomanip>  // Added

// --- Type Definitions ---
using EmbeddingMap = std::unordered_map<int, std::vector<float>>;
using Path = std::vector<int>;
// Enhanced adjacency list: node -> edge_type -> neighbor_type -> timestamp -> neighbor
using AdjacencyList = std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::multimap<int, int>>>>;

struct ShortestPathInfo
{
    int hops;
    std::vector<int> nodes;
    std::vector<int> node_types;
    std::vector<int> edge_types;
    std::vector<int> edge_timestamps; // Added field for edge timestamps
};

struct MetaPathPattern
{
    std::vector<std::tuple<int, int, int>> steps; // (edge_type, target_node_type, timestamp)
};

// --- Utility Functions ---
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    // Ensure std::localtime is used safely if multi-threading becomes a concern here.
    // For this single-threaded preprocess.cpp context, it's generally fine.
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<std::string> split_str(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

// std::string trim(const std::string &str) 
// { ... } // This function might no longer be needed if only used by old JSON parser

// --- Data Loading Functions ---
void load_embeddings_from_file(const std::string &file_path, EmbeddingMap &embeddings, std::ofstream& log_stream)
{
    std::ifstream file(file_path);
    if (!file.is_open()) {
        // Log before throwing if desired, or let the caller log.
        // For now, keeping the throw, caller will log.
        throw std::runtime_error("Could not open embedding file: " + file_path);
    }

    std::string line;
    std::getline(file, line); // Skip header

    int id = 0;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        std::vector<float> vec;
        float val;
        while (iss >> val)
            vec.push_back(val);
        if (!vec.empty())
            embeddings[id++] = vec;
    }
    log_stream << "[Info] " << getCurrentTimestamp() << " Loaded " << embeddings.size() << " embeddings from " << file_path << std::endl;
}

std::unordered_map<int, ShortestPathInfo> load_shortest_paths(
    const std::string &txt_path,
    const std::unordered_map<int, int>& node_to_type_map, // Remains for now, though not used for path_info.node_types
    std::ofstream& log_stream)
{
    std::unordered_map<int, ShortestPathInfo> paths;
    std::ifstream file(txt_path);
    if (!file.is_open())
    {
        log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Could not open shortest paths file: " << txt_path << std::endl;
        return paths;
    }

    int total_paths_in_file_header;
    if (!(file >> total_paths_in_file_header))
    {
        log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Shortest paths file is empty or cannot read count: " << txt_path << std::endl;
        return paths;
    }

    int current_eid;
    while (file >> current_eid)
    {
        ShortestPathInfo path_info;
        bool entry_valid = true;

        if (!(file >> path_info.hops)) {
            log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Failed to read hops for EID " << current_eid << " in " << txt_path << ". File may be truncated or malformed." << std::endl;
            break; 
        }

        path_info.nodes.reserve(path_info.hops + 1);
        for (int i = 0; i < path_info.hops + 1; ++i) {
            int node_val;
            if (!(file >> node_val)) {
                log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Failed to read node " << (i + 1) << "/" << (path_info.hops + 1) 
                          << " for EID " << current_eid << " in " << txt_path << "." << std::endl;
                entry_valid = false;
                break;
            }
            path_info.nodes.push_back(node_val);
        }
        if (!entry_valid) continue;

        path_info.node_types.reserve(path_info.hops + 1);
        for (int i = 0; i < path_info.hops + 1; ++i) {
            int node_type_val;
            if (!(file >> node_type_val)) {
                log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Failed to read node_type " << (i + 1) << "/" << (path_info.hops + 1)
                          << " for EID " << current_eid << " in " << txt_path << "." << std::endl;
                entry_valid = false;
                break;
            }
            path_info.node_types.push_back(node_type_val);
        }
        if (!entry_valid) continue;

        path_info.edge_types.reserve(path_info.hops);
        for (int i = 0; i < path_info.hops; ++i) {
            int edge_type_val;
            if (!(file >> edge_type_val)) {
                log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Failed to read edge_type " << (i + 1) << "/" << path_info.hops
                          << " for EID " << current_eid << " in " << txt_path << "." << std::endl;
                entry_valid = false;
                break;
            }
            path_info.edge_types.push_back(edge_type_val);
        }
        if (!entry_valid) continue;

        // Read edge timestamps if available
        path_info.edge_timestamps.reserve(path_info.hops);
        for (int i = 0; i < path_info.hops; ++i) {
            int edge_timestamp_val;
            if (!(file >> edge_timestamp_val)) {
                log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Failed to read edge_timestamp " << (i + 1) << "/" << path_info.hops
                          << " for EID " << current_eid << " in " << txt_path << ". It will be set to -1." << std::endl;
                edge_timestamp_val = -1; // Default value indicating missing timestamp
            }
            path_info.edge_timestamps.push_back(edge_timestamp_val);
        }
        if (!entry_valid) continue;

        if (entry_valid) {
            if (path_info.nodes.size() == static_cast<size_t>(path_info.hops + 1) &&
                path_info.node_types.size() == static_cast<size_t>(path_info.hops + 1) &&
                path_info.edge_types.size() == static_cast<size_t>(path_info.hops)) {
                paths[current_eid] = path_info;
            } else {
                 log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Data count inconsistency for EID " << current_eid << " after parsing. "
                           << "Nodes: " << path_info.nodes.size() << " (expected " << path_info.hops + 1 << "), "
                           << "Node Types: " << path_info.node_types.size() << " (expected " << path_info.hops + 1 << "), "
                           << "Edge Types: " << path_info.edge_types.size() << " (expected " << path_info.hops << "). "
                           << "Skipping this entry." << std::endl;
            }
        }
    }

    if (file.fail() && !file.eof()) {
        log_stream << "[Error] " << getCurrentTimestamp() << " Warning: File stream error after processing paths in " << txt_path << ". Data might be incomplete or malformed." << std::endl;
    }

    log_stream << "[Info] " << getCurrentTimestamp() << " Loaded " << paths.size() << " shortest paths from " << txt_path << std::endl;
    return paths;
}

MetaPathPattern extract_meta_path_pattern(const ShortestPathInfo &path_info, std::ofstream& log_stream)
{
    MetaPathPattern pattern;
    if (path_info.edge_types.size() != path_info.node_types.size() - 1 ||
        path_info.edge_types.size() != path_info.edge_timestamps.size())
    {
        log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Mismatched sizes in path info when extracting meta-path." << std::endl;
        return pattern;
    }

    for (size_t i = 0; i < path_info.edge_types.size(); ++i)
    {
        int edge_type = path_info.edge_types[i];
        int target_node_type = path_info.node_types[i + 1];
        int timestamp = path_info.edge_timestamps[i];
        pattern.steps.emplace_back(edge_type, target_node_type, timestamp);
    }
    return pattern;
}

// --- KGE Scoring Functions ---
float score_transe(const std::vector<float> &h, const std::vector<float> &r, const std::vector<float> &t)
{
    float sum_sq = 0.0f;
    for (size_t i = 0; i < h.size(); ++i)
    {
        float diff = h[i] + r[i] - t[i];
        sum_sq += diff * diff;
    }
    return -std::sqrt(sum_sq);
}

float score_distmult(const std::vector<float> &h, const std::vector<float> &r, const std::vector<float> &t)
{
    float score = 0.0f;
    for (size_t i = 0; i < h.size(); ++i)
        score += h[i] * r[i] * t[i];
    return score;
}

float score_complex(const std::vector<float> &h, const std::vector<float> &r, const std::vector<float> &t)
{
    float score = 0.0f;
    size_t dim = h.size() / 2;
    for (size_t i = 0; i < dim; ++i)
    {
        score += (h[i] * r[i] - h[i + dim] * r[i + dim]) * t[i] + (h[i] * r[i + dim] + h[i + dim] * r[i]) * t[i + dim];
    }
    return score;
}

float score_rotate(const std::vector<float> &h, const std::vector<float> &r, const std::vector<float> &t)
{
    float sum_sq = 0.0f;
    size_t dim = h.size() / 2;
    for (size_t i = 0; i < dim; ++i)
    {
        float ho_re = h[i] * r[i] - h[i + dim] * r[i + dim];
        float ho_im = h[i] * r[i + dim] + h[i + dim] * r[i];
        float diff_re = ho_re - t[i];
        float diff_im = ho_im - t[i + dim];
        sum_sq += diff_re * diff_re + diff_im * diff_im;
    }
    return -std::sqrt(sum_sq);
}

// --- Tree-like Negative Sampling ---
// Generate negative samples that share prefixes with positive path but differ at last node
std::map<int, std::vector<std::pair<int, int>>> generate_tree_negatives(
    int u, int ts, int beam_width, const ShortestPathInfo &positive_path,
    const AdjacencyList &adj, const std::unordered_map<int, int> &node_to_type,
    const std::set<int> &global_neighbors_of_u,
    const EmbeddingMap &node_embeddings, const EmbeddingMap &relation_embeddings,
    const std::function<float(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &)> &score_func,
    const std::string &criteria,
    std::ofstream &log_stream)
{
    std::map<int, std::vector<std::pair<int, int>>> results; // prefix_length -> [(v', ts)]
    
    // For each prefix of the positive path
    for (int prefix_len = 1; prefix_len <= positive_path.hops; ++prefix_len) {
        // Get the current node at the end of this prefix
        int current_node = positive_path.nodes[prefix_len - 1];
        
        // Get the next edge type in the positive path
        if (prefix_len > positive_path.hops) break;
        int next_edge_type = positive_path.edge_types[prefix_len - 1];
        
        // Get the expected node type for the target
        int expected_node_type = positive_path.node_types[prefix_len];
        
        // Find all temporal neighbors of current_node via next_edge_type
        std::vector<std::tuple<float, int, int>> scored_candidates; // (score, node, timestamp)
        
        if (adj.count(current_node) && 
            adj.at(current_node).count(next_edge_type) &&
            adj.at(current_node).at(next_edge_type).count(expected_node_type)) {
            
            const auto &ts_to_neighbors = adj.at(current_node).at(next_edge_type).at(expected_node_type);
            
            // Find all neighbors connected before timestamp ts
            auto upper_it = ts_to_neighbors.lower_bound(ts);
            for (auto it = ts_to_neighbors.begin(); it != upper_it; ++it) {
                int candidate_node = it->second;
                int edge_timestamp = it->first;
                
                // Skip if this is the positive path node
                if (candidate_node == positive_path.nodes[prefix_len]) continue;
                
                // Skip if this node is ever connected to u (global neighbor check)
                if (global_neighbors_of_u.count(candidate_node)) continue;
                
                if (criteria == "score") {
                    // Score the triple (current_node, next_edge_type, candidate_node)
                    try {
                        const auto &h_emb = node_embeddings.at(current_node);
                        const auto &r_emb = relation_embeddings.at(next_edge_type);
                        const auto &t_emb = node_embeddings.at(candidate_node);
                        float score = score_func(h_emb, r_emb, t_emb);
                        scored_candidates.emplace_back(score, candidate_node, edge_timestamp);
                    } catch (const std::out_of_range &) {
                        // Skip if embeddings are missing
                        continue;
                    }
                } else { // criteria == "time"
                    // Use timestamp as score (higher timestamp = better score)
                    float timestamp_score = static_cast<float>(edge_timestamp);
                    scored_candidates.emplace_back(timestamp_score, candidate_node, edge_timestamp);
                }
            }
        }
        
        // Log if no candidates were found for this prefix length
        // if (scored_candidates.empty()) {
        //     log_stream << "[Info] " << getCurrentTimestamp() << " No negative paths found for (u=" << u 
        //               << ", ts=" << ts << ", prefix_len=" << prefix_len << ")" << std::endl;
        // }

        // Log if candidates were found for this prefix length
        // if (!scored_candidates.empty()) {
        //     log_stream << "[Info] " << getCurrentTimestamp() << " Found " << scored_candidates.size() 
        //                << " candidates for (u=" << u << ", ts=" << ts 
        //                << ", prefix_len=" << prefix_len << ") with edge type " << next_edge_type
        //                << " and expected node type " << expected_node_type << std::endl;
        // }
        
        // Sort by score (descending) and keep top beam_width
        std::sort(scored_candidates.begin(), scored_candidates.end(),
                  [](const auto &a, const auto &b) { return std::get<0>(a) > std::get<0>(b); });
        
        if (scored_candidates.size() > static_cast<size_t>(beam_width)) {
            scored_candidates.resize(beam_width);
        }
        
        // Store results for this prefix length
        std::vector<std::pair<int, int>> prefix_results;
        for (const auto &candidate : scored_candidates) {
            prefix_results.emplace_back(std::get<1>(candidate), std::get<2>(candidate));
        }
        
        if (!prefix_results.empty()) {
            results[prefix_len] = prefix_results;
        }
    }
    
    return results;
}

// --- Main Execution ---
int main(int argc, char *argv[])
{
    if (argc != 9) 
    {
        // Initial error to std::cerr as log_stream is not yet set up.
        std::cerr << "[Error] " << getCurrentTimestamp() << " Usage: " << argv[0] << " <dataset_name> <partition> <model_name> <beam_width> <storage_dir> <num_threads> <thread_id> <criteria>" << std::endl;
        return 1;
    }

    std::string dataset = argv[1];
    std::string partition = argv[2];
    std::string model_name = argv[3];
    int beam_width = std::stoi(argv[4]);
    std::string storage_dir = argv[5]; 
    int num_threads = std::stoi(argv[6]); 
    int thread_id = std::stoi(argv[7]);   
    std::string criteria = argv[8];  // New criteria argument   

    std::string log_file_name = storage_dir + "/" + dataset + "_preprocess_logs_" + std::to_string(num_threads) + "_" + std::to_string(thread_id) + ".txt";
    std::ofstream log_stream(log_file_name, std::ios_base::app);

    if (!log_stream.is_open()) {
        std::cerr << "[Error] " << getCurrentTimestamp() << " Failed to open log file: " << log_file_name << std::endl;
        return 1; // Cannot proceed without log file
    }

    log_stream << "[Info] " << getCurrentTimestamp() << " Starting preprocess for dataset: " << dataset << ", partition: " << partition << ", model: " << model_name 
              << ", beam: " << beam_width << ", storage: " << storage_dir << ", threads: " << num_threads << ", tid: " << thread_id << ", criteria: " << criteria << std::endl;

    // Validate criteria
    if (criteria != "score" && criteria != "time") {
        log_stream << "[Error] " << getCurrentTimestamp() << " Invalid criteria: " << criteria << ". Must be 'score' or 'time'." << std::endl;
        return 1;
    }

    // 1. Load Embeddings (only if criteria is "score")
    EmbeddingMap node_embeddings, relation_embeddings;
    if (criteria == "score") {
        std::string embed_prefix = storage_dir + "/" + model_name + "_" + dataset + "_" + partition;

        try
        {
            load_embeddings_from_file(embed_prefix + "_nodes.txt", node_embeddings, log_stream);
            load_embeddings_from_file(embed_prefix + "_relations.txt", relation_embeddings, log_stream);
        }
        catch (const std::exception &e)
        {
            log_stream << "[Error] " << getCurrentTimestamp() << " Error loading embeddings: " << e.what() << std::endl;
            return 1;
        }
    } else {
        log_stream << "[Info] " << getCurrentTimestamp() << " Skipping embedding loading for time-based criteria." << std::endl;
    }

    // 2. Load Graph Data and build global neighbors
    AdjacencyList adj;
    std::unordered_map<int, int> node_to_type; 
    std::unordered_map<int, std::set<int>> global_neighbors; // u -> all neighbors across all time
    std::vector<std::tuple<int, int, int, int>> queries; 
    std::string csv_path = storage_dir + "/" + dataset + "_edges.csv";
    std::ifstream csv_file(csv_path);
    std::string line;
    
    if (!csv_file.is_open()) {
        log_stream << "[Error] " << getCurrentTimestamp() << " Error: Could not open CSV file: " << csv_path << std::endl;
        return 1;
    }

    std::vector<std::string> col_names;
    if (std::getline(csv_file, line)) { 
        col_names = split_str(line, ',');
    } else {
        log_stream << "[Error] " << getCurrentTimestamp() << " Error: CSV file is empty or header is missing: " << csv_path << std::endl;
        csv_file.close();
        return 1;
    }

    log_stream << "[Info] " << getCurrentTimestamp() << " Building enhanced adjacency list with node types and global neighbors from " << csv_path << "..." << std::endl;
    int line_number = 1;
    std::unordered_map<std::string, int> split_code = {
        {"pre", 0},
        {"train", 1},
        {"valid", 2},
        {"test", 3}
    };
    while (std::getline(csv_file, line)) 
    {
        line_number++;
        std::vector<std::string> row_data_vec = split_str(line, ',');
        if (row_data_vec.size() != col_names.size()) {
            log_stream << "[Warning] " << getCurrentTimestamp() << " Skipping line " << line_number << " due to mismatched column count. Expected " << col_names.size() << ", got " << row_data_vec.size() << std::endl;
            continue;
        }

        std::unordered_map<std::string, std::string> row_map_data;
        for (size_t i = 0; i < col_names.size(); ++i) {
            row_map_data[col_names[i]] = row_data_vec[i];
        }

        try {
            if (row_map_data.count("label") && std::stoi(row_map_data["label"]) == 1)
            { 
                int u = std::stoi(row_map_data.at("u"));
                int v = std::stoi(row_map_data.at("v"));
                int ts_val = std::stoi(row_map_data.at("ts"));
                int edge_type = std::stoi(row_map_data.at("edge_type"));
                int u_type = std::stoi(row_map_data.at("u_type")); 
                int v_type = std::stoi(row_map_data.at("v_type")); 

                // Read v_pos if it exists but discard the information
                if (row_map_data.count("v_pos") > 0 && !row_map_data.at("v_pos").empty() && row_map_data.at("v_pos") != "None") {
                    // We read but don't use this value
                    /*int v_pos = std::stoi(row_map_data.at("v_pos"));*/
                }

                node_to_type[u] = u_type;
                node_to_type[v] = v_type;
                adj[u][edge_type][v_type].emplace(ts_val, v);
                
                // Build global neighbors (bidirectional)
                global_neighbors[u].insert(v);
                global_neighbors[v].insert(u);
            }

            if (row_map_data.count("split") && stoi(row_map_data.at("split")) == split_code[partition])
            { 
                int edge_id = std::stoi(row_map_data.at("edge_id"));
                if (edge_id % num_threads == thread_id) { 
                    queries.emplace_back(edge_id, 
                                         std::stoi(row_map_data.at("u")),
                                         std::stoi(row_map_data.at("v")), 
                                         std::stoi(row_map_data.at("ts")));
                }
            }
        } catch (const std::out_of_range& oor) {
            log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Missing expected column on line " << line_number << ". Error: " << oor.what() << std::endl;
        } catch (const std::invalid_argument& ia) {
            log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Invalid argument (e.g., non-integer value) on line " << line_number << ". Error: " << ia.what() << std::endl;
        }
    }
    csv_file.close();

    log_stream << "[Info] " << getCurrentTimestamp() << " Loaded enhanced graph with " << adj.size() << " source nodes and "
              << queries.size() << " queries for partition " << partition << "." << std::endl;
    log_stream << "[Info] " << getCurrentTimestamp() << " Populated " << node_to_type.size() << " node to type mappings." << std::endl;
    log_stream << "[Info] " << getCurrentTimestamp() << " Built global neighbors for " << global_neighbors.size() << " nodes." << std::endl;

    // 3. Load Shortest Paths
    std::string paths_file_txt = storage_dir + "/" + dataset + "_paths.txt"; 
    auto shortest_paths = load_shortest_paths(paths_file_txt, node_to_type, log_stream); 

    // 4. Select scoring function (only if criteria is "score")
    std::function<float(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &)> score_func;
    if (criteria == "score") {
        if (model_name == "transe")
            score_func = score_transe;
        else if (model_name == "distmult")
            score_func = score_distmult;
        else if (model_name == "complex")
            score_func = score_complex;
        else if (model_name == "rotate")
            score_func = score_rotate;
        else
        {
            log_stream << "[Error] " << getCurrentTimestamp() << " Unsupported model: " << model_name << std::endl;
            return 1;
        }
    }

    // 5. Run Tree-like Negative Sampling
    log_stream << "[Info] " << getCurrentTimestamp() << " Starting tree-like negative sampling with criteria: " << criteria << "..." << std::endl;
    std::map<int, std::map<int, std::vector<std::pair<int, int>>>> final_results; // eid -> {prefix_len -> [(v', ts)]}

    for (const auto &q : queries)
    {
        int eid = std::get<0>(q);
        int u_node = std::get<1>(q); 
        int v_node = std::get<2>(q); 
        int ts_val = std::get<3>(q); 

        if (shortest_paths.count(eid) == 0)
        {
            continue; 
        }

        const ShortestPathInfo& positive_path = shortest_paths.at(eid);
        if (positive_path.hops == 0) {
            continue; // Skip direct edges
        }

        // Get global neighbors of u_node
        std::set<int> global_neighbors_of_u;
        if (global_neighbors.count(u_node)) {
            global_neighbors_of_u = global_neighbors.at(u_node);
        }

        auto tree_negatives = generate_tree_negatives(u_node, ts_val, beam_width, positive_path, adj,
                                                     node_to_type, global_neighbors_of_u,
                                                     node_embeddings, relation_embeddings, score_func, criteria,
                                                     log_stream);
        
        if (!tree_negatives.empty())
        {
            final_results[eid] = tree_negatives;
            // log_stream << "[Info] " << getCurrentTimestamp() << " Found " << tree_negatives.size() 
            //            << " negative paths for edge ID " << eid << " (u=" << u_node << ", ts=" << ts_val << ")" << std::endl;
        }
        else
        {
            // log_stream << "[Info] " << getCurrentTimestamp() << " No negative paths found for any prefix length of edge ID " 
            //           << eid << " (u=" << u_node << ", ts=" << ts_val << ")" << std::endl;
        }
    }

    // 6. Save results to JSON with new tree-like structure
    std::cout << "{\n"; // Output to stdout
    bool first_entry = true;
    for (const auto &pair : final_results)
    {
        if (!first_entry)
            std::cout << ",\n"; // Output to stdout
        
        int eid = pair.first;
        const auto &prefix_results = pair.second;
        
        std::cout << "  \"" << eid << "\": {\n"; // Output to stdout
        
        bool first_prefix = true;
        for (const auto &prefix_pair : prefix_results) {
            if (!first_prefix)
                std::cout << ",\n"; // Output to stdout
            
            int prefix_len = prefix_pair.first;
            const auto &candidates = prefix_pair.second;
            
            std::cout << "    \"" << prefix_len << "\": [\n"; // Output to stdout
            
            for (size_t i = 0; i < candidates.size(); ++i) {
                int node_id = candidates[i].first;
                int timestamp = candidates[i].second;
                
                std::cout << "      [" << node_id << ", " << timestamp << "]";
                if (i < candidates.size() - 1) std::cout << ",";
                std::cout << "\n"; // Output to stdout
            }
            
            std::cout << "    ]"; // Output to stdout
            first_prefix = false;
        }
        std::cout << "\n  }"; // Output to stdout
        first_entry = false;
    }
    std::cout << "\n}\n"; // Output to stdout

    log_stream << "[Info] " << getCurrentTimestamp() << " Finished processing. Tree-like negative sampling results (" << criteria << "-based) output to stdout." << std::endl;
    log_stream.close();

    return 0;
}
