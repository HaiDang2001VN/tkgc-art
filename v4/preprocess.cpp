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
};

struct MetaPathPattern
{
    std::vector<std::pair<int, int>> steps; // (edge_type, target_node_type)
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
    if (path_info.edge_types.size() != path_info.node_types.size() - 1)
    {
        log_stream << "[Error] " << getCurrentTimestamp() << " Warning: Mismatched edge_types and node_types sizes in EID path info when extracting meta-path." << std::endl;
        return pattern;
    }

    for (size_t i = 0; i < path_info.edge_types.size(); ++i)
    {
        int edge_type = path_info.edge_types[i];
        int target_node_type = path_info.node_types[i + 1]; 
        pattern.steps.emplace_back(edge_type, target_node_type);
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

// --- Meta-Path Guided Beam Search ---
std::vector<Path> beam_search_for_edge(
    int u, int ts, int beam_width, const MetaPathPattern &pattern,
    const AdjacencyList &adj, const std::unordered_map<int, int> &node_to_type,
    const std::set<int> &direct_neighbors,
    const EmbeddingMap &node_embeddings, const EmbeddingMap &relation_embeddings,
    const std::function<float(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &)> &score_func)
{
    std::vector<Path> current_beam;
    current_beam.push_back({u});

    int max_depth = static_cast<int>(pattern.steps.size());

    for (int depth = 0; depth < max_depth; ++depth)
    {
        std::vector<std::pair<float, Path>> scored_candidates;

        // Get expected edge type and target node type from meta-path pattern
        int expected_edge_type = pattern.steps[depth].first;
        int expected_node_type = pattern.steps[depth].second;

        for (const auto &path : current_beam)
        {
            int last_node = path.back();

            // Check if node exists in adjacency list
            if (adj.count(last_node) == 0)
                continue;

            // Check if expected edge type exists for this node
            if (adj.at(last_node).count(expected_edge_type) == 0)
                continue;

            // Check if expected neighbor type exists for this edge type
            if (adj.at(last_node).at(expected_edge_type).count(expected_node_type) == 0)
                continue;

            const auto &ts_to_neighbors = adj.at(last_node).at(expected_edge_type).at(expected_node_type);

            // Binary search for the latest valid timestamp
            auto upper_it = ts_to_neighbors.lower_bound(ts);

            // Iterate backwards from the latest valid timestamp
            for (auto rev_it = std::reverse_iterator(upper_it);
                 rev_it != ts_to_neighbors.rend(); ++rev_it)
            {

                int timestamp = rev_it->first;
                int neighbor_node = rev_it->second;

                // Skip direct neighbors for paths of depth > 0
                if (depth > 0 && direct_neighbors.count(neighbor_node))
                    continue;

                // Verify neighbor node type matches expected
                if (node_to_type.count(neighbor_node) == 0 ||
                    node_to_type.at(neighbor_node) != expected_node_type)
                    continue;

                // Create new path
                Path new_path = path;
                new_path.push_back(expected_edge_type);
                new_path.push_back(neighbor_node);

                // Score the path using KGE model
                try
                {
                    const auto &h_emb = node_embeddings.at(u);
                    const auto &r_emb = relation_embeddings.at(expected_edge_type);
                    const auto &t_emb = node_embeddings.at(neighbor_node);
                    float score = score_func(h_emb, r_emb, t_emb);
                    scored_candidates.push_back({score, new_path});
                }
                catch (const std::out_of_range &oor)
                {
                    // Skip path if embeddings are missing
                    continue;
                }
            }
        }

        if (scored_candidates.empty())
            break;

        // Sort by score (descending) and keep top beam_width candidates
        std::sort(scored_candidates.begin(), scored_candidates.end(),
                  [](const auto &a, const auto &b)
                  { return a.first > b.first; });

        if (scored_candidates.size() > beam_width)
        {
            scored_candidates.resize(beam_width);
        }

        // Update current beam
        current_beam.clear();
        for (const auto &scored_path : scored_candidates)
        {
            current_beam.push_back(scored_path.second);
        }
    }

    return current_beam;
}

// --- Main Execution ---
int main(int argc, char *argv[])
{
    if (argc != 8) 
    {
        // Initial error to std::cerr as log_stream is not yet set up.
        std::cerr << "[Error] " << getCurrentTimestamp() << " Usage: " << argv[0] << " <dataset_name> <partition> <model_name> <beam_width> <storage_dir> <num_threads> <thread_id>" << std::endl;
        return 1;
    }

    std::string dataset = argv[1];
    std::string partition = argv[2];
    std::string model_name = argv[3];
    int beam_width = std::stoi(argv[4]);
    std::string storage_dir = argv[5]; 
    int num_threads = std::stoi(argv[6]); 
    int thread_id = std::stoi(argv[7]);   

    std::string log_file_name = storage_dir + "/" + dataset + "_preprocess_logs_" + std::to_string(num_threads) + "_" + std::to_string(thread_id) + ".txt";
    std::ofstream log_stream(log_file_name, std::ios_base::app);

    if (!log_stream.is_open()) {
        std::cerr << "[Error] " << getCurrentTimestamp() << " Failed to open log file: " << log_file_name << std::endl;
        return 1; // Cannot proceed without log file
    }

    log_stream << "[Info] " << getCurrentTimestamp() << " Starting preprocess for dataset: " << dataset << ", partition: " << partition << ", model: " << model_name 
              << ", beam: " << beam_width << ", storage: " << storage_dir << ", threads: " << num_threads << ", tid: " << thread_id << std::endl;

    // 1. Load Embeddings
    EmbeddingMap node_embeddings, relation_embeddings;
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

    // 2. Load Graph Data
    AdjacencyList adj;
    std::unordered_map<int, int> node_to_type; 
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

    log_stream << "[Info] " << getCurrentTimestamp() << " Building enhanced adjacency list with node types from " << csv_path << "..." << std::endl;
    int line_number = 1; 
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

                node_to_type[u] = u_type;
                node_to_type[v] = v_type;
                adj[u][edge_type][v_type].emplace(ts_val, v);
            }

            if (row_map_data.count("split") && row_map_data.at("split") == partition)
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

    // 3. Load Shortest Paths
    std::string paths_file_txt = storage_dir + "/" + dataset + "_paths.txt"; 
    auto shortest_paths = load_shortest_paths(paths_file_txt, node_to_type, log_stream); 

    // 4. Select scoring function
    std::function<float(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &)> score_func;
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

    // 5. Run Meta-Path Guided Beam Search
    log_stream << "[Info] " << getCurrentTimestamp() << " Starting meta-path guided beam search..." << std::endl;
    std::map<int, std::vector<Path>> final_results;

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

        MetaPathPattern pattern = extract_meta_path_pattern(shortest_paths.at(eid), log_stream);
        if (pattern.steps.empty())
        {
            if (shortest_paths.at(eid).hops > 0) { 
                 log_stream << "[Warning] " << getCurrentTimestamp() << " Empty meta-path pattern for EID " << eid << " with " << shortest_paths.at(eid).hops << " hops. Skipping." << std::endl;
            }
            continue; 
        }

        std::set<int> direct_neighbors;
        if (adj.count(u_node))
        {
            for (const auto &edge_type_pair : adj.at(u_node))
            {
                for (const auto &neighbor_type_pair : edge_type_pair.second)
                {
                    for (const auto &ts_neighbor_pair : neighbor_type_pair.second)
                    {
                        if (ts_neighbor_pair.first < ts_val)
                        { 
                            direct_neighbors.insert(ts_neighbor_pair.second);
                        }
                    }
                }
            }
        }

        auto found_paths = beam_search_for_edge(u_node, ts_val, beam_width, pattern, adj,
                                          node_to_type, direct_neighbors,
                                          node_embeddings, relation_embeddings, score_func);
        if (!found_paths.empty())
        {
            final_results[eid] = found_paths;
        }
    }

    // 6. Save results to JSON (now to std::cout)
    std::cout << "{\n"; // Output to stdout
    bool first_entry = true;
    for (const auto &pair : final_results)
    {
        if (!first_entry)
            std::cout << ",\n"; // Output to stdout
        std::cout << "  \"" << pair.first << "\": [\n"; // Output to stdout
        for (size_t i = 0; i < pair.second.size(); ++i)
        {
            std::cout << "    ["; // Output to stdout
            for (size_t j = 0; j < pair.second[i].size(); ++j)
            {
                std::cout << pair.second[i][j] << (j == pair.second[i].size() - 1 ? "" : ", "); // Output to stdout
            }
            std::cout << "]" << (i == pair.second.size() - 1 ? "" : ","); // Output to stdout
            std::cout << "\n"; // Output to stdout
        }
        std::cout << "  ]"; // Output to stdout
        first_entry = false;
    }
    std::cout << "\n}\n"; // Output to stdout

    log_stream << "[Info] " << getCurrentTimestamp() << " Finished processing. Results output to stdout." << std::endl;
    log_stream.close();

    return 0;
}
