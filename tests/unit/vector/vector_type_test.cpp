// Tests for include/vector/vector_type.h
// Testing library/framework: Catch2 (linked via Catch2::Catch2WithMain)
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <utility>

#include <vector/vector_type.h>

using namespace EloqVec;

TEST_CASE("distance_metric_to_string returns expected tokens", "[vector][distance][to_string]") {
    CHECK(distance_metric_to_string(DistanceMetric::L2SQ) == "L2SQ");
    CHECK(distance_metric_to_string(DistanceMetric::IP)   == "IP");
    CHECK(distance_metric_to_string(DistanceMetric::COSINE) == "COSINE");
    CHECK(distance_metric_to_string(DistanceMetric::UNKNOWN) == "UNKNOWN");
}

TEST_CASE("string_to_distance_metric(std::string) parses exact tokens and alias; case-sensitive", "[vector][distance][from_string]") {
    // Exact tokens
    CHECK(string_to_distance_metric(std::string("L2SQ")) == DistanceMetric::L2SQ);
    CHECK(string_to_distance_metric(std::string("IP"))   == DistanceMetric::IP);
    CHECK(string_to_distance_metric(std::string("COSINE")) == DistanceMetric::COSINE);
    // Alias: "L2" -> L2SQ
    CHECK(string_to_distance_metric(std::string("L2")) == DistanceMetric::L2SQ);
    // Unknowns
    CHECK(string_to_distance_metric(std::string("")) == DistanceMetric::UNKNOWN);
    CHECK(string_to_distance_metric(std::string("EUCLIDEAN")) == DistanceMetric::UNKNOWN);
    // Case sensitivity (this overload does NOT upcase internally)
    CHECK(string_to_distance_metric(std::string("l2sq")) == DistanceMetric::UNKNOWN);
}

TEST_CASE("string_to_distance_metric(std::string_view) parses case-insensitively and supports alias", "[vector][distance][from_string_view]") {
    CHECK(string_to_distance_metric(std::string_view("l2sq")) == DistanceMetric::L2SQ);
    CHECK(string_to_distance_metric(std::string_view("L2"))   == DistanceMetric::L2SQ);
    CHECK(string_to_distance_metric(std::string_view("ip"))   == DistanceMetric::IP);
    CHECK(string_to_distance_metric(std::string_view("Cosine")) == DistanceMetric::COSINE);
    // Unknowns
    CHECK(string_to_distance_metric(std::string_view("")) == DistanceMetric::UNKNOWN);
    CHECK(string_to_distance_metric(std::string_view("chebyshev")) == DistanceMetric::UNKNOWN);
}

TEST_CASE("algorithm_to_string returns expected tokens", "[vector][algorithm][to_string]") {
    CHECK(algorithm_to_string(Algorithm::HNSW) == "HNSW");
    CHECK(algorithm_to_string(Algorithm::UNKNOWN) == "UNKNOWN");
}

TEST_CASE("string_to_algorithm(std::string_view) parses case-insensitively", "[vector][algorithm][from_string_view]") {
    CHECK(string_to_algorithm(std::string_view("HNSW")) == Algorithm::HNSW);
    CHECK(string_to_algorithm(std::string_view("hnsw")) == Algorithm::HNSW);
    CHECK(string_to_algorithm(std::string_view("HnSw")) == Algorithm::HNSW);
    CHECK(string_to_algorithm(std::string_view("other")) == Algorithm::UNKNOWN);
    CHECK(string_to_algorithm(std::string_view("")) == Algorithm::UNKNOWN);
}

TEST_CASE("IndexConfig default ctor initializes documented defaults", "[vector][config][defaults]") {
    IndexConfig cfg;
    CHECK(cfg.name == "");
    CHECK(cfg.dimension == static_cast<size_t>(0));
    CHECK(cfg.max_elements == static_cast<size_t>(1000000));
    CHECK(cfg.algorithm == Algorithm::HNSW);
    CHECK(cfg.distance_metric == DistanceMetric::L2SQ);
    CHECK(cfg.storage_path == "");
    CHECK(cfg.params.empty());
}

TEST_CASE("IndexConfig parameterized ctor sets fields and moves params", "[vector][config][ctor]") {
    std::string name = "my_index";
    size_t dim = 128;
    Algorithm alg = Algorithm::HNSW;
    DistanceMetric metric = DistanceMetric::IP;
    std::string storage = "/tmp/eloq/index";
    std::unordered_map<std::string, std::string> params;
    params.emplace("M", "64");
    params.emplace("efConstruction", "200");
    const auto params_copy = params; // snapshot for verification

    IndexConfig cfg(name, dim, alg, metric, storage, std::move(params));

    CHECK(cfg.name == name);
    CHECK(cfg.dimension == dim);
    CHECK(cfg.algorithm == alg);
    CHECK(cfg.distance_metric == metric);
    CHECK(cfg.storage_path == storage);
    REQUIRE(cfg.params.size() == params_copy.size());
    for (const auto& kv : params_copy) {
        auto it = cfg.params.find(kv.first);
        REQUIRE(it \!= cfg.params.end());
        CHECK(it->second == kv.second);
    }
}

TEST_CASE("SearchResult default ctor initializes empty containers", "[vector][search][result]") {
    SearchResult r;
    CHECK(r.ids.empty());
    CHECK(r.distances.empty());
    CHECK(r.vectors.empty());
}

TEST_CASE("SearchResult param ctor stores ids and distances", "[vector][search][result]") {
    std::vector<uint64_t> ids{1, 42, 77};
    std::vector<float> dists{0.1f, 0.5f, 1.2f};

    SearchResult r(std::move(ids), std::move(dists));

    REQUIRE(r.ids.size() == 3u);
    CHECK(r.ids[0] == 1u);
    CHECK(r.ids[1] == 42u);
    CHECK(r.ids[2] == 77u);

    REQUIRE(r.distances.size() == 3u);
    CHECK(r.distances[0] == Catch::Approx(0.1f));
    CHECK(r.distances[1] == Catch::Approx(0.5f));
    CHECK(r.distances[2] == Catch::Approx(1.2f));

    // 'vectors' remains default-empty
    CHECK(r.vectors.empty());
}

TEST_CASE("vector_index_meta_table_name_sv matches specified literal", "[vector][meta][name]") {
    constexpr std::string_view expected{"__vector_index_meta_table"};
    CHECK(vector_index_meta_table_name_sv == expected);
}

TEST_CASE("Parsing robustness: whitespace handling", "[vector][parsing][robustness]") {
    // The std::string_view overload does not trim whitespace
    CHECK(string_to_distance_metric(std::string_view("  L2  ")) == DistanceMetric::UNKNOWN);
    CHECK(string_to_distance_metric(std::string_view("\tIP")) == DistanceMetric::UNKNOWN);
    CHECK(string_to_distance_metric(std::string_view("IP")) == DistanceMetric::IP);

    CHECK(string_to_algorithm(std::string_view(" HNSW ")) == Algorithm::UNKNOWN);
    CHECK(string_to_algorithm(std::string_view("HNSW")) == Algorithm::HNSW);
}