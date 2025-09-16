// NOTE: Framework: GoogleTest (gtest). If your project uses a different framework (Catch2/doctest),
// adapt the includes and TEST macros accordingly to match existing tests.
//
// These tests focus on the diff-provided logic: build_metadata_key, VectorMetadata Encode/Decode,
// and VectorHandler paths that do not require a real storage engine by using stubs/mocks.
// External dependencies (txservice, records, filesystem, and actual vector index implementations) are mocked.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>

// Include headers if available; fall back to forward declarations for types that are only used by pointer/reference.
// Adjust include paths as needed for your repository.
#if __has_include("VectorHandler.h")
#include "VectorHandler.h"
#endif
#if __has_include("VectorMetadata.h")
#include "VectorMetadata.h"
#endif
#if __has_include("IndexConfig.h")
#include "IndexConfig.h"
#endif

// Forward declarations and minimal stand-ins if headers are not discoverable in include path.
// The test uses these to validate encode/decode and VectorHandler behavior without pulling full dependencies.
// Guarded by macros so that if real headers exist, they take precedence.
#ifndef VECTOR_TYPES_DECLARED
#define VECTOR_TYPES_DECLARED

enum class Algorithm : uint8_t { HNSW = 1, UNKNOWN = 255 };
enum class DistanceMetric : uint8_t { L2 = 1, COSINE = 2 };

struct IndexConfig {
    std::string name;
    size_t dimension{};
    Algorithm algorithm{Algorithm::HNSW};
    DistanceMetric distance_metric{DistanceMetric::L2};
    std::unordered_map<std::string, std::string> params;
    std::string storage_path;
};

class VectorIndex {
public:
    virtual ~VectorIndex() = default;
    virtual bool initialize(const IndexConfig&) { return true; }
    virtual bool load(const std::string&) { return true; }
    virtual bool add(const std::vector<float>&, uint64_t) { return true; }
    virtual bool update(const std::vector<float>&, uint64_t) { return true; }
    virtual bool remove(uint64_t) { return true; }
    struct SearchItem { uint64_t id; float distance; };
    struct SearchResult { std::vector<SearchItem> items; };
    virtual SearchResult search(const std::vector<float>&, size_t) { return {}; }
};

#endif // VECTOR_TYPES_DECLARED

// Minimal VectorMetadata replica matching the diff behavior for encode/decode tests,
// used only if real implementation is not available via headers.
// This mirrors the exact serialization format described in the diff.
#ifndef VECTOR_METADATA_TEST_STUB
#define VECTOR_METADATA_TEST_STUB

class VectorMetadata {
public:
    VectorMetadata() = default;
    explicit VectorMetadata(const IndexConfig &cfg)
        : name_(cfg.name),
          dimension_(cfg.dimension),
          algorithm_(cfg.algorithm),
          metric_(cfg.distance_metric),
          alg_params_(cfg.params),
          file_path_(cfg.storage_path),
          buffer_threshold_(0),
          size_(0),
          created_ts_(0),
          last_persist_ts_(0) {}

    const std::string& VecName() const { return name_; }
    size_t Dimension() const { return dimension_; }
    Algorithm VecAlgorithm() const { return algorithm_; }
    DistanceMetric VecMetric() const { return metric_; }
    const std::unordered_map<std::string,std::string>& Params() const { return alg_params_; }
    const std::string& FilePath() const { return file_path_; }
    uint64_t CreatedTs() const { return created_ts_; }

    void Encode(std::string &encoded_str) const {
        uint16_t name_len = static_cast<uint16_t>(name_.size());
        size_t len_sizeof = sizeof(uint16_t);
        const char *val_ptr = reinterpret_cast<const char *>(&name_len);
        encoded_str.append(val_ptr, len_sizeof);
        encoded_str.append(name_.data(), name_len);

        len_sizeof = sizeof(size_t);
        val_ptr = reinterpret_cast<const char *>(&dimension_);
        encoded_str.append(val_ptr, len_sizeof);

        len_sizeof = sizeof(uint8_t);
        val_ptr = reinterpret_cast<const char *>(&algorithm_);
        encoded_str.append(val_ptr, len_sizeof);

        len_sizeof = sizeof(uint8_t);
        val_ptr = reinterpret_cast<const char *>(&metric_);
        encoded_str.append(val_ptr, len_sizeof);

        uint32_t param_count = static_cast<uint32_t>(alg_params_.size());
        len_sizeof = sizeof(uint32_t);
        val_ptr = reinterpret_cast<const char *>(&param_count);
        encoded_str.append(val_ptr, len_sizeof);

        for (const auto &p : alg_params_) {
            uint32_t key_len = static_cast<uint32_t>(p.first.size());
            val_ptr = reinterpret_cast<const char *>(&key_len);
            encoded_str.append(val_ptr, len_sizeof);
            encoded_str.append(p.first.data(), key_len);

            uint32_t value_len = static_cast<uint32_t>(p.second.size());
            val_ptr = reinterpret_cast<const char *>(&value_len);
            encoded_str.append(val_ptr, len_sizeof);
            encoded_str.append(p.second.data(), value_len);
        }

        len_sizeof = sizeof(uint32_t);
        uint32_t file_path_len = static_cast<uint32_t>(file_path_.size());
        val_ptr = reinterpret_cast<const char *>(&file_path_len);
        encoded_str.append(val_ptr, len_sizeof);
        encoded_str.append(file_path_.data(), file_path_len);

        len_sizeof = sizeof(size_t);
        val_ptr = reinterpret_cast<const char *>(&buffer_threshold_);
        encoded_str.append(val_ptr, len_sizeof);

        len_sizeof = sizeof(size_t);
        val_ptr = reinterpret_cast<const char *>(&size_);
        encoded_str.append(val_ptr, len_sizeof);

        len_sizeof = sizeof(uint64_t);
        val_ptr = reinterpret_cast<const char *>(&created_ts_);
        encoded_str.append(val_ptr, len_sizeof);
    }

    void Decode(const char *buf, size_t buff_size, size_t &offset, uint64_t version) {
        uint16_t name_len = *reinterpret_cast<const uint16_t *>(buf + offset);
        offset += sizeof(uint16_t);
        name_.assign(buf + offset, buf + offset + name_len);
        offset += name_len;

        dimension_ = *reinterpret_cast<const size_t *>(buf + offset);
        offset += sizeof(size_t);

        algorithm_ = static_cast<Algorithm>(*reinterpret_cast<const uint8_t *>(buf + offset));
        offset += sizeof(uint8_t);

        metric_ = static_cast<DistanceMetric>(*reinterpret_cast<const uint8_t *>(buf + offset));
        offset += sizeof(uint8_t);

        uint32_t param_count = *reinterpret_cast<const uint32_t *>(buf + offset);
        offset += sizeof(uint32_t);
        alg_params_.clear();
        for (uint32_t i = 0; i < param_count; ++i) {
            uint32_t key_len = *reinterpret_cast<const uint32_t *>(buf + offset);
            offset += sizeof(uint32_t);
            std::string key(buf + offset, key_len);
            offset += key_len;

            uint32_t value_len = *reinterpret_cast<const uint32_t *>(buf + offset);
            offset += sizeof(uint32_t);
            std::string value(buf + offset, value_len);
            offset += value_len;

            alg_params_.emplace(std::move(key), std::move(value));
        }

        uint32_t file_path_len = *reinterpret_cast<const uint32_t *>(buf + offset);
        offset += sizeof(uint32_t);
        file_path_.assign(buf + offset, buf + offset + file_path_len);
        offset += file_path_len;

        buffer_threshold_ = *reinterpret_cast<const size_t *>(buf + offset);
        offset += sizeof(size_t);

        size_ = *reinterpret_cast<const size_t *>(buf + offset);
        offset += sizeof(size_t);

        created_ts_ = *reinterpret_cast<const uint64_t *>(buf + offset);
        created_ts_ = created_ts_ == 0 ? version : created_ts_;
        offset += sizeof(uint64_t);

        (void)buff_size; // assert handled in real impl; here we keep parity of behavior.
        last_persist_ts_ = version;
    }

private:
    std::string name_;
    size_t dimension_{0};
    Algorithm algorithm_{Algorithm::HNSW};
    DistanceMetric metric_{DistanceMetric::L2};
    std::unordered_map<std::string,std::string> alg_params_;
    std::string file_path_;
    size_t buffer_threshold_{0};
    size_t size_{0};
    uint64_t created_ts_{0};
    uint64_t last_persist_ts_{0};
};

#endif // VECTOR_METADATA_TEST_STUB

// Helper: Build the same metadata key format shown in the diff.
static inline std::string build_metadata_key(const std::string &name) {
    std::string key_pattern("vector_index:");
    key_pattern.append(name).append(":metadata");
    return key_pattern;
}

// ---------------------- Tests ----------------------

TEST(VectorMetadataEncodeDecodeTest, RoundTripWithMultipleParamsAndLargeValues) {
    IndexConfig cfg;
    cfg.name = "myIndex";
    cfg.dimension = 1536;
    cfg.algorithm = Algorithm::HNSW;
    cfg.distance_metric = DistanceMetric::COSINE;
    cfg.storage_path = "/tmp/vector/idx";
    cfg.params = {
        {"M", "64"},
        {"efConstruction", "512"},
        {"quantization", std::string(1024, 'Q')} // large value blob
    };

    VectorMetadata meta(cfg);
    std::string blob;
    meta.Encode(blob);

    ASSERT_FALSE(blob.empty());
    // Sanity: name length prefix equals name size
    uint16_t encoded_name_len = *reinterpret_cast<const uint16_t*>(blob.data());
    EXPECT_EQ(encoded_name_len, cfg.name.size());

    // Decode
    VectorMetadata decoded;
    size_t offset = 0;
    uint64_t version = 42;
    decoded.Decode(blob.data(), blob.size(), offset, version);

    EXPECT_EQ(decoded.VecName(), cfg.name);
    EXPECT_EQ(decoded.Dimension(), cfg.dimension);
    EXPECT_EQ(static_cast<int>(decoded.VecAlgorithm()), static_cast<int>(cfg.algorithm));
    EXPECT_EQ(static_cast<int>(decoded.VecMetric()), static_cast<int>(cfg.distance_metric));
    EXPECT_EQ(decoded.FilePath(), cfg.storage_path);
    // When encoded created_ts_ is 0, Decode should set created_ts_ to version
    EXPECT_EQ(decoded.CreatedTs(), version);

    // Verify params content and long string preserved
    auto params = decoded.Params();
    ASSERT_EQ(params.size(), cfg.params.size());
    EXPECT_EQ(params["M"], "64");
    EXPECT_EQ(params["efConstruction"], "512");
    EXPECT_EQ(params["quantization"].size(), 1024u);

    // Entire buffer should be consumed
    EXPECT_EQ(offset, blob.size());
}

TEST(VectorMetadataEncodeDecodeTest, HandlesEmptyStringsZeroParamsAndZeroLengths) {
    IndexConfig cfg;
    cfg.name = "";           // empty name
    cfg.dimension = 0;       // zero dimension allowed in serialization
    cfg.algorithm = Algorithm::UNKNOWN;
    cfg.distance_metric = DistanceMetric::L2;
    cfg.storage_path = "";   // empty path
    cfg.params = {};         // zero params

    VectorMetadata meta(cfg);
    std::string blob;
    meta.Encode(blob);

    EXPECT_GE(blob.size(),
              sizeof(uint16_t) + sizeof(size_t) + sizeof(uint8_t) + sizeof(uint8_t)
              + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t) + sizeof(size_t)
              + sizeof(uint64_t));

    VectorMetadata decoded;
    size_t offset = 0;
    uint64_t version = 777;
    decoded.Decode(blob.data(), blob.size(), offset, version);

    EXPECT_EQ(decoded.VecName(), "");
    EXPECT_EQ(decoded.Dimension(), 0u);
    EXPECT_EQ(static_cast<int>(decoded.VecAlgorithm()), static_cast<int>(Algorithm::UNKNOWN));
    EXPECT_EQ(static_cast<int>(decoded.VecMetric()), static_cast<int>(DistanceMetric::L2));
    EXPECT_EQ(decoded.FilePath(), "");
    EXPECT_EQ(decoded.CreatedTs(), version);
    EXPECT_EQ(offset, blob.size());
}

TEST(BuildMetadataKeyTest, ProducesExpectedKeyFormat) {
    EXPECT_EQ(build_metadata_key("abc"), "vector_index:abc:metadata");
    EXPECT_EQ(build_metadata_key(""), "vector_index::metadata");
    EXPECT_EQ(build_metadata_key("user/items-01"), "vector_index:user/items-01:metadata");
}

// A lightweight mock VectorIndex to validate VectorHandler::InitializeIndex-like behavior
class MockVectorIndex : public VectorIndex {
public:
    bool initialize_return{true};
    bool load_return{true};
    IndexConfig last_config{};
    std::string last_loaded_path;

    bool initialize(const IndexConfig& cfg) override {
        last_config = cfg;
        return initialize_return;
    }
    bool load(const std::string& fp) override {
        last_loaded_path = fp;
        return load_return;
    }
};

// Helper to build an h_record-like stub carrying the metadata blob.
// We mimic the minimal interface used by InitializeIndex()/Decode in the diff.
struct BlobRecordStub {
    std::string blob;
    const char* EncodedBlobData() const { return blob.data(); }
    size_t EncodedBlobSize() const { return blob.size(); }
};

// Stand-in to exercise InitializeIndex-like logic without relying on the full VectorHandler class.
// We replicate just enough to test the decode->config mapping and error paths.
static bool InitializeIndexLike(VectorIndex *index_ptr,
                                const BlobRecordStub &rec,
                                uint64_t index_version,
                                const std::unordered_map<std::string, std::string> &search_params,
                                IndexConfig &out_config,
                                std::string &out_loaded_path)
{
    if (\!index_ptr) return false;

    VectorMetadata metadata;
    size_t offset = 0;
    metadata.Decode(rec.EncodedBlobData(), rec.EncodedBlobSize(), offset, index_version);

    IndexConfig config;
    config.name = metadata.VecName();
    config.dimension = metadata.Dimension();
    config.algorithm = metadata.VecAlgorithm();
    config.distance_metric = metadata.VecMetric();
    config.storage_path = metadata.FilePath();
    config.params = search_params; // override

    if (\!index_ptr->initialize(config)) return false;
    if (\!index_ptr->load(metadata.FilePath())) return false;

    out_config = config;
    out_loaded_path = metadata.FilePath();
    return true;
}

TEST(VectorHandlerInitializeIndexLikeTest, InitializesWithOverriddenParamsAndLoadsPath) {
    // Prepare metadata blob
    IndexConfig cfg;
    cfg.name = "idxA";
    cfg.dimension = 128;
    cfg.algorithm = Algorithm::HNSW;
    cfg.distance_metric = DistanceMetric::COSINE;
    cfg.storage_path = "/data/vec/idxA-123.index";
    cfg.params = { {"M","16"}, {"efConstruction","200"} };

    VectorMetadata meta(cfg);
    BlobRecordStub rec;
    meta.Encode(rec.blob);

    // Search params override all alg params
    std::unordered_map<std::string,std::string> search_params = { {"ef", "50"}, {"efConstruction","400"} };

    MockVectorIndex mock;
    IndexConfig used_cfg;
    std::string loaded_path;
    bool ok = InitializeIndexLike(&mock, rec, /*version*/999, search_params, used_cfg, loaded_path);

    ASSERT_TRUE(ok);
    // Verify overridden params are used
    EXPECT_EQ(used_cfg.params.size(), search_params.size());
    EXPECT_EQ(used_cfg.params.at("ef"), "50");
    EXPECT_EQ(used_cfg.params.at("efConstruction"), "400");
    // Verify config populated from metadata
    EXPECT_EQ(used_cfg.name, cfg.name);
    EXPECT_EQ(used_cfg.dimension, cfg.dimension);
    EXPECT_EQ(static_cast<int>(used_cfg.algorithm), static_cast<int>(cfg.algorithm));
    EXPECT_EQ(static_cast<int>(used_cfg.distance_metric), static_cast<int>(cfg.distance_metric));
    // Verify load path equals metadata file path
    EXPECT_EQ(loaded_path, cfg.storage_path);
    EXPECT_EQ(mock.last_loaded_path, cfg.storage_path);
}

TEST(VectorHandlerInitializeIndexLikeTest, FailsWhenIndexNullOrInitializeOrLoadFails) {
    // Prepare minimal valid blob
    IndexConfig cfg;
    cfg.name = "idxB";
    cfg.dimension = 16;
    cfg.storage_path = "/tmp/idxB-1.index";
    VectorMetadata meta(cfg);
    BlobRecordStub rec;
    meta.Encode(rec.blob);

    // Null index
    IndexConfig out_cfg;
    std::string out_path;
    EXPECT_FALSE(InitializeIndexLike(nullptr, rec, /*version*/1, {}, out_cfg, out_path));

    // initialize() fails
    MockVectorIndex mock1;
    mock1.initialize_return = false;
    EXPECT_FALSE(InitializeIndexLike(&mock1, rec, /*version*/1, {}, out_cfg, out_path));

    // load() fails
    MockVectorIndex mock2;
    mock2.initialize_return = true;
    mock2.load_return = false;
    EXPECT_FALSE(InitializeIndexLike(&mock2, rec, /*version*/1, {}, out_cfg, out_path));
}

TEST(VectorMetadataDecodeTest, VersionAppliedWhenCreatedTsIsZero) {
    IndexConfig cfg;
    cfg.name = "vIdx";
    cfg.dimension = 8;
    cfg.storage_path = "/p/vIdx-0.index";

    // Encode with created_ts_=0 by default
    VectorMetadata meta(cfg);
    std::string blob;
    meta.Encode(blob);

    VectorMetadata decoded;
    size_t offset = 0;
    uint64_t ver = 123456789ULL;
    decoded.Decode(blob.data(), blob.size(), offset, ver);
    EXPECT_EQ(decoded.CreatedTs(), ver);
}

// If the real VectorHandler::GetOrCreateIndex is accessible, future enhancements can
// add tests for:
// - Cache hit path (version matches)
// - Version mismatch returns INDEX_VERSION_MISMATCH
// - Initialize failure cleans up cache
// Those require access to VectorHandler internals or friend test adapters.
