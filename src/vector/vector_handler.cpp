/**
 * @file vector_handler.cpp
 * @brief Implementation of vector operations handler for EloqVec
 *
 * This file implements the vector_handler class methods for vector index
 * operations including create, search, info, add, and drop operations.
 *
 * @author EloqData Inc.
 * @date 2025
 */

#include "vector_handler.h"

#include <algorithm>
#include <filesystem>
#include <utility>

#include "hnsw_vector_index.h"
#include "tx_request.h"
#include "tx_util.h"

namespace EloqVec
{
// Using declarations instead of using-directive
using txservice::AbortTx;
using txservice::CommitTx;
using txservice::EloqStringKey;
using txservice::EloqStringRecord;
using txservice::LocalCcShards;
using txservice::OperationType;
using txservice::ReadTxRequest;
using txservice::RecordStatus;
using txservice::TransactionExecution;
using txservice::TxErrorCode;
using txservice::TxKey;
using txservice::TxRecord;

inline std::string build_metadata_key(const std::string &name)
{
    std::string key_pattern("vector_index:");
    key_pattern.append(name).append(":metadata");
    return key_pattern;
}

VectorMetadata::VectorMetadata(const IndexConfig &vec_spec)
    : name_(vec_spec.name),
      dimension_(vec_spec.dimension),
      algorithm_(vec_spec.algorithm),
      metric_(vec_spec.distance_metric),
      alg_params_(vec_spec.params),
      file_path_(vec_spec.storage_path)
{
    uint64_t ts = LocalCcShards::ClockTs();
    file_path_.append("/")
        .append(name_)
        .append("-")
        .append(std::to_string(ts))
        .append(".index");
    created_ts_ = 0;
    last_persist_ts_ = 0;
}

void VectorMetadata::Encode(std::string &encoded_str) const
{
    /**
     * The format of the encoded metadata:
     * nameLen | name | dimension | algorithm | metric | paramCount | key1Len |
     * key1 | value1Len | value1 | ... | filePathLen | filePath |
     * bufferThreshold | size | createdTs
     * 1. nameLen is a 2-byte integer representing the length of the name. 2.
     * name is a string. 3. dimension is a 8-byte integer representing the
     * dimension. 4. algorithm is a 1-byte integer representing the
     * algorithm. 5. metric is a 1-byte integer representing the metric. 6.
     * paramCount is a 4-byte integer representing the number of algorithm
     * parameters. 7. For each parameter: keyLen (4-byte) | key (string) |
     * valueLen (4-byte) | value (string). 8. filePathLen is a 4-byte integer
     * representing the length of the file path. 9. filePath is a string. 10.
     * bufferThreshold is a 8-byte integer representing the buffer
     * threshold. 11. size is a 8-byte integer representing the size of the
     * index. 12. createdTs is a 8-byte integer representing the creation
     * timestamp.
     */
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

    // Serialize algorithm parameters with the following format:
    // param_count | key1_len | key1 | value1_len | value1 | key2_len | key2 |
    // value2_len | value2 | ...
    uint32_t param_count = static_cast<uint32_t>(alg_params_.size());
    len_sizeof = sizeof(uint32_t);
    val_ptr = reinterpret_cast<const char *>(&param_count);
    encoded_str.append(val_ptr, len_sizeof);

    // Serialize each key-value pair
    for (const auto &param : alg_params_)
    {
        // Write key length and key
        uint32_t key_len = static_cast<uint32_t>(param.first.size());
        val_ptr = reinterpret_cast<const char *>(&key_len);
        encoded_str.append(val_ptr, len_sizeof);
        encoded_str.append(param.first.data(), key_len);

        // Write value length and value
        uint32_t value_len = static_cast<uint32_t>(param.second.size());
        val_ptr = reinterpret_cast<const char *>(&value_len);
        encoded_str.append(val_ptr, len_sizeof);
        encoded_str.append(param.second.data(), value_len);
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

void VectorMetadata::Decode(const char *buf,
                            size_t buff_size,
                            size_t &offset,
                            uint64_t version)
{
    uint16_t name_len = *reinterpret_cast<const uint16_t *>(buf + offset);
    offset += sizeof(uint16_t);
    name_.clear();
    name_.reserve(name_len);
    std::copy(buf + offset, buf + offset + name_len, std::back_inserter(name_));
    offset += name_len;

    dimension_ = *reinterpret_cast<const size_t *>(buf + offset);
    offset += sizeof(size_t);

    algorithm_ = static_cast<Algorithm>(
        *reinterpret_cast<const uint8_t *>(buf + offset));
    offset += sizeof(uint8_t);

    metric_ = static_cast<DistanceMetric>(
        *reinterpret_cast<const uint8_t *>(buf + offset));
    offset += sizeof(uint8_t);

    uint32_t param_count = *reinterpret_cast<const uint32_t *>(buf + offset);
    offset += sizeof(uint32_t);
    // Clear existing parameters
    alg_params_.clear();
    // Deserialize each key-value pair
    for (uint32_t i = 0; i < param_count; ++i)
    {
        // Read key length and key
        uint32_t key_len = *reinterpret_cast<const uint32_t *>(buf + offset);
        offset += sizeof(uint32_t);
        std::string key(buf + offset, key_len);
        offset += key_len;

        // Read value length and value
        uint32_t value_len = *reinterpret_cast<const uint32_t *>(buf + offset);
        offset += sizeof(uint32_t);
        std::string value(buf + offset, value_len);
        offset += value_len;

        // Store the key-value pair
        alg_params_.emplace(std::move(key), std::move(value));
    }

    uint32_t file_path_len = *reinterpret_cast<const uint32_t *>(buf + offset);
    offset += sizeof(uint32_t);
    file_path_.clear();
    file_path_.reserve(file_path_len);
    std::copy(buf + offset,
              buf + offset + file_path_len,
              std::back_inserter(file_path_));
    offset += file_path_len;

    buffer_threshold_ = *reinterpret_cast<const size_t *>(buf + offset);
    offset += sizeof(size_t);

    size_ = *reinterpret_cast<const size_t *>(buf + offset);
    offset += sizeof(size_t);

    created_ts_ = *reinterpret_cast<const uint64_t *>(buf + offset);
    created_ts_ = created_ts_ == 0 ? version : created_ts_;
    offset += sizeof(uint64_t);

    last_persist_ts_ = version;
    assert(offset == buff_size);
}

VectorHandler &VectorHandler::Instance()
{
    static VectorHandler instance;
    return instance;
}

/**
 * @brief Create a new vector index metadata entry and prepare an in-memory index placeholder.
 *
 * Attempts to insert index metadata for the index described by idx_spec into the internal
 * metadata table. If insertion succeeds, creates and caches an uninitialized in-memory
 * HNSWVectorIndex instance (version 0) for later initialization/loading.
 *
 * @param idx_spec Configuration for the index to create (name, dimension, algorithm, metric, params, etc.).
 * @return VectorOpResult
 *   - SUCCEED: metadata inserted and in-memory index placeholder created.
 *   - INDEX_EXISTED: an index with the same name already exists (metadata status Normal).
 *   - INDEX_META_OP_FAILED: an error occurred while reading or upserting metadata.
 */
VectorOpResult VectorHandler::Create(const IndexConfig &idx_spec,
                                     txservice::TransactionExecution *txm)
{
    // For the internal table, there is no need to acquire read lock on catalog.
    std::string h_key = build_metadata_key(idx_spec.name);
    TxKey tx_key = EloqStringKey::Create(h_key.c_str(), h_key.size());
    TxRecord::Uptr h_record = EloqStringRecord::Create();
    ReadTxRequest read_req(&vector_index_meta_table,
                           0,
                           &tx_key,
                           h_record.get(),
                           true,  /* is_for_write */
                           false, /* is_for_share */
                           false, /* read_local */
                           0,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr,
                           txm);
    txm->Execute(&read_req);
    read_req.Wait();
    if (read_req.IsError())
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    if (read_req.Result().first == RecordStatus::Normal)
    {
        // The index already exists
        return VectorOpResult::INDEX_EXISTED;
    }

    // The index does not exist, create it
    // 1. encode the index metadata
    VectorMetadata vec_meta(idx_spec);
    std::string encoded_str;
    vec_meta.Encode(encoded_str);
    h_record->SetEncodedBlob(
        reinterpret_cast<const unsigned char *>(encoded_str.data()),
        encoded_str.size());
    // 2. upsert the metadata to the internal table
    // There is no need to check the schema ts of the internal table.
    TxErrorCode err_code = txm->TxUpsert(vector_index_meta_table,
                                         0,
                                         std::move(tx_key),
                                         std::move(h_record),
                                         OperationType::Insert);
    if (err_code != TxErrorCode::NO_ERROR)
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // Create the vector index object in memory and add to cache
    {
        std::lock_guard<std::shared_mutex> lock(vec_indexes_mutex_);
        auto hnsw_index = std::make_unique<HNSWVectorIndex>();
        auto emplace_pair = vec_indexes_.try_emplace(
            idx_spec.name, std::make_pair(0, std::move(hnsw_index)));
        assert(emplace_pair.second);
    }

    return VectorOpResult::SUCCEED;
}

/**
 * @brief Removes a vector index and its metadata.
 *
 * Deletes the index metadata entry from the internal metadata table, removes the
 * in-memory cached index, and attempts to delete the on-disk index file (if
 * present). The function reads and decodes the stored metadata first to obtain
 * the on-disk file path, performs a transactional delete of the metadata, and
 * then evicts the index from the in-memory cache and removes the file from
 * disk.
 *
 * Behavior and error conditions:
 * - Returns INDEX_NOT_EXIST if the metadata read indicates the index is deleted
 *   or absent.
 * - Returns INDEX_META_OP_FAILED if reading or upserting (deleting) the metadata
 *   entry fails.
 * - On successful metadata deletion returns SUCCEED. Filesystem deletion errors
 *   are logged as warnings and do not change the returned success status.
 *
 * Side effects:
 * - Removes the metadata entry from the internal metadata table via a
 *   transactional upsert(Delete).
 * - Erases the corresponding entry from the in-memory index cache.
 * - Attempts to remove the physical index file referenced by the metadata.
 */
VectorOpResult VectorHandler::Drop(const std::string &name,
                                   txservice::TransactionExecution *txm)
{
    // 1. Check if the index exists by reading from vector_index_meta_table
    std::string h_key = build_metadata_key(name);
    TxKey tx_key = EloqStringKey::Create(h_key.c_str(), h_key.size());
    TxRecord::Uptr h_record = EloqStringRecord::Create();
    ReadTxRequest read_req(&vector_index_meta_table,
                           0,
                           &tx_key,
                           h_record.get(),
                           true,  /* is_for_write */
                           false, /* is_for_share */
                           false, /* read_local */
                           0,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr,
                           txm);
    txm->Execute(&read_req);
    read_req.Wait();
    if (read_req.IsError())
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // 2. Check if the index exists
    if (read_req.Result().first != RecordStatus::Normal)
    {
        assert(read_req.Result().first == RecordStatus::Deleted);
        // The index does not exist
        return VectorOpResult::INDEX_NOT_EXIST;
    }

    // 3. Decode metadata to get file path before deletion
    uint64_t index_version = read_req.Result().second;
    VectorMetadata metadata;
    const char *blob_data = h_record->EncodedBlobData();
    size_t blob_size = h_record->EncodedBlobSize();
    size_t offset = 0;
    metadata.Decode(blob_data, blob_size, offset, index_version);
    const std::string &file_path = metadata.FilePath();

    // 4. The index exists, delete it using OperationType::Delete
    TxErrorCode err_code = txm->TxUpsert(vector_index_meta_table,
                                         0,
                                         std::move(tx_key),
                                         nullptr,
                                         OperationType::Delete);
    if (err_code != TxErrorCode::NO_ERROR)
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // 5. Delete the index file and the vector index object in memory
    {
        std::lock_guard<std::shared_mutex> lock(vec_indexes_mutex_);
        // Remove from cache
        vec_indexes_.erase(name);
    }

    // Delete the physical file
    // TODO(ysw): May do this after the transaction is committed.
    if (!file_path.empty() && std::filesystem::exists(file_path))
    {
        try
        {
            std::filesystem::remove(file_path);
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            LOG(WARNING) << "Failed to delete index file: " << file_path
                         << ", error: " << e.what();
        }
    }

    return VectorOpResult::SUCCEED;
}

VectorOpResult VectorHandler::Info(const std::string &name,
                                   txservice::TransactionExecution *txm,
                                   VectorMetadata &metadata)
{
    // 1. Check if the index exists by reading from vector_index_meta_table
    std::string h_key = build_metadata_key(name);
    TxKey tx_key = EloqStringKey::Create(h_key.c_str(), h_key.size());
    TxRecord::Uptr h_record = EloqStringRecord::Create();
    ReadTxRequest read_req(&vector_index_meta_table,
                           0,
                           &tx_key,
                           h_record.get(),
                           false, /* is_for_write */
                           false, /* is_for_share */
                           false, /* read_local */
                           0,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr,
                           txm);
    txm->Execute(&read_req);
    read_req.Wait();
    if (read_req.IsError())
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // 2. Check if the index exists
    if (read_req.Result().first != RecordStatus::Normal)
    {
        assert(read_req.Result().first == RecordStatus::Deleted);
        // The index does not exist
        return VectorOpResult::INDEX_NOT_EXIST;
    }

    uint64_t index_version = read_req.Result().second;

    // 3. The index exists, decode the metadata
    const char *blob_data = h_record->EncodedBlobData();
    size_t blob_size = h_record->EncodedBlobSize();
    size_t offset = 0;
    metadata.Decode(blob_data, blob_size, offset, index_version);

    return VectorOpResult::SUCCEED;
}

/**
 * @brief Execute a k-nearest-neighbors search against a named vector index.
 *
 * Verifies the index exists by reading its metadata, obtains or initializes the
 * in-memory index (creating and loading it if needed), and runs a nearest-neighbor
 * search using the provided query vector.
 *
 * On success, populates `vector_result` with up to `k_count` nearest entries and
 * returns VectorOpResult::SUCCEED. If metadata cannot be read, the index is
 * missing, or index initialization/search fails, an appropriate non-success
 * VectorOpResult is returned.
 *
 * @param name Index name to search.
 * @param query_vector Query vector used for nearest-neighbor retrieval.
 * @param k_count Maximum number of nearest neighbors to return.
 * @param search_params Optional parameters that can influence index initialization/search.
 * @param txm Transaction execution context used to read index metadata. (service client — no further description)
 * @param[out] vector_result Receives the search results when the call succeeds.
 */
VectorOpResult VectorHandler::Search(
    const std::string &name,
    const std::vector<float> &query_vector,
    size_t k_count,
    const std::unordered_map<std::string, std::string> &search_params,
    txservice::TransactionExecution *txm,
    SearchResult &vector_result)
{
    // 1. Check if the index exists by reading from vector_index_meta_table
    std::string h_key = build_metadata_key(name);
    TxKey tx_key = EloqStringKey::Create(h_key.c_str(), h_key.size());
    TxRecord::Uptr h_record = EloqStringRecord::Create();
    ReadTxRequest read_req(&vector_index_meta_table,
                           0,
                           &tx_key,
                           h_record.get(),
                           false, /* is_for_write */
                           false, /* is_for_share */
                           false, /* read_local */
                           0,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr,
                           txm);
    txm->Execute(&read_req);
    read_req.Wait();
    if (read_req.IsError())
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // 2. Check if the index exists
    if (read_req.Result().first != RecordStatus::Normal)
    {
        assert(read_req.Result().first == RecordStatus::Deleted);
        // The index does not exist
        return VectorOpResult::INDEX_NOT_EXIST;
    }

    uint64_t index_version = read_req.Result().second;

    // 3. Get or create the vector index from cache
    VectorIndex *index_ptr = nullptr;
    // TODO(ysw): search parames ??
    VectorOpResult get_result = GetOrCreateIndex(
        name, h_record, index_version, search_params, index_ptr);
    if (get_result != VectorOpResult::SUCCEED)
    {
        return get_result;
    }

    // 4. Perform vector search
    vector_result = index_ptr->search(query_vector, k_count);
    return VectorOpResult::SUCCEED;
}

/**
 * @brief Adds a vector with a given id to the named vector index.
 *
 * Attempts to validate that the index exists, obtains or initializes the in-memory
 * index instance for the specified index version, and inserts the provided vector
 * under the given id.
 *
 * The provided vector must match the index's configured dimensionality. The function
 * returns a VectorOpResult describing the outcome:
 * - SUCCEED: vector was added successfully.
 * - INDEX_NOT_EXIST: the named index does not exist.
 * - INDEX_META_OP_FAILED: failure reading index metadata.
 * - INDEX_ADD_FAILED: index instance rejected the add operation.
 *
 * @param name Index name.
 * @param id   Numeric identifier to associate with the vector.
 * @param vector Dense vector to insert; dimensionality must match the index.
 *
 * Note: The `txm` transaction execution pointer is a transaction/service parameter
 * and is intentionally not documented here.
 */
VectorOpResult VectorHandler::Add(const std::string &name,
                                  uint64_t id,
                                  const std::vector<float> &vector,
                                  txservice::TransactionExecution *txm)
{
    // 1. Check if the index exists by reading from vector_index_meta_table
    std::string h_key = build_metadata_key(name);
    TxKey tx_key = EloqStringKey::Create(h_key.c_str(), h_key.size());
    TxRecord::Uptr h_record = EloqStringRecord::Create();
    ReadTxRequest read_req(&vector_index_meta_table,
                           0,
                           &tx_key,
                           h_record.get(),
                           false, /* is_for_write */
                           false, /* is_for_share */
                           false, /* read_local */
                           0,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr,
                           txm);
    txm->Execute(&read_req);
    read_req.Wait();
    if (read_req.IsError())
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // 2. Check if the index exists
    if (read_req.Result().first != RecordStatus::Normal)
    {
        assert(read_req.Result().first == RecordStatus::Deleted);
        // The index does not exist
        return VectorOpResult::INDEX_NOT_EXIST;
    }

    uint64_t index_version = read_req.Result().second;

    // 3. Get or create the vector index from cache
    VectorIndex *index_ptr = nullptr;
    VectorOpResult get_result =
        GetOrCreateIndex(name, h_record, index_version, {}, index_ptr);
    if (get_result != VectorOpResult::SUCCEED)
    {
        return get_result;
    }

    // TODO(ysw): Add item to log object.

    // 4. Add the vector to the index
    if (!index_ptr->add(vector, id))
    {
        return VectorOpResult::INDEX_ADD_FAILED;
    }
    return VectorOpResult::SUCCEED;
}

/**
 * @brief Update an existing vector in the named vector index.
 *
 * Reads index metadata, ensures the index exists (initializing or loading the in-memory index if needed),
 * and updates the stored vector for the given id in that index.
 *
 * @param name Name of the vector index to update.
 * @param id Identifier of the vector to update.
 * @param vector New feature vector to store for the given id.
 * @param txm Transaction execution context used to read index metadata. (Client/service parameter — omitted from @param list per project convention.)
 *
 * @return VectorOpResult indicating the operation outcome:
 *         - SUCCEED on successful update.
 *         - INDEX_NOT_EXIST if the index metadata indicates the index is absent.
 *         - INDEX_META_OP_FAILED if reading index metadata failed.
 *         - INDEX_UPDATE_FAILED if the index update operation failed.
 *         - Any error returned by GetOrCreateIndex (e.g., initialization/load failures).
 */
VectorOpResult VectorHandler::Update(const std::string &name,
                                     uint64_t id,
                                     const std::vector<float> &vector,
                                     txservice::TransactionExecution *txm)
{
    // 1. Check if the index exists by reading from vector_index_meta_table
    std::string h_key = build_metadata_key(name);
    TxKey tx_key = EloqStringKey::Create(h_key.c_str(), h_key.size());
    TxRecord::Uptr h_record = EloqStringRecord::Create();
    ReadTxRequest read_req(&vector_index_meta_table,
                           0,
                           &tx_key,
                           h_record.get(),
                           false, /* is_for_write */
                           false, /* is_for_share */
                           false, /* read_local */
                           0,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr,
                           txm);
    txm->Execute(&read_req);
    read_req.Wait();
    if (read_req.IsError())
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // 2. Check if the index exists
    if (read_req.Result().first != RecordStatus::Normal)
    {
        assert(read_req.Result().first == RecordStatus::Deleted);
        // The index does not exist
        return VectorOpResult::INDEX_NOT_EXIST;
    }

    uint64_t index_version = read_req.Result().second;

    // 3. Get or create the vector index from cache
    VectorIndex *index_ptr = nullptr;
    VectorOpResult get_result =
        GetOrCreateIndex(name, h_record, index_version, {}, index_ptr);
    if (get_result != VectorOpResult::SUCCEED)
    {
        return get_result;
    }

    // TODO(ysw): Update item to log object.

    // 4. Update the vector in the index
    if (!index_ptr->update(vector, id))
    {
        return VectorOpResult::INDEX_UPDATE_FAILED;
    }
    return VectorOpResult::SUCCEED;
}

/**
 * @brief Remove a vector entry from a named index.
 *
 * Reads the index metadata to validate the index exists, ensures an
 * in-memory index instance is available (creating/initializing it if
 * necessary), and removes the vector identified by `id` from that index.
 *
 * The function returns a VectorOpResult indicating success or a specific
 * failure reason:
 * - SUCCEED: vector was removed successfully.
 * - INDEX_META_OP_FAILED: internal metadata read operation failed.
 * - INDEX_NOT_EXIST: the named index does not exist or is marked deleted.
 * - INDEX_DELETE_FAILED: the in-memory index failed to remove the given id.
 *
 * @param name Name of the vector index.
 * @param id   Identifier of the vector to remove.
 * @param txm  TransactionExecution used for internal metadata reads (omitted from param
 *             documentation as a general transaction service).
 */
VectorOpResult VectorHandler::Delete(const std::string &name,
                                     uint64_t id,
                                     txservice::TransactionExecution *txm)
{
    // 1. Check if the index exists by reading from vector_index_meta_table
    std::string h_key = build_metadata_key(name);
    TxKey tx_key = EloqStringKey::Create(h_key.c_str(), h_key.size());
    TxRecord::Uptr h_record = EloqStringRecord::Create();
    ReadTxRequest read_req(&vector_index_meta_table,
                           0,
                           &tx_key,
                           h_record.get(),
                           false, /* is_for_write */
                           false, /* is_for_share */
                           false, /* read_local */
                           0,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr,
                           txm);
    txm->Execute(&read_req);
    read_req.Wait();
    if (read_req.IsError())
    {
        return VectorOpResult::INDEX_META_OP_FAILED;
    }

    // 2. Check if the index exists
    if (read_req.Result().first != RecordStatus::Normal)
    {
        assert(read_req.Result().first == RecordStatus::Deleted);
        // The index does not exist
        return VectorOpResult::INDEX_NOT_EXIST;
    }

    uint64_t index_version = read_req.Result().second;

    // 3. Get or create the vector index from cache
    VectorIndex *index_ptr = nullptr;
    VectorOpResult get_result =
        GetOrCreateIndex(name, h_record, index_version, {}, index_ptr);
    if (get_result != VectorOpResult::SUCCEED)
    {
        return get_result;
    }

    // TODO(ysw): Delete item to log object.

    // 4. Delete the vector from the index
    if (!index_ptr->remove(id))
    {
        return VectorOpResult::INDEX_DELETE_FAILED;
    }
    return VectorOpResult::SUCCEED;
}

/**
 * @brief Retrieve a cached VectorIndex matching the requested version or create and initialize one.
 *
 * Checks the in-memory index cache under a shared lock and returns the cached index when its version
 * matches index_version. If the index is absent or uninitialized, acquires an exclusive lock,
 * performs a double-checked creation and initialization of an HNSWVectorIndex, loads it from storage,
 * updates the cached version, and returns the initialized pointer.
 *
 * @param name Index name.
 * @param h_record Transaction record containing the stored index metadata used for initialization.
 * @param index_version Expected metadata version for the index (must be > 0).
 * @param search_params Parameters used to override or augment the index configuration during initialization.
 * @param index_ptr Output parameter set to the initialized VectorIndex on success.
 * @return VectorOpResult SUCCEED on success; INDEX_VERSION_MISMATCH if a cached index has a different
 *         version; other error codes (e.g. INDEX_INIT_FAILED, INDEX_LOAD_FAILED) if initialization or
 *         loading fails.
 */
VectorOpResult VectorHandler::GetOrCreateIndex(
    const std::string &name,
    const txservice::TxRecord::Uptr &h_record,
    uint64_t index_version,
    const std::unordered_map<std::string, std::string> &search_params,
    VectorIndex *&index_ptr)
{
    assert(index_version > 0);
    // First, try to get read lock to check cache
    {
        std::shared_lock<std::shared_mutex> read_lock(vec_indexes_mutex_);
        auto it = vec_indexes_.find(name);
        if (it != vec_indexes_.end())
        {
            uint64_t cached_version = it->second.first;
            auto &cached_index_uptr = it->second.second;

            if (cached_version == 0)
            {
                // Index exists but not initialized, need to initialize
                // Will upgrade to write lock below
            }
            else if (cached_version != index_version)
            {
                return VectorOpResult::INDEX_VERSION_MISMATCH;
            }
            else
            {
                // Version matches, return the cached index
                index_ptr = cached_index_uptr.get();
                return VectorOpResult::SUCCEED;
            }
        }
    }

    // Need to create new index or initialize existing one
    {
        std::lock_guard<std::shared_mutex> write_lock(vec_indexes_mutex_);
        // Double-checked locking pattern
        auto it = vec_indexes_.find(name);
        if (it != vec_indexes_.end() && it->second.first > 0)
        {
            // Another thread has initialized the index, check version
            if (it->second.first != index_version)
            {
                return VectorOpResult::INDEX_VERSION_MISMATCH;
            }
            index_ptr = it->second.second.get();
            return VectorOpResult::SUCCEED;
        }

        // Need to create or initialize index
        if (it == vec_indexes_.end())
        {
            // Create new index
            auto hnsw_index = std::make_unique<HNSWVectorIndex>();
            auto emplace_pair = vec_indexes_.try_emplace(
                name, std::make_pair(index_version, std::move(hnsw_index)));
            assert(emplace_pair.second);
            it = emplace_pair.first;
        }

        // Initialize the index
        VectorOpResult init_result = InitializeIndex(
            it->second.second.get(), h_record, index_version, search_params);
        if (init_result != VectorOpResult::SUCCEED)
        {
            // Clean up on failure
            vec_indexes_.erase(name);
            return init_result;
        }

        // Update version and return
        it->second.first = index_version;
        index_ptr = it->second.second.get();
        return VectorOpResult::SUCCEED;
    }
}

/**
 * @brief Initialize and load a vector index instance from stored metadata.
 *
 * Decodes index metadata from the provided transaction record, builds an IndexConfig
 * (applying any overrides from search_params), calls the index' initialize method,
 * and then loads on-disk data from the metadata's file path.
 *
 * @param index_ptr Pointer to the VectorIndex instance to initialize; must be non-null.
 * @param h_record Transaction record containing the encoded index metadata blob.
 * @param index_version Version used when decoding the stored metadata.
 * @param search_params Parameters that override or augment the index's algorithm parameters.
 *
 * @return VectorOpResult::SUCCEED on successful initialization and load.
 * @return VectorOpResult::INDEX_INIT_FAILED if `index_ptr` is null or index_ptr->initialize(...) fails.
 * @return VectorOpResult::INDEX_LOAD_FAILED if index_ptr->load(...) fails.
 */
VectorOpResult VectorHandler::InitializeIndex(
    VectorIndex *index_ptr,
    const txservice::TxRecord::Uptr &h_record,
    uint64_t index_version,
    const std::unordered_map<std::string, std::string> &search_params)
{
    if (!index_ptr)
    {
        return VectorOpResult::INDEX_INIT_FAILED;
    }

    // Decode metadata from h_record
    VectorMetadata metadata;
    const char *blob_data = h_record->EncodedBlobData();
    size_t blob_size = h_record->EncodedBlobSize();
    size_t offset = 0;
    metadata.Decode(blob_data, blob_size, offset, index_version);

    // Construct IndexConfig from metadata
    IndexConfig config;
    config.name = metadata.VecName();
    config.dimension = metadata.Dimension();
    config.algorithm = metadata.VecAlgorithm();
    config.distance_metric = metadata.VecMetric();
    config.storage_path = metadata.FilePath();

    // Use search_params to override alg_params
    config.params = search_params;

    // Initialize the index
    if (!index_ptr->initialize(config))
    {
        return VectorOpResult::INDEX_INIT_FAILED;
    }

    // Load the index from storage
    if (!index_ptr->load(metadata.FilePath()))
    {
        return VectorOpResult::INDEX_LOAD_FAILED;
    }

    return VectorOpResult::SUCCEED;
}

}  // namespace EloqVec
