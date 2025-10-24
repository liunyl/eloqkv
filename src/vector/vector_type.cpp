/**
 *    Copyright (C) 2025 EloqData Inc.
 *
 *    This program is free software: you can redistribute it and/or  modify
 *    it under either of the following two licenses:
 *    1. GNU Affero General Public License, version 3, as published by the Free
 *    Software Foundation.
 *    2. GNU General Public License as published by the Free Software
 *    Foundation; version 2 of the License.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Affero General Public License or GNU General Public License for more
 *    details.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    and GNU General Public License V2 along with this program.  If not, see
 *    <http://www.gnu.org/licenses/>.
 *
 */
#include "vector_type.h"

#include <algorithm>
#include <stdexcept>

#include "tx_util.h"

namespace EloqVec
{

using txservice::LocalCcShards;

// ============================================================================
// IndexConfig Implementation
// ============================================================================

/**
 * Serialization format:
 * dimension (8 bytes)
 * | max_elements (8 bytes)
 * | algorithm (1 byte)
 * | distance_metric (1 byte)
 * | params_count (4 bytes) | [key_len (4 bytes) | key | value_len (4 bytes)
 * | value]*
 */
void IndexConfig::Encode(std::string &encoded_str) const
{
    // dimension (8 bytes)
    encoded_str.append(reinterpret_cast<const char *>(&dimension),
                       sizeof(size_t));

    // max_elements (8 bytes)
    encoded_str.append(reinterpret_cast<const char *>(&max_elements),
                       sizeof(size_t));

    // algorithm (1 byte)
    encoded_str.append(reinterpret_cast<const char *>(&algorithm),
                       sizeof(uint8_t));

    // distance_metric (1 byte)
    encoded_str.append(reinterpret_cast<const char *>(&distance_metric),
                       sizeof(uint8_t));

    // params
    uint32_t param_count = static_cast<uint32_t>(params.size());
    encoded_str.append(reinterpret_cast<const char *>(&param_count),
                       sizeof(uint32_t));
    for (const auto &[key, value] : params)
    {
        uint32_t key_len = static_cast<uint32_t>(key.size());
        encoded_str.append(reinterpret_cast<const char *>(&key_len),
                           sizeof(uint32_t));
        encoded_str.append(key);

        uint32_t value_len = static_cast<uint32_t>(value.size());
        encoded_str.append(reinterpret_cast<const char *>(&value_len),
                           sizeof(uint32_t));
        encoded_str.append(value);
    }
}

void IndexConfig::Decode(const char *buf, size_t buff_size, size_t &offset)
{
    // dimension
    dimension = *reinterpret_cast<const size_t *>(buf + offset);
    offset += sizeof(size_t);

    // max_elements
    max_elements = *reinterpret_cast<const size_t *>(buf + offset);
    offset += sizeof(size_t);

    // algorithm
    algorithm = static_cast<Algorithm>(
        *reinterpret_cast<const uint8_t *>(buf + offset));
    offset += sizeof(uint8_t);

    // distance_metric
    distance_metric = static_cast<DistanceMetric>(
        *reinterpret_cast<const uint8_t *>(buf + offset));
    offset += sizeof(uint8_t);

    // params
    uint32_t param_count = *reinterpret_cast<const uint32_t *>(buf + offset);
    offset += sizeof(uint32_t);

    params.clear();
    for (uint32_t i = 0; i < param_count; ++i)
    {
        uint32_t key_len = *reinterpret_cast<const uint32_t *>(buf + offset);
        offset += sizeof(uint32_t);
        std::string key(buf + offset, key_len);
        offset += key_len;

        uint32_t value_len = *reinterpret_cast<const uint32_t *>(buf + offset);
        offset += sizeof(uint32_t);
        std::string value(buf + offset, value_len);
        offset += value_len;

        params.emplace(std::move(key), std::move(value));
    }
}

// ============================================================================
// VectorRecordMetadata Implementation
// ============================================================================

/**
 * Serialization format:
 * field_count (4 bytes)
 * | [field_name_len (4 bytes) | field_name | field_type (1 byte)]*
 */
void VectorRecordMetadata::Encode(std::string &encoded_str) const
{
    // 1. Encode field count
    uint32_t field_count = static_cast<uint32_t>(field_names_.size());
    encoded_str.append(reinterpret_cast<const char *>(&field_count),
                       sizeof(uint32_t));

    // 2. Encode each field (name + type)
    for (size_t i = 0; i < field_names_.size(); ++i)
    {
        // Encode field name
        uint32_t name_len = static_cast<uint32_t>(field_names_[i].size());
        encoded_str.append(reinterpret_cast<const char *>(&name_len),
                           sizeof(uint32_t));
        encoded_str.append(field_names_[i].data(), name_len);

        // Encode field type
        uint8_t field_type = static_cast<uint8_t>(field_types_[i]);
        encoded_str.append(reinterpret_cast<const char *>(&field_type),
                           sizeof(uint8_t));
    }
}

void VectorRecordMetadata::Decode(const char *buf,
                                  size_t buff_size,
                                  size_t &offset)
{
    // 1. Decode field count
    uint32_t field_count = *reinterpret_cast<const uint32_t *>(buf + offset);
    offset += sizeof(uint32_t);

    // 2. Clear existing data and reserve space
    field_names_.clear();
    field_types_.clear();
    field_names_.reserve(field_count);
    field_types_.reserve(field_count);

    // 3. Decode each field (name + type)
    for (uint32_t i = 0; i < field_count; ++i)
    {
        // Decode field name
        uint32_t name_len = *reinterpret_cast<const uint32_t *>(buf + offset);
        offset += sizeof(uint32_t);
        std::string field_name(buf + offset, name_len);
        offset += name_len;

        // Decode field type
        MetadataFieldType field_type = static_cast<MetadataFieldType>(
            *reinterpret_cast<const uint8_t *>(buf + offset));
        offset += sizeof(uint8_t);

        // Add to vectors
        field_names_.push_back(std::move(field_name));
        field_types_.push_back(field_type);
    }
}

void VectorRecordMetadata::AddMetadataField(const std::string &field_name,
                                            MetadataFieldType field_type)
{
    // Check if field already exists
    if (HasMetadataField(field_name))
    {
        // Field already exists, don't add duplicate
        return;
    }

    field_names_.push_back(field_name);
    field_types_.push_back(field_type);
}

MetadataFieldType VectorRecordMetadata::GetFieldType(
    const std::string &field_name) const
{
    for (size_t i = 0; i < field_names_.size(); ++i)
    {
        if (field_names_[i] == field_name)
        {
            return field_types_[i];
        }
    }

    // Field not found, this should not happen if caller checks HasMetadataField
    // first
    throw std::runtime_error("Field not found: " + field_name);
}

size_t VectorRecordMetadata::GetFieldIndex(const std::string &field_name) const
{
    for (size_t i = 0;
         i < field_names_.size() && field_names_[i].compare(field_name) == 0;
         ++i)
    {
        return i;
    }
    return UINT64_MAX;  // Not found
}

// ============================================================================
// VectorIndexMetadata Implementation
// ============================================================================

VectorIndexMetadata::VectorIndexMetadata(std::string &&name,
                                         IndexConfig &&config,
                                         VectorRecordMetadata &&metadata,
                                         int64_t persist_threshold,
                                         const std::string &storage_base_path)
    : name_(std::move(name)),
      config_(std::move(config)),
      metadata_(std::move(metadata)),
      persist_threshold_(persist_threshold),
      created_ts_(0),
      last_persist_ts_(0)
{
    // Construct timestamped file path from base path
    uint64_t ts = LocalCcShards::ClockTs();
    file_path_.assign(storage_base_path);
    if (!file_path_.empty() && file_path_.back() != '/')
    {
        file_path_.append("/");
    }
    file_path_.append(name_)
        .append("-")
        .append(std::to_string(ts))
        .append(".index");
}

/**
 * Serialization format:
 * name_len (2 bytes) | name
 * | IndexConfig::Encode()
 * | VectorRecordMetadata::Encode()
 * | persist_threshold (8 bytes)
 * | file_path_len (4 bytes) | file_path
 * | created_ts (8 bytes)
 * | last_persist_ts (8 bytes)
 */
void VectorIndexMetadata::Encode(std::string &encoded_str) const
{
    // name (2 bytes length + data)
    uint16_t name_len = static_cast<uint16_t>(name_.size());
    encoded_str.append(reinterpret_cast<const char *>(&name_len),
                       sizeof(uint16_t));
    encoded_str.append(name_);

    // Encode embedded IndexConfig
    config_.Encode(encoded_str);

    // Encode embedded VectorRecordMetadata
    metadata_.Encode(encoded_str);

    // persist_threshold (8 bytes)
    encoded_str.append(reinterpret_cast<const char *>(&persist_threshold_),
                       sizeof(int64_t));

    // file_path (4 bytes length + data)
    uint32_t file_path_len = static_cast<uint32_t>(file_path_.size());
    encoded_str.append(reinterpret_cast<const char *>(&file_path_len),
                       sizeof(uint32_t));
    encoded_str.append(file_path_);

    // timestamps (8 bytes each)
    encoded_str.append(reinterpret_cast<const char *>(&created_ts_),
                       sizeof(uint64_t));

    // last_persist_ts (8 bytes)
    encoded_str.append(reinterpret_cast<const char *>(&last_persist_ts_),
                       sizeof(uint64_t));
}

void VectorIndexMetadata::Decode(const char *buf,
                                 size_t buff_size,
                                 size_t &offset,
                                 uint64_t version)
{
    // name
    uint16_t name_len = *reinterpret_cast<const uint16_t *>(buf + offset);
    offset += sizeof(uint16_t);
    name_.assign(buf + offset, name_len);
    offset += name_len;

    // Decode embedded IndexConfig
    config_.Decode(buf, buff_size, offset);

    // Decode embedded VectorRecordMetadata
    metadata_.Decode(buf, buff_size, offset);

    // persist_threshold
    persist_threshold_ = *reinterpret_cast<const int64_t *>(buf + offset);
    offset += sizeof(int64_t);

    // file_path
    uint32_t file_path_len = *reinterpret_cast<const uint32_t *>(buf + offset);
    offset += sizeof(uint32_t);
    file_path_.assign(buf + offset, file_path_len);
    offset += file_path_len;

    // timestamps
    created_ts_ = *reinterpret_cast<const uint64_t *>(buf + offset);
    created_ts_ = (created_ts_ == 0) ? version : created_ts_;
    offset += sizeof(uint64_t);

    // last_persist_ts
    last_persist_ts_ = *reinterpret_cast<const uint64_t *>(buf + offset);
    offset += sizeof(uint64_t);
}

}  // namespace EloqVec
