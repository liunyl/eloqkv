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
#pragma once

#include <nlohmann/json.hpp>

#include "vector_type.h"

namespace EloqVec
{
// Parse json field value to binary value based on field type
static inline bool ParseJSONFieldValue(const nlohmann::json &json_value,
                                       MetadataFieldType field_type,
                                       std::string &binary_value)
{
    switch (field_type)
    {
    case MetadataFieldType::Int32:
    {
        if (!json_value.is_number_integer())
        {
            return false;
        }
        int32_t val = json_value.get<int32_t>();
        binary_value.assign(reinterpret_cast<const char *>(&val),
                            sizeof(int32_t));
        break;
    }
    case MetadataFieldType::Int64:
    {
        if (!json_value.is_number_integer())
        {
            return false;
        }
        int64_t val = json_value.get<int64_t>();
        binary_value.assign(reinterpret_cast<const char *>(&val),
                            sizeof(int64_t));
        break;
    }
    case MetadataFieldType::Double:
    {
        if (!json_value.is_number_float())
        {
            return false;
        }
        double val = json_value.get<double>();
        binary_value.assign(reinterpret_cast<const char *>(&val),
                            sizeof(double));
        break;
    }
    case MetadataFieldType::Bool:
    {
        if (!json_value.is_boolean())
        {
            return false;
        }
        uint8_t val = json_value.get<bool>() ? 1 : 0;
        binary_value.assign(reinterpret_cast<const char *>(&val),
                            sizeof(uint8_t));
        break;
    }
    case MetadataFieldType::String:
    {
        if (!json_value.is_string())
        {
            return false;
        }
        binary_value.assign(json_value.get<std::string>().data(),
                            json_value.get<std::string>().size());
        break;
    }
    default:
        return false;  // Unknown type
    }
    return true;
}

// Parse metadata JSON and convert to binary format using schema
static inline bool ParseMetadataJSON(const std::string_view &metadata_json,
                                     const VectorRecordMetadata &schema,
                                     std::vector<std::string> &metadata_values)
{
    metadata_values.clear();
    // Parse JSON
    nlohmann::ordered_json metadata_obj;
    try
    {
        metadata_obj = nlohmann::ordered_json::parse(metadata_json);
    }
    catch (const nlohmann::ordered_json::parse_error &e)
    {
        // Invalid JSON
        return false;
    }

    if (!metadata_obj.is_object())
    {
        // Must be JSON object
        return false;
    }

    assert(metadata_obj.size() == schema.Size() &&
           "Metadata JSON size mismatch");
    metadata_values.reserve(schema.Size());

    // Convert each field to binary using schema
    size_t idx = 0;
    for (auto it = metadata_obj.begin(); it != metadata_obj.end(); ++idx, ++it)
    {
        const std::string &field_name = it.key();
        // Check if field exists in schema
        if (!schema.CheckMetadataField(field_name, idx))
        {
            return false;  // Field not in schema
        }

        MetadataFieldType field_type = schema.GetFieldType(idx);
        const nlohmann::json &json_value = it.value();

        // Convert JSON value to binary based on schema type
        std::string binary_value;
        if (!ParseJSONFieldValue(json_value, field_type, binary_value))
        {
            return false;
        }
        metadata_values.emplace_back(std::move(binary_value));
    }
    return true;
}

}  // namespace EloqVec