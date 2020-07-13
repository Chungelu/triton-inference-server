// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/backends/backend/examples/backend_utils.h"

namespace nvidia { namespace inferenceserver { namespace backend {

//
// InstanceProperties
//
std::string
InstanceProperties::AsString() const
{
  std::string str("id " + std::to_string(id_) + ", ");
  switch (kind_) {
    case Kind::CPU:
      str += "CPU";
      break;
    case Kind::GPU:
      str += "GPU (" + std::to_string(device_id_) + ")";
      break;
    case Kind::MODEL:
      str += "MODEL";
      break;
    default:
      break;
  }

  return str;
}

//
//
//

TRITONSERVER_Error*
ParseInstanceGroups(
    TritonJson::Value& model_config, std::vector<InstanceProperties>* instances)
{
  instances->clear();

  TritonJson::Value instance_groups;
  RETURN_IF_ERROR(
      model_config.MemberAsArray("instance_group", &instance_groups));

  size_t idx = 0;
  for (size_t i = 0; i < instance_groups.ArraySize(); ++i) {
    TritonJson::Value instance_group;
    RETURN_IF_ERROR(instance_groups.IndexAsObject(i, &instance_group));

    int64_t count = 0;
    RETURN_IF_ERROR(instance_group.MemberAsInt("count", &count));

    std::string kind_str;
    RETURN_IF_ERROR(instance_group.MemberAsString("kind", &kind_str));

    if (kind_str == "KIND_CPU") {
      for (int32_t c = 0; c < count; ++c) {
        instances->emplace_back(
            idx++, InstanceProperties::Kind::CPU, 0 /* device_id */);
      }
    } else if (kind_str == "KIND_GPU") {
      TritonJson::Value gpus;
      RETURN_IF_ERROR(instance_group.MemberAsArray("gpus", &gpus));
      for (size_t g = 0; g < gpus.ArraySize(); ++g) {
        int64_t device_id;
        RETURN_IF_ERROR(gpus.IndexAsInt(g, &device_id));
        for (int32_t c = 0; c < count; ++c) {
          instances->emplace_back(
              idx++, InstanceProperties::Kind::GPU, device_id);
        }
      }
    } else if (kind_str == "KIND_MODEL") {
      for (int32_t c = 0; c < count; ++c) {
        instances->emplace_back(
            idx++, InstanceProperties::Kind::MODEL, 0 /* device_id */);
      }
    } else {
      RETURN_ERROR_IF_FALSE(
          false, TRITONSERVER_ERROR_INVALID_ARG,
          std::string("instance_group ") + kind_str + " not supported");
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ParseShape(
    TritonJson::Value& io, const std::string& name, std::vector<int64_t>* shape)
{
  TritonJson::Value shape_array;
  RETURN_IF_ERROR(io.MemberAsArray(name.c_str(), &shape_array));
  for (size_t i = 0; i < shape_array.ArraySize(); ++i) {
    int64_t d;
    RETURN_IF_ERROR(shape_array.IndexAsInt(i, &d));
    shape->push_back(d);
  }

  return nullptr;  // success
}

std::string
ShapeToString(const int64_t* dims, const size_t dims_count)
{
  bool first = true;

  std::string str("[");
  for (size_t i = 0; i < dims_count; ++i) {
    const int64_t dim = dims[i];
    if (!first) {
      str += ",";
    }
    str += std::to_string(dim);
    first = false;
  }

  str += "]";
  return str;
}

std::string
ShapeToString(const std::vector<int64_t>& shape)
{
  return ShapeToString(shape.data(), shape.size());
}

TRITONSERVER_Error*
ReadInputTensor(
    TRITONBACKEND_Request* request, const std::string& input_name, char* buffer,
    size_t* buffer_byte_size)
{
  TRITONBACKEND_Input* input;
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInputByName(request, input_name.c_str(), &input));

  uint64_t input_byte_size;
  uint32_t input_buffer_count;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, nullptr, nullptr, nullptr, nullptr, &input_byte_size,
      &input_buffer_count));
  RETURN_ERROR_IF_FALSE(
      input_byte_size <= *buffer_byte_size, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "buffer to small for input tensor '" + input_name + "', " +
          std::to_string(*buffer_byte_size) + " < " +
          std::to_string(input_byte_size)));

  size_t output_buffer_offset = 0;
  for (uint32_t b = 0; b < input_buffer_count; ++b) {
    const void* input_buffer = nullptr;
    uint64_t input_buffer_byte_size = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, b, &input_buffer, &input_buffer_byte_size, &input_memory_type,
        &input_memory_type_id));
    RETURN_ERROR_IF_FALSE(
        input_memory_type != TRITONSERVER_MEMORY_GPU,
        TRITONSERVER_ERROR_INTERNAL,
        std::string("expected input tensor in CPU memory"));

    memcpy(buffer + output_buffer_offset, input_buffer, input_buffer_byte_size);
    output_buffer_offset += input_buffer_byte_size;
  }

  *buffer_byte_size = input_byte_size;

  return nullptr;  // success
}

}}}  // namespace nvidia::inferenceserver::backend
