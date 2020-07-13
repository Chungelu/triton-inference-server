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

#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace ni = nvidia::inferenceserver;
namespace nib = nvidia::inferenceserver::backend;

//
// WIP: TF Graphdef Backend that implements the TRITONBACKEND API.
//

namespace {

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      TRITONSERVER_ErrorDelete(rarie_err__);                            \
      return;                                                           \
    }                                                                   \
  } while (false)

#define RESPOND_FACTORY_AND_RETURN_IF_ERROR(FACTORY, X)                      \
  do {                                                                       \
    TRITONSERVER_Error* rfarie_err__ = (X);                                  \
    if (rfarie_err__ != nullptr) {                                           \
      TRITONBACKEND_Response* rfarie_response__ = nullptr;                   \
      LOG_IF_ERROR(                                                          \
          TRITONBACKEND_ResponseNewFromFactory(&rfarie_response__, FACTORY), \
          "failed to create response");                                      \
      if (rfarie_response__ != nullptr) {                                    \
        LOG_IF_ERROR(                                                        \
            TRITONBACKEND_ResponseSend(                                      \
                rfarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL,     \
                rfarie_err__),                                               \
            "failed to send error response");                                \
      }                                                                      \
      TRITONSERVER_ErrorDelete(rfarie_err__);                                \
      return;                                                                \
    }                                                                        \
  } while (false)


//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  class ModelFiles {
   public:
    ModelFiles(const std::string& path, const bool is_graphdef);
    ~ModelFiles();
    const std::unordered_map<std::string, std::string>& ModelPaths()
    {
      return paths_;
    }

   private:
    std::unordered_map<std::string, std::string> paths_;
  };

  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  ~ModelState();

  TRITONSERVER_Error* CreateExecutionContexts();

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Spawn a thread to produce outputs for a request. Return the
  // request wait time before it should release.
  void ProcessRequest(TRITONBACKEND_Request* request, uint32_t* wait_ms);

 private:
  ModelState(
      TRITONBACKEND_Model* triton_model, const std::string& name,
      ni::TritonJson::Value&& model_config);
  void ResponseThread(
      TRITONBACKEND_ResponseFactory* factory_ptr, const int32_t* in_buffer_ptr,
      const int32_t* delay_buffer_ptr, const uint32_t element_count);

  TRITONSERVER_Error* CreateExecutionContext(
      const std::string& instance_name, const nib::InstanceProperties& device,
      const std::unordered_map<std::string, std::string>& paths);

  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  ni::TritonJson::Value model_config_;
  std::atomic<size_t> inflight_thread_count_;
};

// FIXME use unix fs api directly
ModelState::ModelFiles::ModelFiles(
    const std::string& path, const bool is_graphdef)
{
  std::set<std::string> model_files;
  if (is_graphdef) {
    // Read all the graphdef files in 'path'.
    RETURN_IF_ERROR(
        GetDirectoryFiles(path, true /* skip_hidden_files */, &model_files));
  } else {
    RETURN_IF_ERROR(GetDirectorySubdirs(path, &model_files));
  }

  for (const auto& filename : model_files) {
    const auto model_path = JoinPath({path, filename});
    std::string local_model_path;

    RETURN_IF_ERROR(DownloadFileFolder(model_path, &local_model_path));
    paths_.emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(local_model_path));
  }
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  ni::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &name));

  *state = new ModelState(triton_model, name, std::move(model_config));
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreateExecutionContexts()
{
  const char* path = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelRepositoryPath(triton_model_, &path));
  std::string platform;
  RETURN_IF_ERROR(model_config_.MemberAsString("platform", &platform));
  bool is_graphdef;
  if (platform == "tensorflow_graphdef") {
    is_graphdef = true;
  } else if (platform == "tensorflow_savedmodel") {
    is_graphdef = false;
  } else {
    RETURN_ERROR_IF_FALSE(
        false, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("platform ") + platform + " not supported");
  }
  auto mf = ModelFiles(path, is_graphdef);
  std::vector<nib::InstanceProperties> instances;
  RETURN_IF_ERROR(nib::ParseInstanceGroups(model_config_, &instances));

  const char* cname = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model_, &cname));
  const std::string name = std::string(cname);
  for (const auto& instance : instances) {
    switch (instance.kind_) {
      case nib::InstanceProperties::Kind::CPU: {
        const std::string instance_name =
            name + "_" + std::to_string(instance.id_) + "_cpu";
        RETURN_IF_ERROR(
            CreateExecutionContext(instance_name, instance, mf.ModelPaths()));
        break;
      }
      case nib::InstanceProperties::Kind::GPU: {
        const std::string instance_name =
            name + "_" + std::to_string(instance.id_) + "_gpu" +
            std::to_string(instance.device_id_);
        RETURN_IF_ERROR(
            CreateExecutionContext(instance_name, instance, mf.ModelPaths()));
        break;
      }
      case nib::InstanceProperties::Kind::MODEL: {
        const std::string instance_name =
            name + "_" + std::to_string(instance.id_) + "_model_device";
        RETURN_IF_ERROR(
            CreateExecutionContext(instance_name, instance, mf.ModelPaths()));
        break;
      }
      default: {
        RETURN_ERROR_IF_FALSE(
            false, TRITONSERVER_ERROR_INVALID_ARG,
            std::string("instance setting ") + instance.AsString() +
                " not supported");
        break;
      }
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreateExecutionContext(
    const std::string& instance_name, const nib::InstanceProperties& device,
    const std::unordered_map<std::string, std::string>& paths)
{
  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc_model_filename;
  model_config_.MemberAsString("default_model_filename", &cc_model_filename);
  int device_id = device.device_id_;

  switch (device.kind_) {
    case nib::InstanceProperties::Kind::CPU: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("Creating instance ") + instance_name +
           " on CPU using " + cc_model_filename)
              .c_str());
      break;
    }
    case nib::InstanceProperties::Kind::MODEL: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("Creating instance ") + instance_name +
           " on devices using " + cc_model_filename)
              .c_str());
      break;
    }
    default: {
#ifdef TRITON_ENABLE_GPU
      cudaDeviceProp cuprops;
      cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, device_id);
      if (cuerr != cudaSuccess) {
        RETURN_ERROR_IF_FALSE(
            false, TRITONSERVER_ERROR_INTERNAL,
            std::string("unable to get CUDA device properties for ") + name_ +
                ": " + cudaGetErrorString(cuerr));
      }

      const std::string cc =
          std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
      ni::TritonJson::Value cc_names;
      ni::TritonJson::Value cc_name;
      if ((model_config_.Find("cc_model_filenames", &cc_names)) &&
          (cc_names.Find(cc.c_str(), &cc_name))) {
        cc_name.AsString(&cc_model_filename);
      }

      // FIXME move virtual device utils into backend
      // // Get virtual device tracker instance, and get next device id
      // if (VirtualDeviceTracker::HasVirtualDevice()) {
      //   RETURN_IF_ERROR(
      //       VirtualDeviceTracker::GetNextVirtualDevice(gpu_device,
      //       &vgpu_device));
      // }
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("Creating instance ") + instance_name + " on GPU " +
           std::to_string(device_id) + " (" + cc + ") using " +
           cc_model_filename)
              .c_str());
#else
      RETURN_ERROR_IF_FALSE(
          false, TRITONSERVER_ERROR_INTERNAL, "GPU instances not supported");
#endif  // TRITON_ENABLE_GPU
      break;
    }
  }

  const auto& gdp_itr = paths.find(cc_model_filename);
  if (gdp_itr == paths.end()) {
    RETURN_ERROR_IF_FALSE(
          false, TRITONSERVER_ERROR_INTERNAL, (std::string("unable to find model '")
          + cc_model_filename + "' for " + name_));
  }

  // FIXME WIP below

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();
  const bool pinned_input =
      Config().optimization().input_pinned_memory().enable();
  const bool pinned_output =
      Config().optimization().output_pinned_memory().enable();

  std::unique_ptr<MetricModelReporter> metric_reporter;
#ifdef TRITON_ENABLE_METRICS
  if (Metrics::Enabled()) {
    metric_reporter.reset(new MetricModelReporter(
        Name(), Version(), gpu_device, Config().metric_tags()));
  }
#endif  // TRITON_ENABLE_METRICS

  contexts_.emplace_back(new Context(
      instance_name, gpu_device, mbs, pinned_input, pinned_output,
      std::move(metric_reporter)));
  Context* context = static_cast<Context*>(contexts_.back().get());

  RETURN_IF_ERROR(context->CreateCudaStream());

  RETURN_IF_ERROR(context->ValidateInputs(Config().input()));
  RETURN_IF_ERROR(context->ValidateOutputs(Config().output()));

  TRTISTF_TFTRTConfig* tftrt_config_ptr = nullptr;
  TRTISTF_TFTRTConfig tftrt_config;
  bool auto_mixed_precision = false;
  if (Config().optimization().has_execution_accelerators()) {
    // Set default values. is_dynamic_op is always true for online
    // TF-TRT.
    tftrt_config.minimum_segment_size_ = 3;
    tftrt_config.max_workspace_size_bytes_ = 1 << 30;
    tftrt_config.max_cached_engines_ = 100;
    tftrt_config.max_batch_size_ = std::max(Config().max_batch_size(), 1);
    tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
    tftrt_config.is_dynamic_op_ = true;

    if (!Config()
             .optimization()
             .execution_accelerators()
             .cpu_execution_accelerator()
             .empty()) {
      return Status(
          Status::Code::INVALID_ARG,
          "CPU Execution Accelerator is not supported in TensorFlow backend");
    }

    if (gpu_device == Context::NO_GPU_DEVICE) {
      return Status(
          Status::Code::INVALID_ARG,
          "GPU Execution Accelerator can only be set on non-CPU backend "
          "context");
    }
    for (const auto& execution_accelerator : Config()
                                                 .optimization()
                                                 .execution_accelerators()
                                                 .gpu_execution_accelerator()) {
      if (execution_accelerator.name() == kTensorRTExecutionAccelerator) {
        // Validate and set parameters
        for (const auto& parameter : execution_accelerator.parameters()) {
          if (parameter.first == "precision_mode") {
            if (parameter.second == "FP32") {
              tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
            } else if (parameter.second == "FP16") {
              tftrt_config.precision_mode_ = TRTISTF_MODE_FP16;
            } else {
              return Status(
                  Status::Code::INVALID_ARG, "unsupported precision mode '" +
                                                 parameter.second +
                                                 "' is requested");
            }
          } else if (parameter.first == "minimum_segment_size") {
            RETURN_IF_ERROR(ParseLongLongParameter(
                parameter.first, parameter.second,
                &tftrt_config.minimum_segment_size_));
          } else if (parameter.first == "max_workspace_size_bytes") {
            RETURN_IF_ERROR(ParseLongLongParameter(
                parameter.first, parameter.second,
                &tftrt_config.max_workspace_size_bytes_));
          } else if (parameter.first == "max_cached_engines") {
            RETURN_IF_ERROR(ParseLongLongParameter(
                parameter.first, parameter.second,
                &tftrt_config.max_cached_engines_));
          } else {
            return Status(
                Status::Code::INVALID_ARG,
                "unknown parameter '" + parameter.first +
                    "' is provided for TensorRT Execution Accelerator");
          }
        }
        tftrt_config_ptr = &tftrt_config;
        LOG_VERBOSE(1) << "TensorRT Execution Accelerator is set for "
                       << instance_name;
      } else if (execution_accelerator.name() == kGPUIOExecutionAccelerator) {
        // GPU I/O can be set, set hint
        if ((gpu_device != Context::NO_GPU_DEVICE) &&
            (gpu_device != Context::MODEL_DEVICE)) {
          // In TensorFlow, TF device (vGPU) is used for device utilities
          context->input_device_id_ = vgpu_device;
        }
      } else if (
          execution_accelerator.name() ==
          kAutoMixedPrecisionExecutionAccelerator) {
        auto_mixed_precision = true;
      } else {
        return Status(
            Status::Code::INVALID_ARG, "unknown Execution Accelerator '" +
                                           execution_accelerator.name() +
                                           "' is requested");
      }
    }
  }

  if (auto_mixed_precision && (tftrt_config_ptr != nullptr)) {
    return Status(
        Status::Code::INVALID_ARG,
        "Auto mixed precision can not be set with TFTRT optimization");
  }

  // [TODO] use framework in model config to create model for different type
  RETURN_IF_ERROR(CreateTRTISTFModel(
      backend_config_, vgpu_device, Config().optimization().has_graph(),
      Config().optimization().graph().level(), gdp_itr->first, gdp_itr->second,
      &context->trtistf_model_, &context->input_name_map_,
      &context->output_name_map_, tftrt_config_ptr, auto_mixed_precision));


  if (context->input_device_id_ != Context::MODEL_DEVICE) {
    const size_t num_inputs = Config().input_size();
    const size_t num_outputs = Config().output_size();
    std::vector<const char*> input_names, output_names;
    std::vector<TRTISTF_DataType> input_types, output_types;
    for (const auto& io : Config().input()) {
      input_names.push_back(io.name().c_str());
      input_types.push_back(ConvertDataType(io.data_type()));
    }
    for (const auto& io : Config().output()) {
      output_names.push_back(io.name().c_str());
      output_types.push_back(ConvertDataType(io.data_type()));
    }
    TRTISTF_ModelMakeCallable(
        context->trtistf_model_.get(), input_names.data(), input_types.data(),
        num_inputs, output_names.data(), output_types.data(), num_outputs);
  }

  return Status::Success;
}

Status
BaseBackend::Context::ValidateInputs(
    const ::google::protobuf::RepeatedPtrField<ModelInput>& ios)
{
  for (const auto& io : ios) {
    if (ConvertDataType(io.data_type()) ==
        TRTISTF_DataType::TRTISTF_TYPE_INVALID) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}

Status
BaseBackend::Context::ValidateOutputs(
    const ::google::protobuf::RepeatedPtrField<ModelOutput>& ios)
{
  for (const auto& io : ios) {
    if (ConvertDataType(io.data_type()) ==
        TRTISTF_DataType::TRTISTF_TYPE_INVALID) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + name_ + "'");
    }
  }

  return Status::Success;
}

ModelState::ModelState(
    TRITONBACKEND_Model* triton_model, const std::string& name,
    ni::TritonJson::Value&& model_config)
    : triton_model_(triton_model), name_(name),
      model_config_(std::move(model_config)), inflight_thread_count_(0)
{
}

ModelState::~ModelState()
{
  // Wait for all threads to exit...
  while (inflight_thread_count_ > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  ni::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  // max_batch_size must be 0 because this backend does not support
  // batching
  int64_t max_batch_size;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &max_batch_size));
  RETURN_ERROR_IF_FALSE(
      max_batch_size == 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "repeat backend only supports models with max_batch_size == 0"));

  ni::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 3 inputs and 2 outputs.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 3, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 3 inputs, got ") +
          std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 2 outputs, got ") +
          std::to_string(outputs.ArraySize()));

  // Here we rely on the model configuation listing the inputs and
  // outputs in a specific order, which we shouldn't really require...
  ni::TritonJson::Value in, delay, wait, out, idx;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &in));
  RETURN_IF_ERROR(inputs.IndexAsObject(1, &delay));
  RETURN_IF_ERROR(inputs.IndexAsObject(2, &wait));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &out));
  RETURN_IF_ERROR(outputs.IndexAsObject(1, &idx));

  // Check tensor names
  std::string in_name, delay_name, wait_name, out_name, idx_name;
  RETURN_IF_ERROR(in.MemberAsString("name", &in_name));
  RETURN_IF_ERROR(delay.MemberAsString("name", &delay_name));
  RETURN_IF_ERROR(wait.MemberAsString("name", &wait_name));
  RETURN_IF_ERROR(out.MemberAsString("name", &out_name));
  RETURN_IF_ERROR(idx.MemberAsString("name", &idx_name));

  RETURN_ERROR_IF_FALSE(
      in_name == "IN", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first input tensor name to be IN, got ") + in_name);
  RETURN_ERROR_IF_FALSE(
      delay_name == "DELAY", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second input tensor name to be DELAY, got ") +
          delay_name);
  RETURN_ERROR_IF_FALSE(
      wait_name == "WAIT", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected third input tensor name to be WAIT, got ") +
          wait_name);
  RETURN_ERROR_IF_FALSE(
      out_name == "OUT", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first output tensor name to be OUT, got ") +
          out_name);
  RETURN_ERROR_IF_FALSE(
      idx_name == "IDX", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second output tensor name to be IDX, got ") +
          idx_name);

  // Check shapes
  std::vector<int64_t> in_shape, delay_shape, wait_shape, out_shape, idx_shape;
  RETURN_IF_ERROR(nib::ParseShape(in, "dims", &in_shape));
  RETURN_IF_ERROR(nib::ParseShape(delay, "dims", &delay_shape));
  RETURN_IF_ERROR(nib::ParseShape(wait, "dims", &wait_shape));
  RETURN_IF_ERROR(nib::ParseShape(out, "dims", &out_shape));
  RETURN_IF_ERROR(nib::ParseShape(idx, "dims", &idx_shape));

  RETURN_ERROR_IF_FALSE(
      in_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN shape to have one dimension, got ") +
          nib::ShapeToString(in_shape));
  RETURN_ERROR_IF_FALSE(
      in_shape == delay_shape, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN and DELAY shape to match, got ") +
          nib::ShapeToString(in_shape) + " and " +
          nib::ShapeToString(delay_shape));
  RETURN_ERROR_IF_FALSE(
      (wait_shape.size() == 1) && (wait_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAIT shape to be [1], got ") +
          nib::ShapeToString(wait_shape));
  RETURN_ERROR_IF_FALSE(
      (out_shape.size() == 1) && (out_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUT shape to be [1], got ") +
          nib::ShapeToString(out_shape));
  RETURN_ERROR_IF_FALSE(
      (idx_shape.size() == 1) && (idx_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IDX shape to be [1], got ") +
          nib::ShapeToString(idx_shape));

  // Check datatypes
  std::string in_dtype, delay_dtype, wait_dtype, out_dtype, idx_dtype;
  RETURN_IF_ERROR(in.MemberAsString("data_type", &in_dtype));
  RETURN_IF_ERROR(delay.MemberAsString("data_type", &delay_dtype));
  RETURN_IF_ERROR(wait.MemberAsString("data_type", &wait_dtype));
  RETURN_IF_ERROR(out.MemberAsString("data_type", &out_dtype));
  RETURN_IF_ERROR(idx.MemberAsString("data_type", &idx_dtype));

  RETURN_ERROR_IF_FALSE(
      in_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN datatype to be INT32, got ") + in_dtype);
  RETURN_ERROR_IF_FALSE(
      delay_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected DELAY datatype to be UINT32, got ") + delay_dtype);
  RETURN_ERROR_IF_FALSE(
      wait_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAIT datatype to be UINT32, got ") + wait_dtype);
  RETURN_ERROR_IF_FALSE(
      out_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUT datatype to be INT32, got ") + out_dtype);
  RETURN_ERROR_IF_FALSE(
      idx_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IDX datatype to be UINT32, got ") + idx_dtype);

  // For simplicity this backend doesn't support multiple
  // instances. So check and give a warning if more than one instance
  // is requested.
  std::vector<nib::InstanceProperties> instances;
  RETURN_IF_ERROR(nib::ParseInstanceGroups(model_config_, &instances));
  if (instances.size() != 1) {
    TRITONSERVER_LogMessage(
        TRITONSERVER_LOG_WARN, __FILE__, __LINE__,
        (std::string("model configuration specifies ") +
         std::to_string(instances.size()) +
         " instances but repeat backend supports only a single CPU instance. "
         "Additional instances ignored")
            .c_str());
  }

  return nullptr;  // success
}

}  // namespace

/////////////

extern "C" {

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelExecute(
    TRITONBACKEND_Model* model, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &model_name));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelExecute: model ") + model_name +
       " with " + std::to_string(request_count) + " requests")
          .c_str());

  // Triton only calls model execute from a single thread at a time
  // *for a given model*. But since this backend could be used by
  // multiple models the implementation needs to handle multiple
  // models executing at the same time. Good practice for this is to
  // use only function-local and model-specific state (obtained from
  // 'model'), which is what we do here.
  ModelState* state;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelState(model, reinterpret_cast<void**>(&state)));

  // This backend does not support models that support batching, so
  // 'request_count' should always be 1.
  RETURN_ERROR_IF_FALSE(
      request_count <= 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("repeat backend does not support batched request execution"));

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);
  uint32_t wait_milliseconds = 0;

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.

  // For simplicity we process each request in a separate thread.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    state->ProcessRequest(request, &wait_milliseconds);
  }

  // Wait, release, return...
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("waiting ") + std::to_string(wait_milliseconds) +
       " ms before releasing requests")
          .c_str());

  std::this_thread::sleep_for(std::chrono::milliseconds(wait_milliseconds));

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelReportStatistics(
            model, request, true /* success */, TRITONBACKEND_NO_DEVICE,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelReportBatchStatistics(
          model, 1 /*total_batch_size*/, exec_start_ns, exec_start_ns,
          exec_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelExecute: model ") + model_name +
       " released " + std::to_string(request_count) + " requests")
          .c_str());

  return nullptr;  // success
}

}  // extern "C"
