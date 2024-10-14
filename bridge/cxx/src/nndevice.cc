/**
 * @brief
 * @author linjh (linjh@zhejianglab.com)
 * @version 0.1
 * @date 2023-06-26
 */
#include "nndevice.h"

#include <glog/logging.h>
#include <inos/ai_chip/client/ai_chip_client.h>

namespace inos {
namespace aichip {
namespace nndevice {
// namespace bridge {

class nndevice_error : public std::exception {
 public:
  explicit nndevice_error(Err e) : e_(e) {
    auto code = static_cast<int>(e_);
    str_code_ = std::to_string(code);
  }

  const char* what() const noexcept override { return str_code_.c_str(); }

 private:
  Err e_;
  std::string str_code_;
};

static std::shared_ptr<Tensor> FromWrapper(
    const bridge::TensorWrapper& wrapper) {
  auto name = std::string(wrapper.Name());
  auto dtype = static_cast<DType>(wrapper.Dtype());
  auto layout = static_cast<Format>(wrapper.Layout());
  auto dims = wrapper.Dims();
  auto data_len = wrapper.DataLen();
  auto shape = Shape{.dim = (uint8_t)dims.size()};
  for (uint8_t i = 0; i < shape.dim; i++) {
    shape.data[i] = dims[i];
  }

  auto tensor = std::make_shared<Tensor>(
      name, shape, layout, dtype,
      data_len == 0 ? Tensor::Type::kVariable : Tensor::Type::kConstant,
      nullptr);

  if (0 != data_len) {
    auto data = wrapper.Data();
    auto tensor_data = std::make_unique<TensorData>();
    tensor_data->set_data_nocopy(data, data_len);
    tensor->set_data(std::move(tensor_data));
  }
  return tensor;
}

static Attribute FromWrapper(const bridge::AttributeWrapper& wrapper) {
  auto name = std::string(wrapper.Name());
  auto type = static_cast<AttrType>(wrapper.Type());
  switch (type) {
    case AttrType::kInt: {
      auto v = wrapper.AsInt();
      return Attribute::FromInt(name, v);
    }
    case AttrType::kInts: {
      auto vs = wrapper.AsInts();
      std::vector<int> ints;
      for (auto v : vs) {
        ints.push_back(v);
      }
      return Attribute::FromInts(name, ints);
    }
    case AttrType::kFloat: {
      auto v = wrapper.AsFloat();
      return Attribute::FromFloat(name, v);
    }
    case AttrType::kFloats: {
      auto vs = wrapper.AsFloats();
      std::vector<float> floats;
      for (auto v : vs) {
        floats.push_back(v);
      }
      return Attribute::FromFloats(name, floats);
    }
    case AttrType::kString: {
      auto v = std::string(wrapper.AsString());
      return Attribute::FromString(name, v);
    }
    case AttrType::kStrings: {
      auto vs = wrapper.AsStrings();
      std::vector<std::string> strs;
      for (auto v : vs) {
        strs.push_back(std::string(v));
      }
      return Attribute::FromStrings(name, strs);
    }
    default:
      CHECK(false) << "[bridge] Not support AttrType: " << (int)type;
      // return;
  }
}

static std::shared_ptr<Operator> FromWrapper(
    const bridge::OperatorWrapper& wrapper) {
  auto op_name = std::string(wrapper.Name());
  auto op_type = std::string(wrapper.Type());
  auto op = Operator::Builder().Create(op_name, op_type).Build();

  auto inputs_wrapper = wrapper.Inputs();
  for (auto& input_wrapper : inputs_wrapper) {
    auto input = FromWrapper(input_wrapper);
    auto tag = std::string(input_wrapper.Tag());
    op->AddInput(tag, input);
  }

  auto outputs_wrapper = wrapper.Outputs();
  for (auto& output_wrapper : outputs_wrapper) {
    auto output = FromWrapper(output_wrapper);
    auto tag = std::string(output_wrapper.Tag());
    op->AddOutput(tag, output);
  }

  auto attributes_wrapper = wrapper.Attributes();
  for (auto& attribute_wrapper : attributes_wrapper) {
    auto attr = FromWrapper(attribute_wrapper);
    op->AddAttribute(attr);
  }

  return op;
}

static std::shared_ptr<Graph> FromWrapper(const bridge::GraphWrapper& wrapper) {
  auto graph_name = std::string(wrapper.GraphName());
  auto graph = Graph::Builder().Create(graph_name).Build();

  LOG(INFO) << "[bridge] grap name: " << graph_name;

  auto op_wrappers = wrapper.GraphAllOperators();
  for (auto& op_wrapper : op_wrappers) {
    auto op = FromWrapper(op_wrapper);
    graph->AddOperator(op);
  }

  return graph;
}

static RustTensor ToRustTensor(std::shared_ptr<Tensor> tensor) {
  RustTensor rust_tensor;
  rust_tensor.name = std::string(tensor->name());
  rust_tensor.dtype = static_cast<uint32_t>(tensor->dtype());
  rust_tensor.layout = static_cast<uint32_t>(tensor->format());
  auto shape = tensor->shape();
  for (size_t i = 0; i < shape.dim; i++) {
    rust_tensor.dims.push_back(shape.data[i]);
  }

  rust_tensor.data = (uint8_t*)tensor->data()->data();
  rust_tensor.len = tensor->data()->length();

  return rust_tensor;
}

#undef CLIENT
#undef ENGINE
#define CLIENT aichip::AiChipClient::getInstance().GetNnDeviceClient()
#define ENGINE CLIENT->GetEngine()

rust::Vec<rust::String> GetCandidateBackends() {
  VLOG(1) << "[bridge] Call GetCandidateBackends in bridge cxx.";
  auto backends = rust::Vec<rust::String>();
  auto result = CLIENT->GetCandidateBackends();
  if (!result.IsOK()) {
    throw nndevice_error(result.GetError());
  }
  VLOG(1) << "[bridge] Call client GetCandidateBackends success.";
  for (auto backend : result.Get()) {
    backends.push_back(rust::String(std::string(backend)));
  }

  return backends;
}

std::unique_ptr<Context> CreateContext(rust::Str backend_id, DevId dev_id,
                                       const RustOptions& opts) {
  VLOG(1) << "[bridge] Call CreateContext in bridge cxx.";
  auto _opts = Options();

  for (size_t i = 0; i < opts.keys.size(); i++) {
    _opts[std::string(opts.keys[i])] = std::string(opts.values[i]);
  }
  auto _backend_id = std::string(backend_id);

  auto ctx = ENGINE.CreateContext(_backend_id, dev_id, _opts);
  if (!ctx.IsOK()) {
    LOG(ERROR) << "[bridge] Call engine DestoryContext failed, "
               << ctx.GetError();
    throw nndevice_error(ctx.GetError());
  }
  VLOG(1) << "[bridge] Call engine CreateContext success.";

  return std::move(ctx.Get());
}

void DestoryContext(std::unique_ptr<Context> ctx) {
  VLOG(1) << "[bridge] Call DestoryContext in bridge cxx.";
  auto result = ENGINE.DestoryContext(std::move(ctx));
  if (!result.IsOK()) {
    LOG(ERROR) << "[bridge] Call engine DestoryContext failed, "
               << result.GetError();
    throw nndevice_error(result.GetError());
  }
  VLOG(1) << "[bridge] Call engine DestoryContext success.";
}

void CompileGraph(const std::unique_ptr<Context>& ctx,
                  const bridge::GraphWrapper& wrapper,
                  bridge::CompileCallback cb,
                  rust::Box<bridge::RustCompileCallback> rust_cb) {
  VLOG(1) << "[bridge] Call CompileGraph in bridge cxx.";
  auto graph = FromWrapper(wrapper);

  auto result = ENGINE.CompileGraph(ctx, graph, [cb, &rust_cb](Err e) {
    cb(std::move(rust_cb), static_cast<int>(e));
  });
  if (!result.IsOK()) {
    LOG(ERROR) << "[bridge] Call engine CompileGraph failed, "
               << result.GetError();
    throw nndevice_error(result.GetError());
  }
  VLOG(1) << "[bridge] Call engine CompileGraph success.";
}

void Execute(const std::unique_ptr<Context>& ctx,
             const rust::Vec<bridge::TensorWrapper>& inputs_wrapper,
             bridge::ExecuteCallback cb,
             rust::Box<bridge::RustExecuteCallback> rust_cb) {
  VLOG(1) << "[bridge] Call Execute in bridge cxx.";
  std::vector<std::shared_ptr<Tensor>> inputs;
  for (auto& wrapper : inputs_wrapper) {
    inputs.push_back(FromWrapper(wrapper));
  }
  auto result = ENGINE.Execute(
      ctx, inputs,
      [cb, &rust_cb](Result<std::vector<std::shared_ptr<Tensor>>> outputs) {
        if (!outputs.IsOK()) {
          cb(std::move(rust_cb), {}, static_cast<int>(outputs.GetError()));
          return;
        }

        rust::Vec<RustTensor> rust_outputs;
        for (auto tensor : outputs.Get()) {
          rust_outputs.push_back(ToRustTensor(tensor));
        }
        cb(std::move(rust_cb), rust_outputs, 0);
      });

  if (!result.IsOK()) {
    LOG(ERROR) << "[bridge] Call engine Execute failed, " << result.GetError();
    throw nndevice_error(result.GetError());
  }
  VLOG(1) << "[bridge] Call engine Execute success.";
}

#undef CLIENT

// }  // namespace bridge
}  // namespace nndevice
}  // namespace aichip
}  // namespace inos
