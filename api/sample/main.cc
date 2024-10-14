#include <glog/logging.h>
#include <inos/airuntime/airuntime.h>

void test_config() {
  LOG(INFO) << "test_config >>>>>>>>>>>>>>>>>>>>";
  Config* config = airuntime_config_new("sample", 1);

  airuntime_config_add_option(config, "key1", "value1");
  airuntime_config_add_option(config, "key2", "value2");

  CString* backend_id = airuntime_config_get_backend_id(config);
  LOG(INFO) << "backend_id: " << airuntime_cstring_get(backend_id);
  airuntime_cstring_destory(backend_id);
 
  LOG(INFO) << "dev_id: " << airuntime_config_get_dev_id(config);
  CString* key1 = airuntime_config_get_option(config, "key1");
  LOG(INFO) << "options key1: " << airuntime_cstring_get(key1);
  airuntime_cstring_destory(key1);

  CString* key2 = airuntime_config_get_option(config, "key2");
  LOG(INFO) << "options key2: " << airuntime_cstring_get(key2);
  airuntime_cstring_destory(key2);

  airuntime_config_print(config);

  airuntime_config_destory(config);
  LOG(INFO) << "test_config <<<<<<<<<<<<<<<<<<<<<<<";
}

void test_tensor() {
  LOG(INFO) << "test_tensor >>>>>>>>>>>>>>>>>>>>";
  uint32_t shape[] = {1, 1, 2, 2};
  Tensor* tensor = airuntime_tensor_new("data1", 4, shape, TensorFormat::NCHW,
                                        TensorDType::Float32);
  float data[2][2] = {{1.1, 2.2}, {3.3, 4.4}};
  airuntime_tensor_set_data(tensor, (uint8_t*)data, 4 * sizeof(float));

  CString* name = airuntime_tensor_get_name(tensor);
  LOG(INFO) << "name: " << airuntime_cstring_get(name);
  airuntime_cstring_destory(name);
  LOG(INFO) << "format: " << airuntime_tensor_get_format(tensor);
  LOG(INFO) << "dtype: " << airuntime_tensor_get_dtype(tensor);
  const uint32_t* r_shape;
  uint32_t dim = airuntime_tensor_get_shape(tensor, &r_shape);
  std::string s = "";
  for (uint32_t i = 0; i < dim; i++) {
    s += std::to_string(r_shape[i]) + ",";
  }
  LOG(INFO) << "dim: " << dim << ", shape: [" << s << "]";
  uint8_t* r_data;
  uint32_t len = airuntime_tensor_get_data(tensor, &r_data);
  uint32_t data_len = airuntime_tensor_get_shape_len(tensor);
  s = "";
  for (size_t i = 0; i < data_len; i++) {
    s += std::to_string(((float*)r_data)[i]) + ",";
  }
  LOG(INFO) << "data: [" << s << "]";
  airuntime_tensor_destory(tensor);
  LOG(INFO) << "test_tensor <<<<<<<<<<<<<<<<<<<<<<<";
}

Tensor* create_tenosr(const char* name, uint32_t dim, uint32_t shape[],
                      float* data, uint32_t data_len) {
  Tensor* tensor = airuntime_tensor_new(name, dim, shape, TensorFormat::NCHW,
                                        TensorDType::Float32);
  airuntime_tensor_set_data(tensor, (uint8_t*)data, data_len);

  return tensor;
}

void load_cb(AiruntimeErrCode code, void* userdata) {
  LOG(INFO) << "load_cb >>>>>>>>>>>>>>>>>>>>";
  LOG(INFO) << "load_cb code: " << code;
  LOG(INFO) << "userdata: " << (char*)userdata;
  LOG(INFO) << "load_cb <<<<<<<<<<<<<<<<<<<<<<<";
}

void run_cb(TensorVec* outputs, unsigned int len, AiruntimeErrCode code,
            void* userdata) {
  LOG(INFO) << "run_cb >>>>>>>>>>>>>>>>>>>>";
  LOG(INFO) << "run_cb code: " << code;
  LOG(INFO) << "userdata: " << (char*)userdata;
  LOG(INFO) << "run_cb outputs len: " << len;
  LOG(INFO) << "run_cb outputs len: " << airuntime_tensorvec_get_len(outputs);
  for (uint32_t i = 0; i < len; i++) {
    LOG(INFO) << "1111: " << outputs;
    auto tensor = airuntime_tensorvec_get(outputs, i);
    LOG(INFO) << "2222: " << tensor;
    CString* name = airuntime_tensor_get_name((Tensor*)tensor);
    LOG(INFO) << "name: " << airuntime_cstring_get(name);
    airuntime_cstring_destory(name);
    uint8_t* data;
    uint32_t len = airuntime_tensor_get_data((Tensor*)tensor, &data);
    LOG(INFO) << "data for byte len: " << len;
    uint32_t data_len = airuntime_tensor_get_shape_len((Tensor*)tensor);
    LOG(INFO) << "data_len: " << data_len;
  }
  LOG(INFO) << "run_cb <<<<<<<<<<<<<<<<<<<<<<<";
}

void test_model() {
  LOG(INFO) << "test_model >>>>>>>>>>>>>>>>>>>>";
  Backends* candidates;
  LOG(INFO) << "code: " << airuntime_get_candidate_backends(&candidates);
  auto candidates_len = airuntime_backends_get_len(candidates);
  LOG(INFO) << "candidates len: " << candidates_len;
  if (candidates_len <= 0) {
    exit(-1);
  }
  const char* candidate = airuntime_backends_get(candidates, 0);
  LOG(INFO) << "candidate: " << candidate;

  Config* config = airuntime_config_new(candidate, 1);

  Context* ctx;
  LOG(INFO) << "load code: "
            << airuntime_load(&ctx, "/var/test/yolov3-fp32.onnx", config,
                              load_cb, (void*)"load user data");

  uint32_t shape[] = {1, 1, 2, 2};
  float data[2][2] = {{1.1, 2.2}, {3.3, 4.4}};
  uint32_t data_len = 4 * sizeof(float);
  Tensor* inputs[] = {
      create_tenosr("data1", 4, shape, (float*)data, data_len),
      create_tenosr("data2", 4, shape, (float*)data, data_len),
  };

  LOG(INFO) << "run code: "
            << airuntime_run(ctx, inputs, 2, run_cb, (void*)"run user data");

  LOG(INFO) << "destory...";
  airuntime_context_destory(ctx);
  airuntime_config_destory(config);
  airuntime_backends_destory(candidates);

  LOG(INFO) << "test_model <<<<<<<<<<<<<<<<<<<<<<<";
}

int main(int argc, const char** argv) {
  LOG(INFO) << "main start";

  test_config();
  test_tensor();
  test_model();

  LOG(INFO) << "main end";
  return 0;
}
