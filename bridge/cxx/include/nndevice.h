/**
 * @brief
 * @author linjh (linjh@zhejianglab.com)
 * @version 0.1
 * @date 2023-06-25
 */
#ifndef NNDEVICE_H_
#define NNDEVICE_H_

#include <inos/ai_chip/nn_device/context.h>

#include <memory>

#include "nndevice/ffi.rs.h"
#include "rust/cxx.h"

namespace inos {
namespace aichip {
namespace nndevice {

namespace bridge {
using CompileCallback =
    rust::Fn<void(rust::Box<bridge::RustCompileCallback>, int)>;
using ExecuteCallback =
    rust::Fn<void(rust::Box<bridge::RustExecuteCallback>,
                  const rust::Vec<RustTensor>&, int)>;
}  // namespace bridge

rust::Vec<rust::String> GetCandidateBackends();

std::unique_ptr<Context> CreateContext(rust::Str backend_id, DevId dev_id,
                                       const RustOptions& opts);

void DestoryContext(std::unique_ptr<Context> ctx);

void CompileGraph(const std::unique_ptr<Context>& ctx,
                  const bridge::GraphWrapper& wrapper,
                  bridge::CompileCallback cb,
                  rust::Box<bridge::RustCompileCallback> rust_cb);

void Execute(const std::unique_ptr<Context>& ctx,
             const rust::Vec<bridge::TensorWrapper>& inputs_wrapper,
             bridge::ExecuteCallback cb,
             rust::Box<bridge::RustExecuteCallback> rust_cb);

}  // namespace nndevice
}  // namespace aichip
}  // namespace inos

#endif  // !NNDEVICE_H_
