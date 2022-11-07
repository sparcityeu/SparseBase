#include "converter_cuda.cuh"
#include "sparsebase/context/context.h"
#include "sparsebase/context/cuda_context_cuda.cuh"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

namespace sparsebase::converter {


bool CUDAPeerToPeer(context::Context *from, context::Context *to) {
  if (!(to->get_id() ==
            context::CUDAContext::get_id_static() ||
        from->get_id() ==
            context::CUDAContext::get_id_static()))
    return false;
  auto from_gpu = static_cast<context::CUDAContext *>(from);
  auto to_gpu = static_cast<context::CUDAContext *>(to);
  int can_access;
  cudaDeviceCanAccessPeer(&can_access, from_gpu->device_id, to_gpu->device_id);
  return can_access;
}
}  // namespace sparsebase
