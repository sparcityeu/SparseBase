#ifndef SPARSEBASE_PROJECT_CONVERTER_ORDER_ONE_CUDA_CUH
#define SPARSEBASE_PROJECT_CONVERTER_ORDER_ONE_CUDA_CUH
#include "converter.h"
#include "sparsebase/config.h"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
namespace sparsebase::converter {

template <typename ValueType>
format::Format *CUDAArrayArrayConditionalFunction(format::Format *source,
                                                  context::Context *context);
template <typename ValueType>
format::Format *ArrayCUDAArrayConditionalFunction(format::Format *source,
                                                  context::Context *context);
}
#ifdef _HEADER_ONLY
#include "converter_order_one_cuda.cu"
#endif
#endif  // SPARSEBASE_PROJECT_CONVERTER_ORDER_ONE_CUDA_CUH
