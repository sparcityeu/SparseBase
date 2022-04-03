#include "sparsebase/format/format.h"
#include "sparsebase/utils/converter/converter.h"
#include "sparsebase/format/cuda/format.cuh"
#ifndef SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CUDA_CONVERTER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CUDA_CONVERTER_H_

namespace sparsebase{
namespace utils {
namespace converter{
namespace cuda{
template <typename ValueType>
sparsebase::format::Format *
CUDAArrayArrayConditionalFunction(sparsebase::format::Format *source, sparsebase::context::Context*context);

template <typename ValueType>
sparsebase::format::Format *
ArrayCUDAArrayConditionalFunction(sparsebase::format::Format *source, sparsebase::context::Context*context);

template <typename IDType, typename NNZType, typename ValueType>
sparsebase::format::Format *
CsrCUDACsrConditionalFunction(sparsebase::format::Format *source, sparsebase::context::Context*context);

template <typename IDType, typename NNZType, typename ValueType>
sparsebase::format::Format *
CUDACsrCsrConditionalFunction(sparsebase::format::Format *source, sparsebase::context::Context*context);

template <typename IDType, typename NNZType, typename ValueType>
sparsebase::format::Format *
CUDACsrCUDACsrConditionalFunction(sparsebase::format::Format *source, sparsebase::context::Context*context);

bool CUDAPeerToPeer(sparsebase::context::Context* from, sparsebase::context::Context* to);

}

}
}
}

#endif // SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_