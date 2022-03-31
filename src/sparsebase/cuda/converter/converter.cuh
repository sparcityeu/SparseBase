#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"
#include "sparsebase/cuda/format.cuh"

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