#include "sparsebase/converter/converter.h"
#include "sparsebase/converter/converter_order_one.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/array.h"

#ifdef USE_CUDA
#include "sparsebase/converter/converter_cuda.cuh"
#include "sparsebase/converter/converter_order_one_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"
#endif
namespace sparsebase::converter {

template <typename ValueType>
Converter *ConverterOrderOne<ValueType>::Clone() const {
  return new ConverterOrderOne<ValueType>(*this);
}

template <typename ValueType>
void ConverterOrderOne<ValueType>::Reset() {
  this->ResetConverterOrderOne();
}
template <typename ValueType>
void ConverterOrderOne<ValueType>::ResetConverterOrderOne() {
#ifdef USE_CUDA
  this->RegisterConversionFunction(
      format::Array<ValueType>::get_id_static(),
      format::CUDAArray<ValueType>::get_id_static(),
      converter::ArrayCUDAArrayConditionalFunction<ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() ==
               context::CUDAContext::get_id_static();
      });
  this->RegisterConversionFunction(
      format::CUDAArray<ValueType>::get_id_static(),
      format::Array<ValueType>::get_id_static(),
      converter::CUDAArrayArrayConditionalFunction<ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() ==
               context::CPUContext::get_id_static();
      });
#endif
}

template <typename ValueType>
ConverterOrderOne<ValueType>::ConverterOrderOne() {
  this->ResetConverterOrderOne();
}

#ifndef _HEADER_ONLY
#include "init/converter_order_one.inc"
#endif
}