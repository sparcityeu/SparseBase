#include "sparsebase/format/format_order_one.h"
#include "sparsebase/utils/utils_cuda.cuh"
#include "sparsebase/context/cuda_context_cuda.cuh"

#ifndef SPARSEBASE_PROJECT_CUDA_ARRAY_CUDA_H
#define SPARSEBASE_PROJECT_CUDA_ARRAY_CUDA_H

namespace sparsebase::format {

template <typename ValueType>
class CUDAArray : public utils::IdentifiableImplementation<CUDAArray<ValueType>,
    FormatOrderOne<ValueType>> {
public:
CUDAArray(DimensionType nnz, ValueType *row_ptr,
          context::CUDAContext context, Ownership own = kNotOwned);
CUDAArray(const CUDAArray<ValueType> &);
CUDAArray(CUDAArray<ValueType> &&);
CUDAArray<ValueType> &operator=(const CUDAArray<ValueType> &);
Format *Clone() const override;
virtual ~CUDAArray();
ValueType *get_vals() const;

ValueType *release_vals();

void set_vals(ValueType *, Ownership own = kNotOwned);

virtual bool ValsIsOwned();

protected:
std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;
};
}

#ifdef _HEADER_ONLY
#include "cuda_array_cuda.cu"
#endif
#endif  // SPARSEBASE_PROJECT_CUDA_ARRAY_CUDA_H
