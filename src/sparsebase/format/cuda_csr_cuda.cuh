/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/context/cuda_context_cuda.cuh"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/utils.h"
#include "sparsebase/utils/utils_cuda.cuh"

#ifndef SPARSEBASE_SPARSEBASE_FORMAT_CUDA_FORMAT_H_
#define SPARSEBASE_SPARSEBASE_FORMAT_CUDA_FORMAT_H_

namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
class CUDACSR : public utils::IdentifiableImplementation<
                    CUDACSR<IDType, NNZType, ValueType>,
                    FormatOrderTwo<IDType, NNZType, ValueType>> {
 public:
  CUDACSR(IDType n, IDType m, NNZType nnz, NNZType *row_ptr, IDType *col,
          ValueType *vals, context::CUDAContext context,
          Ownership own = kOwned);
  CUDACSR(const CUDACSR<IDType, NNZType, ValueType> &);
  CUDACSR(CUDACSR<IDType, NNZType, ValueType> &&);
  CUDACSR<IDType, NNZType, ValueType> &operator=(
      const CUDACSR<IDType, NNZType, ValueType> &);
  Format *Clone() const override;
  virtual ~CUDACSR();
  NNZType *get_row_ptr() const;
  IDType *get_col() const;
  ValueType *get_vals() const;

  NNZType *release_row_ptr();
  IDType *release_col();
  ValueType *release_vals();

  void set_row_ptr(NNZType *, context::CUDAContext context,
                   Ownership own = kNotOwned);
  void set_col(IDType *, context::CUDAContext context,
               Ownership own = kNotOwned);
  void set_vals(ValueType *, context::CUDAContext context,
                Ownership own = kNotOwned);

  virtual bool ColIsOwned();
  virtual bool RowPtrIsOwned();
  virtual bool ValsIsOwned();

  context::CUDAContext *get_cuda_context() const;

 protected:
  std::unique_ptr<NNZType, std::function<void(NNZType *)>> row_ptr_;
  std::unique_ptr<IDType, std::function<void(IDType *)>> col_;
  std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;
};

}  // namespace sparsebase::format
#ifdef _HEADER_ONLY
#include "cuda_csr_cuda.cu"
#endif
#endif  // SPARSEBASE_SPARSEBASE_FORMAT_CUDA_FORMAT_H_