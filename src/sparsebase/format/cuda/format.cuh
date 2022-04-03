#include "sparsebase/format/format.h"
#include "sparsebase/context/context.h"
#include "sparsebase/context/cuda/context.cuh"
#ifndef SPARSEBASE_SPARSEBASE_FORMAT_CUDA_FORMAT_H_
#define SPARSEBASE_SPARSEBASE_FORMAT_CUDA_FORMAT_H_

namespace sparsebase {


namespace format {
namespace cuda {

  template <typename T> struct CUDADeleter {
    void operator()(T *obj) {
      cudaFree(obj);
    }
  };

  template <typename IDType, typename NNZType, typename ValueType>
  class CUDACSR : public FormatImplementation<CUDACSR<IDType, NNZType, ValueType>> {
    public:
    CUDACSR(IDType n, IDType m, NNZType nnz, NNZType *row_ptr, IDType *col, ValueType *vals, context::cuda::CUDAContext context,
        Ownership own = kNotOwned);
    CUDACSR(const CUDACSR<IDType, NNZType, ValueType> &);
    CUDACSR(CUDACSR<IDType, NNZType, ValueType> &&);
    CUDACSR<IDType, NNZType, ValueType> &
    operator=(const CUDACSR<IDType, NNZType, ValueType> &);
    Format *Clone() const override;
    virtual ~CUDACSR();
    NNZType *get_row_ptr() const;
    IDType *get_col() const;
    ValueType *get_vals() const;

    NNZType *release_row_ptr();
    IDType *release_col();
    ValueType *release_vals();

    void set_row_ptr(NNZType *, context::cuda::CUDAContext context, Ownership own = kNotOwned);
    void set_col(IDType *, context::cuda::CUDAContext context, Ownership own = kNotOwned);
    void set_vals(ValueType *, context::cuda::CUDAContext context, Ownership own = kNotOwned);

    virtual bool ColIsOwned();
    virtual bool RowPtrIsOwned();
    virtual bool ValsIsOwned();

    context::cuda::CUDAContext* get_cuda_context() const;

  protected:
    std::unique_ptr<NNZType, std::function<void(NNZType *)>> row_ptr_;
    std::unique_ptr<IDType, std::function<void(IDType *)>> col_;
    std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;

  };

  template <typename ValueType>
  class CUDAArray : public FormatImplementation<CUDAArray<ValueType>> {
  public:
    CUDAArray(DimensionType nnz, ValueType *row_ptr, context::cuda::CUDAContext context, Ownership own = kNotOwned);
    CUDAArray(const CUDAArray<ValueType> &);
    CUDAArray(CUDAArray<ValueType> &&);
    CUDAArray<ValueType> &
    operator=(const CUDAArray<ValueType> &);
    Format *Clone() const override;
    virtual ~CUDAArray();
    ValueType *get_vals() const;

    ValueType *release_vals();

    void set_vals(ValueType *, Ownership own = kNotOwned);

    virtual bool ValsIsOwned();

  protected:
    std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;
  };

};
}

}
#endif // SPARSEBASE_SPARSEBASE_FORMAT_CUDA_FORMAT_H_