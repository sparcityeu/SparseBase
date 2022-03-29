#ifndef _CUDA_H_FILE
#define _CUDA_H_FILE
#include "sparsebase/sparse_format.h"

namespace sparsebase {

namespace context {
  struct CUDAContext : ContextImplementation<CUDAContext>{
    int device_id;
    CUDAContext(int did);
    virtual bool IsEquivalent(Context *) const;
  };
}

namespace format {
  template <typename T> struct CUDADeleter {
    void operator()(T *obj) {
      cudaFree(obj);
    }
  };

  template <typename IDType, typename NNZType, typename ValueType>
  class CUDACSR : public FormatImplementation<CUDACSR<IDType, NNZType, ValueType>> {
    public:
    CUDACSR(IDType n, IDType m, NNZType nnz, NNZType *row_ptr, IDType *col, ValueType *vals, context::CUDAContext context,
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

    void set_row_ptr(NNZType *, context::CUDAContext context, Ownership own = kNotOwned);
    void set_col(IDType *, context::CUDAContext context, Ownership own = kNotOwned);
    void set_vals(ValueType *, context::CUDAContext context, Ownership own = kNotOwned);

    virtual bool ColIsOwned();
    virtual bool RowPtrIsOwned();
    virtual bool ValsIsOwned();

    context::CUDAContext* get_cuda_context() const;

  protected:
    std::unique_ptr<NNZType, std::function<void(NNZType *)>> row_ptr_;
    std::unique_ptr<IDType, std::function<void(IDType *)>> col_;
    std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;

  };

  template <typename ValueType>
  class CUDAArray : public FormatImplementation<CUDAArray<ValueType>> {
  public:
    CUDAArray(DimensionType nnz, ValueType *row_ptr, context::CUDAContext context, Ownership own = kNotOwned);
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
#endif