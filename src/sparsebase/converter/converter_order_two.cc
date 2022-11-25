#include "sparsebase/converter/converter_order_two.h"

#include "sparsebase/converter/converter.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

#ifdef USE_CUDA
#include "sparsebase/converter/converter_cuda.cuh"
#include "sparsebase/converter/converter_order_one_cuda.cuh"
#include "sparsebase/converter/converter_order_two_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#endif

namespace sparsebase::converter {

template <typename IDType, typename NNZType, typename ValueType>
format::Format *CooCscFunctionConditional(format::Format *source,
                                          context::Context *context) {
  auto *coo = source->AsAbsolute<format::COO<IDType, NNZType, ValueType>>();

  std::vector<format::DimensionType> dimensions = coo->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = coo->get_num_nnz();
  auto coo_col = coo->get_col();
  auto coo_row = coo->get_row();
  auto coo_vals = coo->get_vals();
  NNZType *col_ptr = new NNZType[n + 1];
  NNZType *col_counter = new NNZType[n]();
  IDType *row = new IDType[nnz];
  ValueType *vals;
  if constexpr (!std::is_same_v<void, ValueType>) {
    if (coo->get_vals() != nullptr) {
      vals = new ValueType[nnz];
    } else {
      vals = nullptr;
    }
  } else {
    vals = nullptr;
  }

  std::fill(col_ptr, col_ptr + n + 1, 0);
  std::fill(row, row + nnz, 0);

  for (IDType i = 0; i < nnz; i++) {
    col_ptr[coo_col[i] + 1]++;
  }
  for (IDType i = 1; i <= n; i++) {
    col_ptr[i] += col_ptr[i - 1];
  }

  for (IDType i = 0; i < nnz; i++) {
    auto this_nnz_col = coo_col[i];
    auto this_nnz_row = coo_row[i];
    row[col_ptr[this_nnz_col] + col_counter[this_nnz_col]++] = this_nnz_row;
    if constexpr (!std::is_same_v<void, ValueType>) {
      if (coo_vals != nullptr) {
        vals[col_ptr[this_nnz_col] + col_counter[this_nnz_col] - 1] =
            coo_vals[i];
      }
    }
  }
  auto csc = new format::CSC<IDType, NNZType, ValueType>(
      n, m, col_ptr, row, vals, format::kOwned, false);
  return csc;
}
template <typename IDType, typename NNZType, typename ValueType>
format::Format *CsrCooFunctionConditional(format::Format *source,
                                          context::Context *context) {
  auto *csr = source->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  std::vector<format::DimensionType> dimensions = csr->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = csr->get_num_nnz();

  auto col = new IDType[nnz];
  auto row = new IDType[nnz];
  auto csr_row_ptr = csr->get_row_ptr();
  auto csr_col = csr->get_col();

  IDType count = 0;
  for (IDType i = 0; i < n; i++) {
    IDType start = csr_row_ptr[i];
    IDType end = csr_row_ptr[i + 1];

    for (IDType j = start; j < end; j++) {
      row[count] = i;
      count++;
    }
  }

  for (IDType i = 0; i < nnz; i++) {
    col[i] = csr_col[i];
  }

  // if (csr->vals != nullptr)
  ValueType *vals = nullptr;
  if constexpr (!std::is_same_v<void, ValueType>) {
    if (csr->get_vals() != nullptr) {
      vals = new ValueType[nnz];
      auto csr_vals = csr->get_vals();
      for (NNZType i = 0; i < nnz; i++) {
        vals[i] = csr_vals[i];
      }
    } else {
      vals = nullptr;
    }
  }
  auto *coo = new format::COO<IDType, NNZType, ValueType>(n, m, nnz, row, col,
                                                          vals, format::kOwned);

  return coo;
}
template <typename IDType, typename NNZType, typename ValueType>
format::Format *CsrCscFunctionConditional(format::Format *source,
                                          context::Context *context) {
  auto *csr = source->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  auto coo =
      CsrCooFunctionConditional<IDType, NNZType, ValueType>(csr, context);

  return CooCscFunctionConditional<IDType, NNZType, ValueType>(coo, context);
}

template <typename IDType, typename NNZType, typename ValueType>
format::Format *CsrCooMoveConditionalFunction(format::Format *source,
                                              context::Context *) {
  auto *csr = source->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  auto col = csr->release_col();
  auto vals = csr->release_vals();
  std::vector<format::DimensionType> dimensions = csr->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = csr->get_num_nnz();

  auto row = new IDType[nnz];
  auto csr_row_ptr = csr->get_row_ptr();

  IDType count = 0;
  for (IDType i = 0; i < n; i++) {
    IDType start = csr_row_ptr[i];
    IDType end = csr_row_ptr[i + 1];

    for (IDType j = start; j < end; j++) {
      row[count] = i;
      count++;
    }
  }

  // if (csr->vals != nullptr)
  auto *coo = new format::COO<IDType, NNZType, ValueType>(n, m, nnz, row, col,
                                                          vals, format::kOwned);

  return coo;
}

template <typename IDType, typename NNZType, typename ValueType>
format::Format *CooCsrFunctionConditional(format::Format *source,
                                          context::Context *context) {
  auto *coo = source->AsAbsolute<format::COO<IDType, NNZType, ValueType>>();

  std::vector<format::DimensionType> dimensions = coo->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = coo->get_num_nnz();
  auto coo_col = coo->get_col();
  auto coo_row = coo->get_row();
  NNZType *row_ptr = new NNZType[n + 1];
  IDType *col = new IDType[nnz];
  ValueType *vals;

  std::fill(row_ptr, row_ptr + n + 1, 0);
  std::fill(col, col + nnz, 0);

  for (IDType i = 0; i < nnz; i++) {
    col[i] = coo_col[i];
    row_ptr[coo_row[i]]++;
  }

  for (IDType i = 1; i <= n; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  for (IDType i = n; i > 0; i--) {
    row_ptr[i] = row_ptr[i - 1];
  }
  row_ptr[0] = 0;

  if constexpr (!std::is_same_v<void, ValueType>) {
    if (coo->get_vals() != nullptr) {
      vals = new ValueType[nnz];
      auto coo_vals = coo->get_vals();
      std::fill(vals, vals + nnz, 0);
      for (NNZType i = 0; i < nnz; i++) {
        vals[i] = coo_vals[i];
      }
    } else {
      vals = nullptr;
    }
  } else {
    vals = nullptr;
  }

  auto csr = new format::CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col,
                                                         vals, format::kOwned);
  return csr;
}

template <typename IDType, typename NNZType, typename ValueType>
format::Format *CooCsrMoveConditionalFunction(format::Format *source,
                                              context::Context *) {
  auto *coo = source->AsAbsolute<format::COO<IDType, NNZType, ValueType>>();

  std::vector<format::DimensionType> dimensions = coo->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = coo->get_num_nnz();
  auto coo_row = coo->get_row();
  NNZType *row_ptr = new NNZType[n + 1];
  IDType *col = coo->release_col();
  ValueType *vals = coo->release_vals();

  std::fill(row_ptr, row_ptr + n + 1, 0);

  for (IDType i = 0; i < nnz; i++) {
    row_ptr[coo_row[i]]++;
  }

  for (IDType i = 1; i <= n; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  for (IDType i = n; i > 0; i--) {
    row_ptr[i] = row_ptr[i - 1];
  }
  row_ptr[0] = 0;

  auto csr = new format::CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col,
                                                         vals, format::kOwned);
  return csr;
}
template <typename IDType, typename NNZType, typename ValueType>
Converter *ConverterOrderTwo<IDType, NNZType, ValueType>::Clone() const {
  return new ConverterOrderTwo<IDType, NNZType, ValueType>(*this);
}

template <typename IDType, typename NNZType, typename ValueType>
void ConverterOrderTwo<IDType, NNZType, ValueType>::Reset() {
  this->ResetConverterOrderTwo();
}

template <typename IDType, typename NNZType, typename ValueType>
void ConverterOrderTwo<IDType, NNZType, ValueType>::ResetConverterOrderTwo() {
  this->RegisterConversionFunction(
      format::COO<IDType, NNZType, ValueType>::get_id_static(),
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      CooCsrFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      });
  this->RegisterConversionFunction(
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      format::COO<IDType, NNZType, ValueType>::get_id_static(),
      CsrCooFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      });
  this->RegisterConversionFunction(
      format::COO<IDType, NNZType, ValueType>::get_id_static(),
      format::CSC<IDType, NNZType, ValueType>::get_id_static(),
      CooCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      });
  this->RegisterConversionFunction(
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      format::CSC<IDType, NNZType, ValueType>::get_id_static(),
      CsrCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      });
#ifdef USE_CUDA
  this->RegisterConversionFunction(
      format::CUDACSR<IDType, NNZType, ValueType>::get_id_static(),
      format::CUDACSR<IDType, NNZType, ValueType>::get_id_static(),
      converter::CUDACsrCUDACsrConditionalFunction<IDType, NNZType, ValueType>,
      converter::CUDAPeerToPeer);
  this->RegisterConversionFunction(
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      format::CUDACSR<IDType, NNZType, ValueType>::get_id_static(),
      converter::CsrCUDACsrConditionalFunction<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CUDAContext::get_id_static();
      });
  this->RegisterConversionFunction(
      format::CUDACSR<IDType, NNZType, ValueType>::get_id_static(),
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      converter::CUDACsrCsrConditionalFunction<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      });
#endif
  this->RegisterConversionFunction(
      format::COO<IDType, NNZType, ValueType>::get_id_static(),
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      CooCsrMoveConditionalFunction<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      },
      true);
  this->RegisterConversionFunction(
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      format::COO<IDType, NNZType, ValueType>::get_id_static(),
      CsrCooMoveConditionalFunction<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      },
      true);
  this->RegisterConversionFunction(
      format::COO<IDType, NNZType, ValueType>::get_id_static(),
      format::CSC<IDType, NNZType, ValueType>::get_id_static(),
      CooCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      },
      true);
  this->RegisterConversionFunction(
      format::CSR<IDType, NNZType, ValueType>::get_id_static(),
      format::CSC<IDType, NNZType, ValueType>::get_id_static(),
      CsrCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *to) -> bool {
        return to->get_id() == context::CPUContext::get_id_static();
      },
      true);
}

template <typename IDType, typename NNZType, typename ValueType>
ConverterOrderTwo<IDType, NNZType, ValueType>::ConverterOrderTwo() {
  this->ResetConverterOrderTwo();
}

#ifndef _HEADER_ONLY
#include "init/converter_order_two.inc"
#endif
}  // namespace sparsebase::converter