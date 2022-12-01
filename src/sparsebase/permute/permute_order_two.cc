#include "sparsebase/permute/permute_order_two.h"

#include "sparsebase/format/array.h"
#include "sparsebase/format/csr.h"

namespace sparsebase::permute {
template <typename IDType, typename NNZType, typename ValueType>
PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwo(
    IDType *row_order, IDType *col_order) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      PermuteOrderTwoCSR);
  this->params_ =
      std::make_unique<PermuteOrderTwoParams<IDType>>(row_order, col_order);
}
template <typename IDType, typename NNZType, typename ValueType>
PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwo(
    PermuteOrderTwoParams<IDType> params) {
  PermuteOrderTwo(params.row_order, params.col_order);
}
template <typename IDType, typename NNZType, typename ValueType>
format::FormatOrderTwo<IDType, NNZType, ValueType>
    *PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwoCSR(
        std::vector<format::Format *> formats, utils::Parameters *params) {
  auto *sp = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  auto row_order =
      static_cast<PermuteOrderTwoParams<IDType> *>(params)->row_order;
  auto col_order =
      static_cast<PermuteOrderTwoParams<IDType> *>(params)->col_order;
  std::vector<format::DimensionType> dimensions = sp->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = sp->get_num_nnz();
  NNZType *xadj = sp->get_row_ptr();
  IDType *adj = sp->get_col();
  ValueType *vals = sp->get_vals();
  NNZType *nxadj = new NNZType[n + 1]();
  IDType *nadj = new IDType[nnz]();
  ValueType *nvals = nullptr;
  if constexpr (!std::is_same_v<void, ValueType>) {
    if (sp->get_vals() != nullptr) nvals = new ValueType[nnz]();
  }
  std::function<IDType(IDType)> get_i_row_order;
  std::function<IDType(IDType)> get_col_order;
  IDType *inverse_row_order;
  if (row_order != nullptr) {
    inverse_row_order = new IDType[n]();
    for (IDType i = 0; i < n; i++) inverse_row_order[row_order[i]] = i;
    get_i_row_order = [&inverse_row_order](IDType i) -> IDType {
      return inverse_row_order[i];
    };
  } else {
    get_i_row_order = [&inverse_row_order](IDType i) -> IDType { return i; };
  }
  if (col_order != nullptr) {
    get_col_order = [&col_order](IDType i) -> IDType { return col_order[i]; };
  } else {
    get_col_order = [](IDType i) -> IDType { return i; };
  }
  // IDType *inverse_col_order = new IDType[n]();
  // for (IDType i = 0; i < n; i++)
  //  inverse_col_order[col_order[i]] = i;
  NNZType c = 0;
  for (IDType i = 0; i < n; i++) {
    IDType u = get_i_row_order(i);
    nxadj[i + 1] = nxadj[i] + (xadj[u + 1] - xadj[u]);
    for (NNZType v = xadj[u]; v < xadj[u + 1]; v++) {
      nadj[c] = get_col_order(adj[v]);
      if constexpr (!std::is_same_v<void, ValueType>) {
        if (sp->get_vals() != nullptr) nvals[c] = vals[v];
      }
      c++;
    }
  }
  if (row_order == nullptr) delete[] inverse_row_order;
  format::CSR<IDType, NNZType, ValueType> *csr =
      new format::CSR(n, m, nxadj, nadj, nvals);
  return csr;
}

#if !defined(_HEADER_ONLY)
#include "init/permute_order_two.inc"
#endif
}  // namespace sparsebase::permute
