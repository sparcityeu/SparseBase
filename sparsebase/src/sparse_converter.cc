#include "sparsebase/sparse_converter.h"
#include "sparsebase/sparse_format.h"
#include <iostream>
#include <set>

using namespace std;

namespace sparsebase {

size_t FormatHash::operator()(Format f) const { return f; }

template <typename IDType, typename NNZType, typename ValueType>
SparseFormat<IDType, NNZType, ValueType> *CsrCooFunctor<IDType, NNZType, ValueType>::operator()(
    SparseFormat<IDType, NNZType, ValueType> *source) {
  CSR<IDType, NNZType, ValueType> *csr =
      dynamic_cast<CSR<IDType, NNZType, ValueType> *>(source);
  COO<IDType, NNZType, ValueType> *coo = new COO<IDType, NNZType, ValueType>();

  std::vector<IDType> dimensions = csr->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = csr->get_num_nnz();

  coo->col_ = new IDType[nnz];
  coo->row_ = new IDType[nnz];

  IDType count = 0;
  for (IDType i = 0; i < n; i++) {
    IDType start = csr->row_ptr_[i];
    IDType end = csr->row_ptr_[i + 1];

    for (IDType j = start; j < end; j++) {
      coo->row_[count] = i;
      count++;
    }
  }

  for (IDType i = 0; i < m; i++) {
    coo->col_[i] = csr->col_[i];
  }

  // if (csr->vals != nullptr)
  if constexpr (!std::is_same_v<void, ValueType>) {
    if (csr->vals_ != nullptr){
      coo->vals_ = new ValueType[nnz];
      for (NNZType i = 0; i < nnz; i++) {
        coo->vals_[i] = csr->vals_[i];
      }
    } else{
      coo->vals_ = nullptr;
    }
  } else {
    coo->vals_ = nullptr;
  }
  std::vector<IDType> dims{n, m};
  coo->dimension_ = dims;
  coo->nnz_ = nnz;

  return coo;
}

// Ai -> row indices -> col
// Aj -> col indices -> is
// Ax -> nnz values -> vals

// Bp -> row -> row_ptr
// Bj -> col -> col
// Bx -> nnz -> vals
template <typename IDType, typename NNZType, typename ValueType>
SparseFormat<IDType, NNZType, ValueType> *CooCsrFunctor<IDType, NNZType, ValueType>::operator()(
    SparseFormat<IDType, NNZType, ValueType> *source) {
  COO<IDType, NNZType, ValueType> *coo =
      dynamic_cast<COO<IDType, NNZType, ValueType> *>(source);

  std::vector<IDType> dimensions = coo->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = coo->get_num_nnz();

  NNZType *row_ptr = new NNZType[n + 1];
  IDType *col = new IDType[m];
  ValueType *vals;

  fill(row_ptr, row_ptr + n + 1, 0);
  fill(col, col + m, 0);

  // We need to ensure that they are sorted
  // Maybe add a sort check and then not do this if it is already sorted
  std::vector<std::pair<IDType, IDType>> edges;
  for (IDType i = 0; i < nnz; i++) {
    edges.emplace_back(coo->col_[i], coo->row_[i]);
  }
  sort(edges.begin(), edges.end(), less<std::pair<IDType, IDType>>());

  for (IDType i = 0; i < m; i++) {
    col[i] = edges[i].second;
    row_ptr[edges[i].first]++;
  }

  for (IDType i = 1; i <= n; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  for (IDType i = n; i > 0; i--) {
    row_ptr[i] = row_ptr[i - 1];
  }
  row_ptr[0] = 0;

  if constexpr (!std::is_same_v<void, ValueType>) {
    if (coo->vals_ != nullptr){
      vals = new ValueType[nnz];
      fill(vals, vals + nnz, 0);
      for (NNZType i = 0; i < nnz; i++) {
        vals[i] = coo->vals_[i];
      }
    } else {
      vals = nullptr;
    }
  } else {
    vals = nullptr;
  }

  auto csr = new CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col, vals);
  return csr;
}

template <typename IDType, typename NNZType, typename ValueType>
SparseConverter<IDType, NNZType, ValueType>::SparseConverter() {
  this->RegisterConversionFunction(kCOOFormat, kCSRFormat,
                                     new CooCsrFunctor<IDType, NNZType, ValueType>());
  this->RegisterConversionFunction(kCSRFormat, kCOOFormat,
                                     new CsrCooFunctor<IDType, NNZType, ValueType>());
}

template <typename IDType, typename NNZType, typename ValueType>
SparseConverter<IDType, NNZType, ValueType>::~SparseConverter() {
  set<uintptr_t> deleted_ptrs;

  for (auto x : conversion_map_) {
    for (auto y : x.second) {
      ConversionFunctor<IDType, NNZType, ValueType> *ptr = y.second;
      if (deleted_ptrs.count((uintptr_t)ptr) == 0) {
        deleted_ptrs.insert((uintptr_t)ptr);
        delete ptr;
      }
    }
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void SparseConverter<IDType, NNZType, ValueType>::RegisterConversionFunction(
    Format from_format, Format to_format,
    ConversionFunctor<IDType, NNZType, ValueType> *conv_func) {
  if (conversion_map_.count(from_format) == 0) {
    conversion_map_.emplace(
        from_format,
        std::unordered_map<Format, ConversionFunctor<IDType, NNZType, ValueType> *,
                      FormatHash>());
  }

  if (conversion_map_[from_format].count(to_format) == 0) {
    conversion_map_[from_format].emplace(to_format, conv_func);
  } else {
    conversion_map_[from_format][to_format] = conv_func;
  }
}

template <typename IDType, typename NNZType, typename ValueType>
SparseFormat<IDType, NNZType, ValueType> *SparseConverter<IDType, NNZType, ValueType>::Convert(
    SparseFormat<IDType, NNZType, ValueType> *source, Format to_format) {
  if (to_format == source->get_format()) {
    return source;
  }

  try {
    ConversionFunctor<IDType, NNZType, ValueType> *conv_func =
        GetConversionFunction(source->get_format(), to_format);
    return (*conv_func)(source);
  } catch (...) {
    throw "Unsupported conversion error"; // TODO: Add decent exception
                                          // mechanism
  }
}

template <typename IDType, typename NNZType, typename ValueType>
ConversionFunctor<IDType, NNZType, ValueType> *
SparseConverter<IDType, NNZType, ValueType>::GetConversionFunction(Format from_format,
                                                             Format to_format) {
  try {
    return conversion_map_[from_format][to_format];
  } catch (...) {
    throw "Unsupported conversion error"; // TODO: Add decent exception
                                          // mechanism
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool SparseConverter<IDType, NNZType, ValueType>::CanConvert(Format from_format,
                                                      Format to_format) {
  if (conversion_map_.find(from_format) != conversion_map_.end()) {
    if (conversion_map_[from_format].find(to_format) !=
        conversion_map_[from_format].end()) {
      return true;
    }
  }
  return false;
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<SparseFormat<IDType, NNZType, ValueType> *>
SparseConverter<IDType, NNZType, ValueType>::ApplyConversionSchema(
    ConversionSchema cs,
    std::vector<SparseFormat<IDType, NNZType, ValueType> *> packed_sfs) {
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> ret;
  for (int i = 0; i < cs.size(); i++) {
    auto conversion = cs[i];
    if (std::get<0>(conversion)) {
      ret.push_back(this->Convert(packed_sfs[i], std::get<1>(conversion)));
    } else {
      ret.push_back(packed_sfs[i]);
    }
  }
  return ret;
}


#ifdef NDEBUG
#include "init/sparse_converter.inc"
#else
template class SparseConverter<int, int, int>;
template class CooCsrFunctor<int, int, int>;
template class CsrCooFunctor<int, int, int>;

template class SparseConverter<unsigned int, unsigned int, void>;
template class CooCsrFunctor<unsigned int, unsigned int, void>;
template class CsrCooFunctor<unsigned int, unsigned int, void>;

template class SparseConverter<unsigned int, unsigned int, unsigned int>;
template class CooCsrFunctor<unsigned int, unsigned int, unsigned int>;
template class CsrCooFunctor<unsigned int, unsigned int, unsigned int>;
#endif
} // namespace sparsebase
