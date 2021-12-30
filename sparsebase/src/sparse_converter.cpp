#include "sparsebase/sparse_converter.hpp"
#include "sparsebase/sparse_format.hpp"
#include <iostream>
#include <set>

using namespace std;

namespace sparsebase {

size_t FormatHash::operator()(Format f) const { return f; }

template <typename ID, typename NumNonZeros, typename Value>
SparseFormat<ID, NumNonZeros, Value> *CsrCooFunctor<ID, NumNonZeros, Value>::operator()(
    SparseFormat<ID, NumNonZeros, Value> *source) {
  CSR<ID, NumNonZeros, Value> *csr =
      dynamic_cast<CSR<ID, NumNonZeros, Value> *>(source);
  COO<ID, NumNonZeros, Value> *coo = new COO<ID, NumNonZeros, Value>();

  std::vector<ID> dimensions = csr->get_dimensions();
  ID n = dimensions[0];
  ID m = dimensions[1];
  NumNonZeros nnz = csr->get_num_nnz();

  coo->col_ = new ID[nnz];
  coo->row_ = new ID[nnz];
  if (csr->vals_ != nullptr)
    coo->vals_ = new NumNonZeros[nnz];
  else
    coo->vals_ = nullptr;

  ID count = 0;
  for (ID i = 0; i < n; i++) {
    ID start = csr->row_ptr_[i];
    ID end = csr->row_ptr_[i + 1];

    for (ID j = start; j < end; j++) {
      coo->row_[count] = i;
      count++;
    }
  }

  for (ID i = 0; i < m; i++) {
    coo->col_[i] = csr->col_[i];
  }

  // if (csr->vals != nullptr)
  if constexpr (!std::is_same_v<void, Value>) {
    if (coo->vals_ != nullptr)
      for (NumNonZeros i = 0; i < nnz; i++) {
        coo->vals_[i] = csr->vals_[i];
      }
  }
  std::vector<ID> dims{n, m};
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
template <typename ID, typename NumNonZeros, typename Value>
SparseFormat<ID, NumNonZeros, Value> *CooCsrFunctor<ID, NumNonZeros, Value>::operator()(
    SparseFormat<ID, NumNonZeros, Value> *source) {
  COO<ID, NumNonZeros, NumNonZeros> *coo =
      dynamic_cast<COO<ID, NumNonZeros, NumNonZeros> *>(source);

  std::vector<ID> dimensions = coo->get_dimensions();
  ID n = dimensions[0];
  ID m = dimensions[1];
  NumNonZeros nnz = coo->get_num_nnz();

  ID *row_ptr = new ID[n + 1];
  ID *col = new ID[m];
  NumNonZeros *vals;
  if (coo->vals_ != nullptr)
    vals = new NumNonZeros[nnz];
  else
    vals = nullptr;

  fill(row_ptr, row_ptr + n + 1, 0);
  fill(col, col + m, 0);
  if (coo->vals_ != nullptr)
    fill(vals, vals + nnz, 0);

  // We need to ensure that they are sorted
  // Maybe add a sort check and then not do this if it is already sorted
  std::vector<std::pair<ID, ID>> edges;
  for (ID i = 0; i < nnz; i++) {
    edges.emplace_back(coo->col_[i], coo->row_[i]);
  }
  sort(edges.begin(), edges.end(), less<std::pair<ID, ID>>());

  for (ID i = 0; i < m; i++) {
    col[i] = edges[i].second;
    row_ptr[edges[i].first]++;
  }

  for (ID i = 1; i <= n; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  for (ID i = n; i > 0; i--) {
    row_ptr[i] = row_ptr[i - 1];
  }
  row_ptr[0] = 0;

  if constexpr (!std::is_same_v<void, Value>) {
    if (coo->vals_ != nullptr)
      for (NumNonZeros i = 0; i < nnz; i++) {
        vals[i] = coo->vals_[i];
      }
  }

  auto csr = new CSR<ID, NumNonZeros, Value>(n, m, row_ptr, col, vals);
  return csr;
}

template <typename ID, typename NumNonZeros, typename Value>
SparseConverter<ID, NumNonZeros, Value>::SparseConverter() {
  this->RegisterConversionFunction(kCOOFormat, kCSRFormat,
                                     new CooCsrFunctor<ID, NumNonZeros, Value>());
  this->RegisterConversionFunction(kCSRFormat, kCOOFormat,
                                     new CsrCooFunctor<ID, NumNonZeros, Value>());
}

template <typename ID, typename NumNonZeros, typename Value>
SparseConverter<ID, NumNonZeros, Value>::~SparseConverter() {
  set<uintptr_t> deleted_ptrs;

  for (auto x : conversion_map_) {
    for (auto y : x.second) {
      ConversionFunctor<ID, NumNonZeros, Value> *ptr = y.second;
      if (deleted_ptrs.count((uintptr_t)ptr) == 0) {
        deleted_ptrs.insert((uintptr_t)ptr);
        delete ptr;
      }
    }
  }
}

template <typename ID, typename NumNonZeros, typename Value>
void SparseConverter<ID, NumNonZeros, Value>::RegisterConversionFunction(
    Format from_format, Format to_format,
    ConversionFunctor<ID, NumNonZeros, Value> *conv_func) {
  if (conversion_map_.count(from_format) == 0) {
    conversion_map_.emplace(
        from_format,
        std::unordered_map<Format, ConversionFunctor<ID, NumNonZeros, Value> *,
                      FormatHash>());
  }

  if (conversion_map_[from_format].count(to_format) == 0) {
    conversion_map_[from_format].emplace(to_format, conv_func);
  } else {
    conversion_map_[from_format][to_format] = conv_func;
  }
}

template <typename ID, typename NumNonZeros, typename Value>
SparseFormat<ID, NumNonZeros, Value> *SparseConverter<ID, NumNonZeros, Value>::Convert(
    SparseFormat<ID, NumNonZeros, Value> *source, Format to_format) {
  if (to_format == source->get_format()) {
    return source;
  }

  try {
    ConversionFunctor<ID, NumNonZeros, Value> *conv_func =
        GetConversionFunction(source->get_format(), to_format);
    return (*conv_func)(source);
  } catch (...) {
    throw "Unsupported conversion error"; // TODO: Add decent exception
                                          // mechanism
  }
}

template <typename ID, typename NumNonZeros, typename Value>
ConversionFunctor<ID, NumNonZeros, Value> *
SparseConverter<ID, NumNonZeros, Value>::GetConversionFunction(Format from_format,
                                                             Format to_format) {
  try {
    return conversion_map_[from_format][to_format];
  } catch (...) {
    throw "Unsupported conversion error"; // TODO: Add decent exception
                                          // mechanism
  }
}

template <typename ID, typename NumNonZeros, typename Value>
bool SparseConverter<ID, NumNonZeros, Value>::CanConvert(Format from_format,
                                                      Format to_format) {
  if (conversion_map_.find(from_format) != conversion_map_.end()) {
    if (conversion_map_[from_format].find(to_format) !=
        conversion_map_[from_format].end()) {
      return true;
    }
  }
  return false;
}

template <typename ID, typename NumNonZeros, typename Value>
std::vector<SparseFormat<ID, NumNonZeros, Value> *>
SparseConverter<ID, NumNonZeros, Value>::ApplyConversionSchema(
    ConversionSchema cs,
    std::vector<SparseFormat<ID, NumNonZeros, Value> *> packed_sfs) {
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> ret;
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

template class SparseConverter<int, int, int>;
template class CooCsrFunctor<int, int, int>;
template class CsrCooFunctor<int, int, int>;

template class SparseConverter<unsigned int, unsigned int, void>;
template class CooCsrFunctor<unsigned int, unsigned int, void>;
template class CsrCooFunctor<unsigned int, unsigned int, void>;

template class SparseConverter<unsigned int, unsigned int, unsigned int>;
template class CooCsrFunctor<unsigned int, unsigned int, unsigned int>;
template class CsrCooFunctor<unsigned int, unsigned int, unsigned int>;
} // namespace sparsebase
