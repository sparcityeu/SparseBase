#include "sparsebase/sparse_converter.h"
#include "sparsebase/sparse_format.h"
#include <iostream>
#include <set>

using namespace std;

namespace sparsebase::utils {


template <typename IDType, typename NNZType, typename ValueType>
Format *CsrCooFunctor<IDType, NNZType, ValueType>::operator()(
        Format *source) {
  auto *csr = source->As<CSR<IDType,NNZType,ValueType>>();

  std::vector<DimensionType> dimensions = csr->get_dimensions();
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

  for (IDType i = 0; i < m; i++) {
    col[i] = csr_col[i];
  }

  // if (csr->vals != nullptr)
  ValueType * vals = nullptr;
  if constexpr (!std::is_same_v<void, ValueType>) {
    if (csr->get_vals() != nullptr){
      vals = new ValueType[nnz];
      auto csr_vals = csr->get_vals();
      for (NNZType i = 0; i < nnz; i++) {
        vals[i] = csr_vals[i];
      }
    } else{
      vals = nullptr;
    }
  }  
  auto *coo = new COO<IDType, NNZType, ValueType>(n, m, nnz, row, col, vals, kOwned);

  return coo;
}

// Ai -> row indices -> col
// Aj -> col indices -> is
// Ax -> nnz values -> vals

// Bp -> row -> row_ptr
// Bj -> col -> col
// Bx -> nnz -> vals
template <typename IDType, typename NNZType, typename ValueType>
    Format *CooCsrFunctor<IDType, NNZType, ValueType>::operator()(
            Format *source) {
  auto *coo = source->As<COO<IDType,NNZType,ValueType>>();

  std::vector<DimensionType> dimensions = coo->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = coo->get_num_nnz();
  auto coo_col = coo->get_col();
  auto coo_row = coo->get_row();
  NNZType *row_ptr = new NNZType[n + 1];
  IDType *col = new IDType[m];
  ValueType *vals;

  fill(row_ptr, row_ptr + n + 1, 0);
  fill(col, col + m, 0);

  // We need to ensure that they are sorted
  // Maybe add a sort check and then not do this if it is already sorted
  std::vector<std::pair<IDType, IDType>> edges;
  for (IDType i = 0; i < nnz; i++) {
    edges.emplace_back(coo_col[i], coo_row[i]);
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
    if (coo->get_vals() != nullptr){
      vals = new ValueType[nnz];
      auto coo_vals = coo->get_vals();
      fill(vals, vals + nnz, 0);
      for (NNZType i = 0; i < nnz; i++) {
        vals[i] = coo_vals[i];
      }
    } else {
      vals = nullptr;
    }
  } else {
    vals = nullptr;
  }

  auto csr = new CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col, vals, kOwned);
  return csr;
}

template <typename IDType, typename NNZType, typename ValueType>
Converter<IDType, NNZType, ValueType>::Converter() {
  this->RegisterConversionFunction(COO<IDType,NNZType,ValueType>::get_format_id_static(), CSR<IDType,NNZType,ValueType>::get_format_id_static(),
                                     new CooCsrFunctor<IDType, NNZType, ValueType>());
  this->RegisterConversionFunction(CSR<IDType,NNZType,ValueType>::get_format_id_static(), COO<IDType,NNZType,ValueType>::get_format_id_static(),
                                     new CsrCooFunctor<IDType, NNZType, ValueType>());
}

template <typename IDType, typename NNZType, typename ValueType>
Converter<IDType, NNZType, ValueType>::~Converter() {
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
void Converter<IDType, NNZType, ValueType>::RegisterConversionFunction(
        std::type_index from_format, std::type_index to_format,
    ConversionFunctor<IDType, NNZType, ValueType> *conv_func) {
  if (conversion_map_.count(from_format) == 0) {
    conversion_map_.emplace(
        from_format,
        std::unordered_map<std::type_index, ConversionFunctor<IDType, NNZType, ValueType>*>());
  }

  if (conversion_map_[from_format].count(to_format) == 0) {
    conversion_map_[from_format].emplace(to_format, conv_func);
  } else {
    conversion_map_[from_format][to_format] = conv_func;
  }
}

template <typename IDType, typename NNZType, typename ValueType>
Format *Converter<IDType, NNZType, ValueType>::Convert(
        Format *source, std::type_index to_format) {
  if (to_format == source->get_format_id()) {
    return source;
  }

  try {
    ConversionFunctor<IDType, NNZType, ValueType> *conv_func =
        GetConversionFunction(source->get_format_id(), to_format);
    return (*conv_func)(source);
  } catch (...) {
    throw ConversionException(source->get_format_id().name(), to_format.name()); // TODO: Add decent exception
                                          // mechanism
  }
}

/*template <typename IDType, typename NNZType, typename ValueType>
template <typename FormatType>
FormatType* Converter<IDType,NNZType,ValueType>::ConvertAs(Format *source) {
    auto* res = this->Convert(source, FormatType::get_format_id_static());
    return res->template As<FormatType>();
}*/

template <typename IDType, typename NNZType, typename ValueType>
ConversionFunctor<IDType, NNZType, ValueType> *
Converter<IDType, NNZType, ValueType>::GetConversionFunction(std::type_index from_format,
                                                             std::type_index to_format) {
  try {
    return conversion_map_[from_format][to_format];
  } catch (...) {
    throw ConversionException(from_format.name(), to_format.name()); // TODO: Add decent exception
                                          // mechanism
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool Converter<IDType, NNZType, ValueType>::CanConvert(std::type_index from_format,
                                                       std::type_index to_format) {
  if (conversion_map_.find(from_format) != conversion_map_.end()) {
    if (conversion_map_[from_format].find(to_format) !=
        conversion_map_[from_format].end()) {
      return true;
    }
  }
  return false;
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<Format *>
Converter<IDType, NNZType, ValueType>::ApplyConversionSchema(
    ConversionSchema cs,
    std::vector<Format *> packed_sfs) {
  std::vector<Format *> ret;
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
template class Converter<int, int, int>;
template class CooCsrFunctor<int, int, int>;
template class CsrCooFunctor<int, int, int>;

template class Converter<unsigned int, unsigned int, unsigned int>;
template class CooCsrFunctor<unsigned int, unsigned int, unsigned int>;
template class CsrCooFunctor<unsigned int, unsigned int, unsigned int>;
#endif

} // namespace sparsebase
