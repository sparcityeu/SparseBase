#include "sparsebase/sparse_converter.h"
#include "sparsebase/sparse_format.h"
#include <iostream>
#include <set>

using namespace std;

using namespace sparsebase::format;

namespace sparsebase::utils {

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<
    std::type_index,
    std::unordered_map<std::type_index,
                       ConversionFunctor<IDType, NNZType, ValueType> *>> *
Converter<IDType, NNZType, ValueType>::get_conversion_map(
    bool is_move_conversion) {
  if (is_move_conversion)
    return &move_conversion_map_;
  else
    return &conversion_map_;
}
template <typename IDType, typename NNZType, typename ValueType>
Format *CsrCooFunctor<IDType, NNZType, ValueType>::operator()(Format *source) {
  auto *csr = source->As<CSR<IDType, NNZType, ValueType>>();

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
  auto *coo =
      new COO<IDType, NNZType, ValueType>(n, m, nnz, row, col, vals, kOwned);

  return coo;
}

template <typename IDType, typename NNZType, typename ValueType>
Format *
CsrCooMoveFunctor<IDType, NNZType, ValueType>::operator()(Format *source) {
  auto *csr = source->As<CSR<IDType, NNZType, ValueType>>();
  auto col = csr->release_col();
  auto vals = csr->release_vals();
  std::vector<DimensionType> dimensions = csr->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = csr->get_num_nnz();

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

  // if (csr->vals != nullptr)
  auto *coo =
      new COO<IDType, NNZType, ValueType>(n, m, nnz, row, col, vals, kOwned);

  return coo;
}

// Ai -> row indices -> col
// Aj -> col indices -> is
// Ax -> nnz values -> vals

// Bp -> row -> row_ptr
// Bj -> col -> col
// Bx -> nnz -> vals
template <typename IDType, typename NNZType, typename ValueType>
Format *CooCsrFunctor<IDType, NNZType, ValueType>::operator()(Format *source) {
  auto *coo = source->As<COO<IDType, NNZType, ValueType>>();

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
  edges.reserve(nnz);
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
    if (coo->get_vals() != nullptr) {
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

  auto csr =
      new CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col, vals, kOwned);
  return csr;
}

template <typename IDType, typename NNZType, typename ValueType>
Format *
CooCsrMoveFunctor<IDType, NNZType, ValueType>::operator()(Format *source) {
  auto *coo = source->As<COO<IDType, NNZType, ValueType>>();

  std::vector<DimensionType> dimensions = coo->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = coo->get_num_nnz();
  auto coo_row = coo->get_row();
  NNZType *row_ptr = new NNZType[n + 1];
  IDType *col = coo->release_col();
  ValueType *vals = coo->release_vals();

  fill(row_ptr, row_ptr + n + 1, 0);

  // We need to ensure that they are sorted
  // Maybe add a sort check and then not do this if it is already sorted
  std::vector<std::pair<IDType, IDType>> edges;
  edges.reserve(nnz);
  for (IDType i = 0; i < nnz; i++) {
    edges.emplace_back(col[i], coo_row[i]);
  }
  sort(edges.begin(), edges.end(), less<std::pair<IDType, IDType>>());

  for (IDType i = 0; i < m; i++) {
    row_ptr[edges[i].first]++;
  }

  for (IDType i = 1; i <= n; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  for (IDType i = n; i > 0; i--) {
    row_ptr[i] = row_ptr[i - 1];
  }
  row_ptr[0] = 0;

  auto csr =
      new CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col, vals, kOwned);
  return csr;
}

template <typename IDType, typename NNZType, typename ValueType>
Converter<IDType, NNZType, ValueType>::Converter() {
  this->RegisterConversionFunction(
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      new CooCsrFunctor<IDType, NNZType, ValueType>());
  this->RegisterConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      new CsrCooFunctor<IDType, NNZType, ValueType>());
  this->RegisterConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      new CsrCooMoveFunctor<IDType, NNZType, ValueType>(), true);
  this->RegisterConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      new CooCsrMoveFunctor<IDType, NNZType, ValueType>(),

      true);
}

template <typename IDType, typename NNZType, typename ValueType>
Converter<IDType, NNZType, ValueType>::~Converter() {
  set<uintptr_t> deleted_ptrs;

  auto maps = {conversion_map_, move_conversion_map_};
  for (auto &w : maps) {
    for (auto x : w) {
      for (auto y : x.second) {
        ConversionFunctor<IDType, NNZType, ValueType> *ptr = y.second;
        if (deleted_ptrs.count((uintptr_t)ptr) == 0) {
          deleted_ptrs.insert((uintptr_t)ptr);
          delete ptr;
        }
      }
    }
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void Converter<IDType, NNZType, ValueType>::RegisterConversionFunction(
    std::type_index from_type, std::type_index to_type,
    ConversionFunctor<IDType, NNZType, ValueType> *conv_func,
    bool is_move_conversion) {
  auto map = get_conversion_map(is_move_conversion);
  if (map->count(from_type) == 0) {
    map->emplace(
        from_type,
        std::unordered_map<std::type_index,
                           ConversionFunctor<IDType, NNZType, ValueType> *>());
  }

  if ((*map)[from_type].count(to_type) == 0) {
    (*map)[from_type].emplace(to_type, conv_func);
  } else {
    (*map)[from_type][to_type] = conv_func;
  }
}

template <typename IDType, typename NNZType, typename ValueType>
Format *Converter<IDType, NNZType, ValueType>::Convert(
    Format *source, std::type_index to_type, bool is_move_conversion) {
  if (to_type == source->get_format_id()) {
    return source;
  }

  try {
    ConversionFunctor<IDType, NNZType, ValueType> *conv_func =
        GetConversionFunction(source->get_format_id(), to_type,
                              is_move_conversion);
    return (*conv_func)(source);
  } catch (...) {
    // TODO: add type here
    throw ConversionException(source->get_format_id().name(), to_type.name());
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
Converter<IDType, NNZType, ValueType>::GetConversionFunction(
    std::type_index from_type, std::type_index to_type,
    bool is_move_conversion) {
  try {
    auto map = get_conversion_map(is_move_conversion);
    return (*map)[from_type][to_type];
  } catch (...) {
    throw ConversionException(from_type.name(), to_type.name());
    // mechanism
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool Converter<IDType, NNZType, ValueType>::CanConvert(
    std::type_index from_type, std::type_index to_type,
    bool is_move_conversion) {
  auto map = get_conversion_map(is_move_conversion);
  if (map->find(from_type) != map->end()) {
    if ((*map)[from_type].find(to_type) != (*map)[from_type].end()) {
      return true;
    }
  }
  return false;
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<Format *>
Converter<IDType, NNZType, ValueType>::ApplyConversionSchema(
    ConversionSchema cs, std::vector<Format *> packed_sfs,
    bool is_move_conversion) {
  std::vector<Format *> ret;
  for (int i = 0; i < cs.size(); i++) {
    auto conversion = cs[i];
    if (std::get<0>(conversion)) {
      ret.push_back(this->Convert(packed_sfs[i], std::get<1>(conversion),
                                  is_move_conversion));
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

} // namespace sparsebase::utils
