#include "sparsebase/format/format.h"
#include "sparsebase/utils/utils.h"
#include "sparsebase/utils/converter/converter.h"
#include <iostream>
#include <set>

using namespace sparsebase::format;

namespace sparsebase::utils::converter {

std::unordered_map<
    std::type_index,
    std::unordered_map<std::type_index,
                       std::vector<std::tuple<
                           EdgeConditional, ConditionalConversionFunction>>>> *
Converter::get_conversion_map(bool is_move_conversion) {
  if (is_move_conversion)
    return &conditional_move_map_;
  else
    return &conditional_map_;
}



template <typename IDType, typename NNZType, typename ValueType>
Format *CooCscFunctionConditional(Format *source, context::Context *context) {
  auto *coo = source->As<COO<IDType, NNZType, ValueType>>();

  std::vector<DimensionType> dimensions = coo->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = coo->get_num_nnz();
  auto coo_col = coo->get_col();
  auto coo_row = coo->get_row();
  auto coo_vals = coo->get_vals();
  NNZType *col_ptr = new NNZType[n + 1];
  NNZType * col_counter = new NNZType[n]();
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
    col_ptr[coo_col[i]+1]++;
  }
  for (IDType i = 1; i <= n; i++) {
    col_ptr[i] += col_ptr[i - 1];
  }

  for (IDType i = 0; i < nnz; i++) {
    auto this_nnz_col = coo_col[i];
    auto this_nnz_row = coo_row[i];
    row[col_ptr[this_nnz_col]+col_counter[this_nnz_col]++] = this_nnz_row;
    if constexpr (!std::is_same_v<void, ValueType>) {
      if (coo_vals != nullptr) {
        vals[col_ptr[this_nnz_col]+col_counter[this_nnz_col]-1] = coo_vals[i];
      }
    }
  }
#pragma omp parallel for default(none) shared(col_ptr, row, vals, n)
      for (IDType i = 0; i < n; i++) {
        NNZType start = col_ptr[i];
        NNZType end = col_ptr[i + 1];

        if (end - start <= 1) {
          continue;
        }

        std::vector<std::pair<IDType, ValueType>> sort_vec;
        for (NNZType j = start; j < end; j++) {
          ValueType val = (vals != nullptr) ? vals[j] : 0;
          sort_vec.emplace_back(row[j], val);
        }
        std::sort(sort_vec.begin(), sort_vec.end(),
                  std::less<std::pair<IDType, ValueType>>());
        for (NNZType j = start; j < end; j++) {
          if (vals != nullptr) {
            vals[j] = sort_vec[j - start].second;
          }
          row[j] = sort_vec[j - start].first;
        }
      }

  auto csc =
      new CSC<IDType, NNZType, ValueType>(n, m, col_ptr, row, vals, kOwned, false);
  return csc;
}
template <typename IDType, typename NNZType, typename ValueType>
Format *CsrCooFunctionConditional(Format *source, context::Context *context) {
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
  auto *coo =
      new COO<IDType, NNZType, ValueType>(n, m, nnz, row, col, vals, kOwned);

  return coo;
}
template <typename IDType, typename NNZType, typename ValueType>
Format *CsrCscFunctionConditional(Format *source, context::Context *context) {
  auto *csr = source->As<CSR<IDType, NNZType, ValueType>>();

  auto coo = CsrCooFunctionConditional<IDType, NNZType, ValueType>(csr, context);


  return CooCscFunctionConditional<IDType, NNZType, ValueType>(coo, context);
}

template <typename IDType, typename NNZType, typename ValueType>
Format *CsrCooMoveConditionalFunction(Format *source, context::Context *) {
  auto *csr = source->As<CSR<IDType, NNZType, ValueType>>();
  auto col = csr->release_col();
  auto vals = csr->release_vals();
  std::vector<DimensionType> dimensions = csr->get_dimensions();
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
  auto *coo =
      new COO<IDType, NNZType, ValueType>(n, m, nnz, row, col, vals, kOwned);

  return coo;
}

template <typename IDType, typename NNZType, typename ValueType>
Format *CooCsrFunctionConditional(Format *source, context::Context *context) {
  auto *coo = source->As<COO<IDType, NNZType, ValueType>>();

  std::vector<DimensionType> dimensions = coo->get_dimensions();
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

  auto csr =
      new CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col, vals, kOwned);
  return csr;
}

template <typename IDType, typename NNZType, typename ValueType>
Format *CooCsrMoveConditionalFunction(Format *source, context::Context *) {
  auto *coo = source->As<COO<IDType, NNZType, ValueType>>();

  std::vector<DimensionType> dimensions = coo->get_dimensions();
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

  auto csr =
      new CSR<IDType, NNZType, ValueType>(n, m, row_ptr, col, vals, kOwned);
  return csr;
}

template <typename ValueType>
Converter *ConverterOrderOne<ValueType>::Clone() const {
  return new ConverterOrderOne<ValueType>(*this);
}

template <typename ValueType> void ConverterOrderOne<ValueType>::Reset() {
#ifdef USE_CUDA
  this->RegisterConditionalConversionFunction(
      Array<ValueType>::get_format_id_static(),
      format::cuda::CUDAArray<ValueType>::get_format_id_static(),
      converter::cuda::ArrayCUDAArrayConditionalFunction<ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
  this->RegisterConditionalConversionFunction(
      format::cuda::CUDAArray<ValueType>::get_format_id_static(),
      Array<ValueType>::get_format_id_static(),
      converter::cuda::CUDAArrayArrayConditionalFunction<ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
#endif
}

template <typename ValueType>
ConverterOrderOne<ValueType>::ConverterOrderOne() {
  this->Reset();
}

template <typename IDType, typename NNZType, typename ValueType>
Converter *ConverterOrderTwo<IDType, NNZType, ValueType>::Clone() const {
  return new ConverterOrderTwo<IDType, NNZType, ValueType>(*this);
}

template <typename IDType, typename NNZType, typename ValueType>
void ConverterOrderTwo<IDType, NNZType, ValueType>::Reset() {
  this->RegisterConditionalConversionFunction(
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      CooCsrFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
  this->RegisterConditionalConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      CsrCooFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
  this->RegisterConditionalConversionFunction(
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      CSC<IDType, NNZType, ValueType>::get_format_id_static(),
      CooCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
  this->RegisterConditionalConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      CSC<IDType, NNZType, ValueType>::get_format_id_static(),
      CsrCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
#ifdef USE_CUDA
  this->RegisterConditionalConversionFunction(
      format::cuda::CUDACSR<IDType, NNZType, ValueType>::get_format_id_static(),
      format::cuda::CUDACSR<IDType, NNZType, ValueType>::get_format_id_static(),
      converter::cuda::CUDACsrCUDACsrConditionalFunction<IDType, NNZType,
                                                         ValueType>,
      converter::cuda::CUDAPeerToPeer);
  this->RegisterConditionalConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      format::cuda::CUDACSR<IDType, NNZType, ValueType>::get_format_id_static(),
      converter::cuda::CsrCUDACsrConditionalFunction<IDType, NNZType,
                                                     ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
  this->RegisterConditionalConversionFunction(
      format::cuda::CUDACSR<IDType, NNZType, ValueType>::get_format_id_static(),
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      converter::cuda::CUDACsrCsrConditionalFunction<IDType, NNZType,
                                                     ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; });
#endif
  this->RegisterConditionalConversionFunction(
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      CooCsrMoveConditionalFunction<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; },
      true);
  this->RegisterConditionalConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      CsrCooMoveConditionalFunction<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; },
      true);
  this->RegisterConditionalConversionFunction(
      COO<IDType, NNZType, ValueType>::get_format_id_static(),
      CSC<IDType, NNZType, ValueType>::get_format_id_static(),
      CooCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; },
      true);
  this->RegisterConditionalConversionFunction(
      CSR<IDType, NNZType, ValueType>::get_format_id_static(),
      CSC<IDType, NNZType, ValueType>::get_format_id_static(),
      CsrCscFunctionConditional<IDType, NNZType, ValueType>,
      [](context::Context *, context::Context *) -> bool { return true; },
      true);
}

template <typename IDType, typename NNZType, typename ValueType>
ConverterOrderTwo<IDType, NNZType, ValueType>::ConverterOrderTwo() {
  this->Reset();
}

void Converter::RegisterConditionalConversionFunction(
    std::type_index from_type, std::type_index to_type,
    ConditionalConversionFunction conv_func, EdgeConditional edge_condition,
    bool is_move_conversion) {
  auto map = get_conversion_map(is_move_conversion);
  if (map->count(from_type) == 0) {
    map->emplace(
        from_type,
        std::unordered_map<
            std::type_index,
            std::vector<
                std::tuple<EdgeConditional, ConditionalConversionFunction>>>());
  }

  if ((*map)[from_type].count(to_type) == 0) {
    (*map)[from_type][to_type].push_back(
        std::make_tuple<EdgeConditional, ConditionalConversionFunction>(
            std::forward<EdgeConditional>(edge_condition),
            std::forward<ConditionalConversionFunction>(conv_func)));
    //(*map)[from_type].emplace(to_type, {make_tuple<EdgeConditional,
    // ConversionFunction>(std::forward<EdgeConditional>(edge_condition),
    // std::forward<ConversionFunction>(conv_func))});
  } else {
    (*map)[from_type][to_type].push_back(
        std::make_tuple<EdgeConditional, ConditionalConversionFunction>(
            std::forward<EdgeConditional>(edge_condition),
            std::forward<ConditionalConversionFunction>(conv_func)));
  }
}

Format *Converter::Convert(Format *source, std::type_index to_type,
                           context::Context *to_context,
                           bool is_move_conversion) {

  if (to_type == source->get_format_id() &&
      source->get_context()->IsEquivalent(to_context)) {
    return source;
  }

  try {
    ConditionalConversionFunction conv_func =
        GetConversionFunction(source->get_format_id(), source->get_context(),
                              to_type, to_context, is_move_conversion);
    return conv_func(source, to_context);
  } catch (...) {
    throw ConversionException(source->get_format_name(), utils::demangle(to_type));
    // mechanism
  }
}

/*template <typename IDType, typename NNZType, typename ValueType>
template <typename FormatType>
FormatType* Converter<IDType,NNZType,ValueType>::ConvertAs(Format *source) {
    auto* res = this->Convert(source, FormatType::get_format_id_static());
    return res->template As<FormatType>();
}*/
ConditionalConversionFunction Converter::GetConversionFunction(
    std::type_index from_type, context::Context *from_context,
    std::type_index to_type, context::Context *to_context,
    bool is_move_conversion) {
  try {
    auto map = get_conversion_map(is_move_conversion);
    for (auto conditional_function_tuple : (*map)[from_type][to_type]) {
      auto conditional = std::get<0>(conditional_function_tuple);
      if (conditional(from_context, to_context)) {
        return std::get<1>(conditional_function_tuple);
      }
    }
    throw ConversionException(utils::demangle(from_type), utils::demangle(to_type));
  } catch (...) {
    throw ConversionException(utils::demangle(from_type), utils::demangle(to_type));
    // mechanism
  }
}

std::tuple<bool, context::Context *>
Converter::CanConvert(std::type_index from_type, context::Context *from_context,
                      std::type_index to_type,
                      const std::vector<context::Context *> &to_contexts,
                      bool is_move_conversion) {
  auto map = get_conversion_map(is_move_conversion);
  if (map->find(from_type) != map->end()) {
    if ((*map)[from_type].find(to_type) != (*map)[from_type].end()) {
      for (auto condition_function_pair : (*map)[from_type][to_type]) {
        for (auto to_context : to_contexts) {
          if (std::get<0>(condition_function_pair)(from_context, to_context)) {
            return std::make_tuple<bool, context::Context *>(
                true, std::forward<context::Context *>(to_context));
          }
        }
      }
    }
  }
  return std::make_tuple<bool, context::Context *>(false, nullptr);
}
bool Converter::CanConvert(std::type_index from_type,
                           context::Context *from_context,
                           std::type_index to_type,
                           context::Context *to_context,
                           bool is_move_conversion) {
  auto map = get_conversion_map(is_move_conversion);
  if (map->find(from_type) != map->end()) {
    if ((*map)[from_type].find(to_type) != (*map)[from_type].end()) {
      for (auto condition_function_pair : (*map)[from_type][to_type]) {
        if (std::get<0>(condition_function_pair)(from_context, to_context)) {
          return true;
        }
      }
    }
  }
  return false;
}

  void Converter::ClearConversionFunctions(std::type_index from_type, std::type_index to_type, bool move_conversion){
    auto map = get_conversion_map(move_conversion);
    if (map->find(from_type) != map->end()){
      if ((*map)[from_type].find(to_type)!= (*map)[from_type].end()){
        (*map)[from_type].erase(to_type);
        if ((*map)[from_type].size() == 0) map->erase(from_type);
      }
    }
  }
  
  /*! Removes all conversion functions from the current converter
   */
  void Converter::ClearConversionFunctions(bool move_conversion){
    auto map = get_conversion_map(move_conversion);
    map->clear();
  }
  
  /*! Removes all move conversion functions from the current converter
   */
  void ClearMoveConversionFunctions(std::type_index from_type, std::type_index to_type);
std::vector<Format *>
Converter::ApplyConversionSchema(ConversionSchemaConditional cs,
                                 std::vector<Format *> packed_sfs,
                                 bool is_move_conversion) {
  std::vector<Format *> ret;
  for (int i = 0; i < cs.size(); i++) {
    auto conversion = cs[i];
    if (std::get<0>(conversion)) {
      ret.push_back(this->Convert(packed_sfs[i], std::get<1>(conversion),
                                  std::get<2>(conversion), is_move_conversion));
    } else {
      ret.push_back(packed_sfs[i]);
    }
  }
  return ret;
}
Converter::~Converter() {}

#if !defined(_HEADER_ONLY)
#include "init/converter.inc"
#endif

} // namespace sparsebase::utils::converter
