#include "preprocess.h"
#include "sparsebase/format/format.h"
#include "sparsebase/utils/converter/converter.h"
#ifdef USE_CUDA
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#endif
#include <iostream>
#include <memory>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace sparsebase::format;

namespace sparsebase {

namespace preprocess {

std::size_t
TypeIndexVectorHash::operator()(const std::vector<std::type_index> &vf) const {
  size_t hash = 0;
  for (auto f : vf)
    hash += f.hash_code();
  return hash;
}
template <typename ReturnType, class Preprocess, typename Function,
          typename Key, typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<
    ReturnType, Preprocess, Function, Key, KeyHash,
    KeyEqualTo>::RegisterFunctionNoOverride(const Key &key_of_function,
                                            const Function &func_ptr) {
  if (map_to_function_.find(key_of_function) != map_to_function_.end()) {
    return false; // function already exists for this Key
  } else {
    map_to_function_[key_of_function] = func_ptr;
    return true;
  }
}

template <typename ReturnType, class Preprocess, typename Function,
          typename Key, typename KeyHash, typename KeyEqualTo>
void FunctionMatcherMixin<
    ReturnType, Preprocess, Function, Key, KeyHash,
    KeyEqualTo>::RegisterFunction(const Key &key_of_function,
                                  const Function &func_ptr) {
  map_to_function_[key_of_function] = func_ptr;
}
template <typename ReturnType, class Preprocess, typename Function,
          typename Key, typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<ReturnType, Preprocess, Function, Key, KeyHash,
                          KeyEqualTo>::UnregisterFunction(const Key &
                                                              key_of_function) {
  if (map_to_function_.find(key_of_function) == map_to_function_.end()) {
    return false; // function already exists for this Key
  } else {
    map_to_function_.erase(key_of_function);
    return true;
  }
}
template <class Parent>
void ConverterMixin<Parent>::SetConverter(
    const utils::converter::Converter &new_sc) {
  sc_ = std::unique_ptr<utils::converter::Converter>(new_sc.Clone());
}
template <class Parent> void ConverterMixin<Parent>::ResetConverter() {
  sc_->Reset();
}
template <class Parent>
std::unique_ptr<utils::converter::Converter>
ConverterMixin<Parent>::GetConverter() {
  if (sc_ == nullptr)
    return nullptr;
  return std::unique_ptr<utils::converter::Converter>(sc_->Clone());
}
template <typename IDType>
ReorderPreprocessType<IDType>::~ReorderPreprocessType() = default;
;

template <typename ReturnType, class PreprocessingImpl, typename Function,
          typename Key, typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<
    ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::CheckIfKeyMatches(ConversionMap map, Key key,
                                   std::vector<format::Format *> packed_sfs,
                                   std::vector<context::Context *> contexts) {
  bool match = true;
  if (map.find(key) != map.end()) {
    for (auto sf : packed_sfs) {
      bool found_context = false;
      for (auto context : contexts) {
        if (sf->get_context()->IsEquivalent(context)) {
          found_context = true;
        }
      }
      if (!found_context)
        match = false;
    }
  } else {
    match = false;
  }
  return match;
}
//! Return the correct function for the operation and a conversion schema to
//! convert the input formats
/*!
 * \param key defines the types of input objects (default is vector of format
 * types) \param map the map between keys and functions \param sc
 * utils::converter::Converter object to query possible conversions \return the
 * function to be executed and the conversion schema the conversions to carry
 * out on inputs
 */
template <typename ReturnType, class PreprocessingImpl, typename Function,
          typename Key, typename KeyHash, typename KeyEqualTo>
std::tuple<Function, utils::converter::ConversionSchemaConditional>
FunctionMatcherMixin<
    ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::GetFunction(std::vector<format::Format *> packed_sfs, Key key,
                             ConversionMap map,
                             std::vector<context::Context *> contexts,
                             utils::converter::Converter *sc) {
  utils::converter::ConversionSchemaConditional cs;
  Function func = nullptr;
  if (CheckIfKeyMatches(map, key, packed_sfs, contexts)) {
    for (auto f : key) {
      cs.push_back(std::make_tuple(false, f, nullptr));
    }
    func = map[key];
  } else {
    // the keys of all the possible functions
    std::vector<Key> all_keys;
    for (auto key_func : map) {
      all_keys.push_back(key_func.first);
    }
    std::vector<std::tuple<unsigned int,
                           utils::converter::ConversionSchemaConditional, Key>>
        usable_keys;
    for (auto potential_key : all_keys) {
      if (potential_key.size() == key.size()) {
        utils::converter::ConversionSchemaConditional temp_cs;
        int conversions = 0;
        bool is_usable = true;
        for (int i = 0; i < potential_key.size(); i++) {
          if (key[i] == potential_key[i]) {
            temp_cs.push_back(
                std::make_tuple(false, potential_key[i], nullptr));
          } else { //  if (sc.CanConvert(key[i], potential_key[i])) {
            if (sc == nullptr) {
              throw utils::NoConverterException();
            }
            auto convertable =
                sc->CanConvert(key[i], packed_sfs[i]->get_context(),
                               potential_key[i], contexts);
            if (std::get<0>(convertable)) {
              temp_cs.push_back(std::make_tuple(true, potential_key[i],
                                                std::get<1>(convertable)));
              conversions++;
            } else {
              is_usable = false;
            }
          }
        }
        if (is_usable) {
          usable_keys.push_back(
              std::make_tuple(conversions, temp_cs, potential_key));
        }
      }
    }
    if (usable_keys.size() == 0) {
      std::string message;
      message = "Could not find a function that matches the formats: {";
      for (auto f : packed_sfs) {
        message += f->get_format_name();
        message += " ";
      }
      message += "} using the contexts {";
      for (auto c : contexts) {
        message += c->get_context_type_member().name();
        message += " ";
      }
      message += "}";

      throw sparsebase::utils::FunctionNotFoundException(
          message); // TODO: add a custom exception type
    }
    std::tuple<Function, utils::converter::ConversionSchemaConditional>
        best_conversion;
    unsigned int num_conversions = (unsigned int)-1;
    for (auto potential_usable_key : usable_keys) {
      if (num_conversions > std::get<0>(potential_usable_key)) {
        num_conversions = std::get<0>(potential_usable_key);
        cs = std::get<1>(potential_usable_key);
        func = map[std::get<2>(potential_usable_key)];
      }
    }
  }
  return std::make_tuple(func, cs);
}
template <typename ReturnType, class PreprocessingImpl, typename Key,
          typename KeyHash, typename KeyEqualTo, typename Function>
template <typename Object>
std::vector<Object>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Key, KeyHash, KeyEqualTo,
                     Function>::PackObjects(Object object) {
  return {object};
}
template <typename ReturnType, class PreprocessingImpl, typename Key,
          typename KeyHash, typename KeyEqualTo, typename Function>
template <typename Object, typename... Objects>
std::vector<Object>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Key, KeyHash, KeyEqualTo,
                     Function>::PackObjects(Object object, Objects... objects) {
  std::vector<Object> v = {object};
  std::vector<Object> remainder = PackObjects(objects...);
  for (auto i : remainder) {
    v.push_back(i);
  }
  return v;
}
template <typename ReturnType, class PreprocessingImpl, typename Function,
          typename Key, typename KeyHash, typename KeyEqualTo>
template <typename F, typename... SF>
std::tuple<std::vector<format::Format *>, ReturnType>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function, Key, KeyHash,
                     KeyEqualTo>::CachedExecute(PreprocessParams *params,
                                                utils::converter::Converter *sc,
                                                std::vector<context::Context *>
                                                    contexts, bool convert_input,
                                                F format, SF... formats) {
  ConversionMap map = this->map_to_function_;
  // pack the Formats into a vector
  std::vector<format::Format *> packed_formats =
      PackObjects(format, formats...);
  // pack the types of Formats into a vector
  std::vector<std::type_index> packed_format_types;
  for (auto f : packed_formats)
    packed_format_types.push_back(f->get_format_id());
  // get conversion schema
  std::tuple<Function, utils::converter::ConversionSchemaConditional> ret =
      GetFunction(packed_formats, packed_format_types, map, contexts, sc);
  Function func = std::get<0>(ret);
  utils::converter::ConversionSchemaConditional cs = std::get<1>(ret);
  // carry out conversion
  // ready_formats contains the format to use in preprocessing
  if (!convert_input){
    for (auto convert_order : cs){
      if (std::get<0>(convert_order))
        throw utils::DirectExecutionNotAvailableException(packed_format_types, this->GetAvailableFormats());
    }
  }
  std::vector<Format *> ready_formats =
      sc->ApplyConversionSchema(cs, packed_formats);
  // `converted` contains the results of conversions
  std::vector<Format *> converted;
  for (int i = 0; i < ready_formats.size(); i++) {
    auto conversion = cs[i];
    if (std::get<0>(conversion)) {
      converted.push_back(ready_formats[i]);
    } else {
      converted.push_back(nullptr);
    }
  }
  // carry out the correct call
  return std::make_tuple(converted, func(ready_formats, params));
}

template <typename ReturnType, class PreprocessingImpl, typename Function,
          typename Key, typename KeyHash, typename KeyEqualTo>
template <typename F, typename... SF>
ReturnType
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function, Key, KeyHash,
                     KeyEqualTo>::Execute(PreprocessParams *params,
                                          utils::converter::Converter *sc,
                                          std::vector<context::Context *>
                                              contexts, bool convert_input,
                                          F sf, SF... sfs) {
  auto cached_output = CachedExecute(params, sc, contexts, convert_input, sf, sfs...);
  auto converted_formats = std::get<0>(cached_output);
  auto return_object = std::get<1>(cached_output);
  for (auto *converted_format : converted_formats) {
    if (converted_format != nullptr)
      delete converted_format;
  }
  return return_object;
}

template <typename IDType, typename NNZType, typename ValueType>
GenericReorder<IDType, NNZType, ValueType>::GenericReorder() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
}
template <typename IDType, typename NNZType, typename ValueType>
DegreeReorder<IDType, NNZType, ValueType>::DegreeReorder(DegreeReorderParams params) {
 DegreeReorder<IDType, NNZType, ValueType>(params.ascending);
}
template <typename IDType, typename NNZType, typename ValueType>
DegreeReorder<IDType, NNZType, ValueType>::DegreeReorder(bool ascending) {
  // this->map[{kCSRFormat}]= calculate_order_csr;
  // this->RegisterFunction({kCSRFormat}, CalculateReorderCSR);
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()},
      CalculateReorderCSR);
  this->params_ =
      std::unique_ptr<DegreeReorderParams>(new DegreeReorderParams(ascending));
}

template <typename ReturnType>
GenericPreprocessType<ReturnType>::~GenericPreprocessType() = default;
;

template <typename ReturnType>
int GenericPreprocessType<ReturnType>::GetOutput(
    Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->Execute(params, (this->sc_.get()), contexts, convert_input, format);
}

template <typename ReturnType>
std::tuple<std::vector<format::Format *>, int>
GenericPreprocessType<ReturnType>::GetOutputCached(
    Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->CachedExecute(params, (this->sc_.get()), contexts, convert_input, format);
}
template <typename IDType>
IDType *ReorderPreprocessType<IDType>::GetReorder(
    Format *format, std::vector<context::Context *> contexts, bool convert_input) {
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts, convert_input,
                       format);
}

template <typename IDType>
IDType *ReorderPreprocessType<IDType>::GetReorder(
    Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->Execute(params, (this->sc_.get()), contexts, convert_input, format);
}

template <typename IDType>
std::tuple<std::vector<format::Format *>, IDType *>
ReorderPreprocessType<IDType>::GetReorderCached(
    Format *format, std::vector<context::Context *> contexts, bool convert_input) {
  return this->CachedExecute(this->params_.get(), (this->sc_.get()), contexts, convert_input,
                             format);
}

template <typename IDType>
std::tuple<std::vector<format::Format *>, IDType *>
ReorderPreprocessType<IDType>::GetReorderCached(
    Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->CachedExecute(params, (this->sc_.get()), contexts, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *DegreeReorder<IDType, NNZType, ValueType>::CalculateReorderCSR(
    std::vector<format::Format *> formats, PreprocessParams *params) {
  CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  DegreeReorderParams *cast_params = static_cast<DegreeReorderParams *>(params);
  bool ascending = cast_params->ascending;
  IDType n = csr->get_dimensions()[0];
  IDType *counts = new IDType[n]();
  auto row_ptr = csr->get_row_ptr();
  auto col = csr->get_col();
  for (IDType u = 0; u < n; u++) {
    counts[row_ptr[u + 1] - row_ptr[u] + 1]++;
  }
  for (IDType u = 1; u < n; u++) {
    counts[u] += counts[u - 1];
  }
  IDType *sorted = new IDType[n];
  memset(sorted, -1, sizeof(IDType) * n);
  IDType *mr = new IDType[n]();
  for (IDType u = 0; u < n; u++) {
    IDType ec = counts[row_ptr[u + 1] - row_ptr[u]];
    sorted[ec + mr[ec]] = u;
    mr[ec]++;
  }
  if (!ascending) {
    for (IDType i = 0; i < n / 2; i++) {
      IDType swp = sorted[i];
      sorted[i] = sorted[n - i - 1];
      sorted[n - i - 1] = swp;
    }
  }
  auto *inverse_permutation = new IDType[n];
  for (IDType i = 0; i < n; i++) {
    inverse_permutation[sorted[i]] = i;
  }
  delete[] mr;
  delete[] counts;
  delete[] sorted;
  return inverse_permutation;
}
template <typename IDType, typename NNZType, typename ValueType>
RCMReorder<IDType, NNZType, ValueType>::RCMReorder() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetReorderCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
RCMReorder<IDType, NNZType, ValueType>::RCMReorder(RCMReorderParams p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetReorderCSR);

}
template <typename IDType, typename NNZType, typename ValueType>
IDType RCMReorder<IDType, NNZType, ValueType>::peripheral(NNZType *xadj,
                                                          IDType *adj, IDType n,
                                                          IDType start,
                                                          SignedID *distance,
                                                          IDType *Q) {
  IDType r = start;
  SignedID rlevel = -1;
  SignedID qlevel = 0;

  while (rlevel != qlevel) {
    // cout << "Finding peripheral: current dist = " << qlevel << std::endl;;
    rlevel = qlevel;

    for (IDType i = 0; i < n; i++)
      distance[i] = -1;
    IDType qrp = 0, qwp = 0;
    distance[r] = 0;
    Q[qwp++] = r;

    while (qrp < qwp) {
      IDType u = Q[qrp++];
      for (NNZType ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
        IDType v = adj[ptr];
        if (distance[v] == (IDType)-1) {
          distance[v] = distance[u] + 1;
          Q[qwp++] = v;
        }
      }
    }

    qlevel = 0;
    for (IDType i = 0; i < qrp; i++) {
      if (qlevel < distance[Q[i]]) {
        qlevel = distance[Q[i]];
        r = Q[i];
      }
    }
  }
  return r;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *RCMReorder<IDType, NNZType, ValueType>::GetReorderCSR(
    std::vector<format::Format *> formats, PreprocessParams *params) {
  CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  NNZType *xadj = csr->get_row_ptr();
  IDType *adj = csr->get_col();
  IDType n = csr->get_dimensions()[0];
  IDType *Q = new IDType[n];

  IDType *Qp = new IDType[n];
  SignedID *distance = new SignedID[n];
  IDType *V = new IDType[n];
  for (IDType i = 0; i < n; i++)
    V[i] = 0;
  std::priority_queue<std::pair<IDType, IDType>> PQ;
  int qrp = 0, qwp = 0;
  IDType reverse = n - 1;

  for (IDType i = 0; i < n; i++) {
    if (V[i] == 0) {
      if (xadj[i] == xadj[i + 1]) {
        Q[reverse--] = i;
        V[i] = 1;
        continue;
      }

      // cout << i << std::endl;
      IDType perv = peripheral(xadj, adj, n, i, distance, Qp);
      V[perv] = 1;
      Q[qwp++] = perv;

      while (qrp < qwp) {
        IDType u = Q[qrp++];
        for (IDType ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
          IDType v = adj[ptr];
          if (V[v] == 0) {
            PQ.push(std::make_pair(xadj[v + 1] - xadj[v], v));
            V[v] = 1;
          }
        }

        while (!PQ.empty()) {
          Q[qwp++] = PQ.top().second;
          ;
          PQ.pop();
        }
      }
    }
  }

  // Reverse
  for (IDType i = 0; i < n / 2; i++) {
    Qp[i] = Q[n - i - 1];
    Qp[n - i - 1] = Q[i];
  }
  // Place it in the form that the transform function takes
  for (IDType i = 0; i < n; i++) {
    Q[Qp[i]] = i;
  }

  delete[] Qp;
  delete[] distance;
  delete[] V;
  return Q;
}

template <typename IDType, typename ValueType>
PermuteOrderOne<IDType, ValueType>::PermuteOrderOne(PermuteOrderOneParams params) {
  PermuteOrderOne(params.order);
}
template <typename IDType, typename ValueType>
PermuteOrderOne<IDType, ValueType>::PermuteOrderOne(IDType *order) {
  this->SetConverter(
      utils::converter::ConverterOrderOne<ValueType>{});
  this->RegisterFunction(
      {Array<ValueType>::get_format_id_static()}, PermuteArray);
  this->params_ = std::unique_ptr<PermuteOrderOneParams>(new PermuteOrderOneParams(order));
}
template <typename IDType, typename ValueType>
format::FormatOrderOne<ValueType> *PermuteOrderOne<IDType, ValueType>::PermuteArray(
    std::vector<Format *> formats, PreprocessParams *params) {
  auto *sp = formats[0]->As<Array<ValueType>>();
  auto order = static_cast<PermuteOrderOneParams *>(params)->order;
  std::vector<DimensionType> dimensions = sp->get_dimensions();
  IDType length = dimensions[0];
  ValueType *vals = sp->get_vals();
  ValueType *nvals = new ValueType[length]();
  IDType* inv_order = new IDType[length];
  for (IDType i = 0; i < length; i++){
    inv_order[order[i]] = i;
  }

  for (IDType i = 0; i < length; i++){
    nvals[i] = vals[inv_order[i]];
  }
  Array<ValueType> *arr = new Array<ValueType>(length, nvals, kOwned);
  return arr;
}
template <typename IDType, typename NNZType, typename ValueType>
PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwo(IDType *row_order, IDType* col_order) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, PermuteOrderTwoCSR);
  this->params_ = std::unique_ptr<PermuteOrderTwoParams>(new PermuteOrderTwoParams(row_order, col_order));
}
template <typename IDType, typename NNZType, typename ValueType>
PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwo(PermuteOrderTwoParams params) {
  PermuteOrderTwo(params.row_order, params.col_order);
}
template <typename InputFormatType, typename ReturnFormtType>
TransformPreprocessType<InputFormatType, ReturnFormtType>::~TransformPreprocessType() = default;
template <typename IDType, typename NNZType, typename ValueType>
format::FormatOrderTwo<IDType, NNZType, ValueType> *PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwoCSR(
    std::vector<Format *> formats, PreprocessParams *params) {
  auto *sp = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto row_order = static_cast<PermuteOrderTwoParams *>(params)->row_order;
  auto col_order = static_cast<PermuteOrderTwoParams *>(params)->col_order;
  std::vector<DimensionType> dimensions = sp->get_dimensions();
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
    if (sp->get_vals() != nullptr)
      nvals = new ValueType[nnz]();
  }
  std::function<IDType(IDType)> get_i_row_order;
  std::function<IDType(IDType)> get_col_order;
  IDType *inverse_row_order;
  if (row_order!=nullptr){
    inverse_row_order= new IDType[n]();
    for (IDType i = 0; i < n; i++)
      inverse_row_order[row_order[i]] = i;
    get_i_row_order= [&inverse_row_order](IDType i) -> IDType {
      return inverse_row_order[i];
    };
  } else {
    get_i_row_order= [&inverse_row_order](IDType i) -> IDType {
      return i;
    };
  }
  if (col_order!=nullptr){
    get_col_order= [&col_order](IDType i) -> IDType {
      return col_order[i];
    };
  } else {
    get_col_order= [](IDType i) -> IDType {
      return i;
    };
  }
  //IDType *inverse_col_order = new IDType[n]();
  //for (IDType i = 0; i < n; i++)
  //  inverse_col_order[col_order[i]] = i;
  NNZType c = 0;
  for (IDType i = 0; i < n; i++) {
    IDType u = get_i_row_order(i);
    nxadj[i + 1] = nxadj[i] + (xadj[u + 1] - xadj[u]);
    for (NNZType v = xadj[u]; v < xadj[u + 1]; v++) {
      nadj[c] = get_col_order(adj[v]);
      if constexpr (!std::is_same_v<void, ValueType>) {
        if (sp->get_vals() != nullptr)
          nvals[c] = vals[v];
      }
      c++;
    }
  }
  if (row_order == nullptr)
    delete[] inverse_row_order;
  CSR<IDType, NNZType, ValueType> *csr = new CSR(n, m, nxadj, nadj, nvals);
  return csr;
}

template <typename InputFormatType, typename ReturnFormatType>
std::tuple<std::vector<format::Format *>, ReturnFormatType *>
TransformPreprocessType<InputFormatType, ReturnFormatType>::GetTransformationCached(
    format::Format *format, std::vector<context::Context *> contexts, bool convert_input) {
  if (dynamic_cast<InputFormatType*>(format) == nullptr) throw utils::TypeException(format->get_format_name(), InputFormatType::get_format_name_static());
  return this->CachedExecute(this->params_.get(), (this->sc_.get()), contexts, convert_input,
                             format);
}

template <typename InputFormatType, typename ReturnFormatType>
std::tuple<std::vector<format::Format *>, ReturnFormatType *>
TransformPreprocessType<InputFormatType, ReturnFormatType>::GetTransformationCached(
    format::Format  *format, PreprocessParams *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  if (dynamic_cast<InputFormatType*>(format) == nullptr) throw utils::TypeException(format->get_format_name(), InputFormatType::get_format_name_static());
  return this->CachedExecute(params, (this->sc_.get()), contexts, convert_input, format);
}

template <typename InputFormatType, typename ReturnFormatType>
ReturnFormatType *TransformPreprocessType<InputFormatType, ReturnFormatType>::GetTransformation(
    format::Format *format, std::vector<context::Context *> contexts, bool convert_input) {
  if (dynamic_cast<InputFormatType*>(format) == nullptr) throw utils::TypeException(format->get_format_name(), InputFormatType::get_format_name_static());
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts, convert_input, format);
}

template <typename InputFormatType, typename ReturnFormatType>
ReturnFormatType *TransformPreprocessType<InputFormatType, ReturnFormatType>::GetTransformation(
    format::Format  *format, PreprocessParams *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  if (dynamic_cast<InputFormatType*>(format) == nullptr) throw utils::TypeException(format->get_format_name(), InputFormatType::get_format_name_static());
  return this->Execute(params, (this->sc_.get()), contexts, convert_input, format);
}

template <typename FeatureType>
FeaturePreprocessType<FeatureType>::~FeaturePreprocessType() = default;

template <typename FeatureType>
std::shared_ptr<PreprocessParams>
FeaturePreprocessType<FeatureType>::get_params() {
  return this->params_;
}
template <typename FeatureType>
std::shared_ptr<PreprocessParams>
FeaturePreprocessType<FeatureType>::get_params(std::type_index t) {
  if (this->pmap_.find(t) != this->pmap_.end()) {
    return this->pmap_[t];
  } else {
    throw utils::FeatureParamsException(get_feature_id().name(), t.name());
  }
}
template <typename FeatureType>
void FeaturePreprocessType<FeatureType>::set_params(
    std::type_index t, std::shared_ptr<PreprocessParams> p) {
  auto ids = this->get_sub_ids();
  if (std::find(ids.begin(), ids.end(), t) != ids.end()) {
    this->pmap_[t] = p;
  } else {
    throw utils::FeatureParamsException(get_feature_id().name(), t.name());
  }
}
template <typename FeatureType>
std::type_index FeaturePreprocessType<FeatureType>::get_feature_id() {
  return typeid(*this);
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights(ParamsType) {}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
#ifdef USE_CUDA
  std::vector<std::type_index> formats = {
      format::cuda::CUDACSR<IDType, NNZType,
                            ValueType>::get_format_id_static()};
  this->RegisterFunction(formats, GetJaccardWeightCUDACSR);
#endif
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::~JaccardWeights(){};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::Format *
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::GetJaccardWeights(
    Format *format, std::vector<context::Context *> contexts, bool convert_input) {
  return this->Execute(nullptr, (this->sc_.get()), contexts, convert_input,
                       format); // func(sfs, this->params_.get());
}
#ifdef USE_CUDA
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::Format *
preprocess::JaccardWeights<IDType, NNZType, ValueType, FeatureType>::
    GetJaccardWeightCUDACSR(std::vector<Format *> formats,
                            PreprocessParams *params) {
  auto cuda_csr =
      formats[0]->As<format::cuda::CUDACSR<IDType, NNZType, ValueType>>();
  return preprocess::cuda::RunJaccardKernel<IDType, NNZType, ValueType,
                                            FeatureType>(cuda_csr);
}
#endif

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType,
                   FeatureType>::DegreeDistribution() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<DegreeDistributionParams>(new DegreeDistributionParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType,
    FeatureType>::DegreeDistribution(DegreeDistributionParams params) {
  DegreeDistribution();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(
    const DegreeDistribution &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(
    const std::shared_ptr<DegreeDistributionParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()},
      GetDegreeDistributionCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c, bool convert_input) {
  return {{this->get_feature_id(),
           std::forward<FeatureType *>(GetDistribution(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new DegreeDistribution<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index DegreeDistribution<IDType, NNZType, ValueType,
                                   FeatureType>::get_feature_id_static() {
  return typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType,
                   FeatureType>::~DegreeDistribution() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType *>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
    GetDistributionCached(Format *format,
                          std::vector<context::Context *> contexts, bool convert_input) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts, convert_input,
                             format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(
    Format *format, std::vector<context::Context *> contexts, bool convert_input) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->Execute(&params, (this->sc_.get()), contexts, convert_input,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts, convert_input,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
    GetDegreeDistributionCSR(std::vector<Format *> formats,
                             PreprocessParams *params) {
  auto csr = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto dims = csr->get_dimensions();
  IDType num_vertices = dims[0];
  NNZType num_edges = csr->get_num_nnz();
  FeatureType *dist = new FeatureType[num_vertices]();
  auto *rows = csr->get_row_ptr();
  for (int i = 0; i < num_vertices; i++) {
    dist[i] = (rows[i + 1] - rows[i]) / (FeatureType)num_edges;
    // std::cout<< dist[i] << std::endl;
  }
  return dist;
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(DegreesParams) {
  Degrees();
}
template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = std::shared_ptr<DegreesParams>(new DegreesParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(
    const Degrees<IDType, NNZType, ValueType> &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(
    const std::shared_ptr<DegreesParams> r) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = r;
  this->pmap_[get_feature_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::~Degrees() = default;

template <typename IDType, typename NNZType, typename ValueType>
void Degrees<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetDegreesCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
Degrees<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(Degrees<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<ExtractableType *> Degrees<IDType, NNZType, ValueType>::get_subs() {
  return {new Degrees<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index Degrees<IDType, NNZType, ValueType>::get_feature_id_static() {
  return typeid(Degrees<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
Degrees<IDType, NNZType, ValueType>::Extract(
    format::Format *format, std::vector<context::Context *> c, bool convert_input) {
  return {
      {this->get_feature_id(), std::forward<IDType *>(GetDegrees(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
IDType *Degrees<IDType, NNZType, ValueType>::GetDegrees(
    Format *format, std::vector<context::Context *> c, bool convert_input) {
  return this->Execute(this->params_.get(), (this->sc_.get()), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *Degrees<IDType, NNZType, ValueType>::GetDegreesCSR(
    std::vector<Format *> formats, PreprocessParams *params) {
  auto csr = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto dims = csr->get_dimensions();
  IDType num_vertices = dims[0];
  NNZType num_edges = csr->get_num_nnz();
  IDType *degrees = new IDType[num_vertices]();
  auto *rows = csr->get_row_ptr();
  for (int i = 0; i < num_vertices; i++) {
    degrees[i] = rows[i + 1] - rows[i];
  }
  return degrees;
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
    FeatureType>::Degrees_DegreeDistribution(Params) {
  Degrees_DegreeDistribution();
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
                           FeatureType>::Degrees_DegreeDistribution() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  this->Register();
  // this->RegisterFunction(
  //     {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetCSR);
  this->params_ = std::shared_ptr<Params>(new Params());
  this->pmap_.insert({get_feature_id_static(), this->params_});
  std::shared_ptr<PreprocessParams> deg_dist_param(
      new typename DegreeDistribution<IDType, NNZType, ValueType,
                                      FeatureType>::DegreeDistributionParams);
  std::shared_ptr<PreprocessParams> degs_param(
      new typename Degrees<IDType, NNZType, ValueType>::DegreesParams);
  this->pmap_[DegreeDistribution<IDType, NNZType, ValueType,
                                 FeatureType>::get_feature_id_static()] =
      deg_dist_param;
  this->pmap_[Degrees<IDType, NNZType, ValueType>::get_feature_id_static()] =
      degs_param;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void Degrees_DegreeDistribution<IDType, NNZType, ValueType,
                                FeatureType>::Register() {
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
    Degrees_DegreeDistribution(const Degrees_DegreeDistribution<
                               IDType, NNZType, ValueType, FeatureType> &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
    Degrees_DegreeDistribution(const std::shared_ptr<Params> r) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = r;
  this->pmap_[get_feature_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
                           FeatureType>::~Degrees_DegreeDistribution() =
    default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
                           FeatureType>::get_sub_ids() {
  std::vector<std::type_index> r = {
      typeid(Degrees<IDType, NNZType, ValueType>),
      typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>)};
  std::sort(r.begin(), r.end());
  return r;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
                           FeatureType>::get_subs() {
  auto *f1 = new Degrees<IDType, NNZType, ValueType>();
  if (this->pmap_.find(
          Degrees<IDType, NNZType, ValueType>::get_feature_id_static()) !=
      this->pmap_.end()) {
    f1->set_params(Degrees<IDType, NNZType, ValueType>::get_feature_id_static(),
                   this->pmap_[Degrees<IDType, NNZType,
                                       ValueType>::get_feature_id_static()]);
  }

  auto *f2 = new DegreeDistribution<IDType, NNZType, ValueType, FeatureType>();
  if (this->pmap_.find(
          DegreeDistribution<IDType, NNZType, ValueType,
                             FeatureType>::get_feature_id_static()) !=
      this->pmap_.end()) {
    f2->set_params(
        DegreeDistribution<IDType, NNZType, ValueType,
                           FeatureType>::get_feature_id_static(),
        this->pmap_[DegreeDistribution<IDType, NNZType, ValueType,
                                       FeatureType>::get_feature_id_static()]);
  }

  auto ids = this->get_sub_ids();
  if (ids[0] == f1->get_feature_id())
    return {f1, f2};
  else
    return {f2, f1};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
                           FeatureType>::get_feature_id_static() {
  return typeid(
      Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c, bool convert_input) {
  return Get(format, c, convert_input);
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Get(
    Format *format, std::vector<context::Context *> c, bool convert_input) {
  Params params;
  return this->Execute(this->params_.get(), (this->sc_.get()), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetCSR(
    std::vector<Format *> formats, PreprocessParams *params) {
  auto csr = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto dims = csr->get_dimensions();
  IDType num_vertices = dims[0];
  NNZType num_edges = csr->get_num_nnz();
  auto *degrees = new IDType[num_vertices]();
  auto *dist = new FeatureType[num_vertices];
  auto *rows = csr->get_row_ptr();
  for (int i = 0; i < num_vertices; i++) {
    degrees[i] = rows[i + 1] - rows[i];
    dist[i] = (rows[i + 1] - rows[i]) / (FeatureType)num_edges;
  }
  return {{Degrees<IDType, NNZType, ValueType>::get_feature_id_static(),
           std::forward<IDType *>(degrees)},
          {DegreeDistribution<IDType, NNZType, ValueType,
                              FeatureType>::get_feature_id_static(),
           std::forward<FeatureType *>(dist)}};
}

template <typename IDType>
PartitionPreprocessType<IDType>::PartitionPreprocessType() = default;

template <typename IDType>
IDType *PartitionPreprocessType<IDType>::Partition(format::Format* format,
                                                   std::vector<context::Context*> contexts, bool convert_input) {
    return this->Execute(this->params_.get(), (this->sc_.get()), contexts, convert_input, format);
}

template <typename IDType>
IDType *PartitionPreprocessType<IDType>::Partition(
    format::Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->Execute(params, (this->sc_.get()), contexts, convert_input, format);
}

template <typename IDType>
PartitionPreprocessType<IDType>::~PartitionPreprocessType() = default;

#ifdef USE_METIS

#include "sparsebase/external/metis/metis.h"

template <typename IDType, typename NNZType, typename ValueType>
MetisPartition<IDType, NNZType, ValueType>::MetisPartition(){
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>());

  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, PartitionCSR);

  this->params_ = std::unique_ptr<MetisParams>(new MetisParams);
}


template <typename IDType, typename NNZType, typename ValueType>
IDType* MetisPartition<IDType, NNZType, ValueType>::PartitionCSR(std::vector<format::Format*> formats, PreprocessParams* params){
  CSR<IDType, NNZType, ValueType>* csr = formats[0]->As<CSR<IDType, NNZType, ValueType>>();

  MetisParams* mparams = static_cast<MetisParams*>(params);

  idx_t n = (idx_t) csr->get_dimensions()[0];

  IDType* partition = new IDType[n];

  idx_t options[METIS_NOPTIONS];
  options[METIS_OPTION_OBJTYPE] = (idx_t) mparams->objtype;
  options[METIS_OPTION_CTYPE] = (idx_t) mparams->ctype;
  options[METIS_OPTION_IPTYPE] = (idx_t) mparams->iptype;
  options[METIS_OPTION_RTYPE] = (idx_t) mparams->rtype;
  options[METIS_OPTION_NO2HOP] = (idx_t) mparams->no2hop;
  options[METIS_OPTION_NCUTS] = (idx_t) mparams->ncuts;
  options[METIS_OPTION_NITER] = (idx_t) mparams->niter;
  options[METIS_OPTION_UFACTOR] = (idx_t) mparams->ufactor;
  options[METIS_OPTION_MINCONN] = (idx_t) mparams->minconn;
  options[METIS_OPTION_CONTIG] = (idx_t) mparams->contig;
  options[METIS_OPTION_SEED] = (idx_t) mparams->seed;
  options[METIS_OPTION_NUMBERING] = (idx_t) mparams->numbering;
  options[METIS_OPTION_DBGLVL] = (idx_t) 0;

  idx_t np = (idx_t) mparams->num_partitions;
  idx_t nw = (idx_t) 1;
  idx_t objval;

  if constexpr(std::is_signed_v<IDType> && std::is_signed_v<NNZType>
                && sizeof(IDType) == sizeof(idx_t) && sizeof(NNZType) == sizeof(idx_t)){

    if(mparams->ptype == METIS_PTYPE_RB){
     METIS_PartGraphRecursive(&n, &nw, (idx_t*) csr->get_row_ptr(), (idx_t*) csr->get_col(),
                        nullptr, nullptr, nullptr, &np, nullptr, nullptr, options,
                        &objval, partition);

    } else {

      METIS_PartGraphKway(&n, &nw, (idx_t *)csr->get_row_ptr(),
                          (idx_t *)csr->get_col(), nullptr, nullptr, nullptr,
                          &np, nullptr, nullptr, options,
                          &objval, partition);
    }
  } else {
    throw utils::TypeException("Metis Partitioner supports only " + std::to_string(sizeof(idx_t)*8) + "-bit signed integers for ids");
  }
  return partition;
}

#endif


#if !defined(_HEADER_ONLY)
#include "init/preprocess.inc"
#endif

} // namespace preprocess

} // namespace sparsebase
