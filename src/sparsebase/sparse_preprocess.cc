#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_converter.h"
#include "sparsebase/sparse_format.h"
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
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType, class Preprocess, typename Key, typename KeyHash,
          typename KeyEqualTo>
bool FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, Preprocess, Key, KeyHash, KeyEqualTo>::
    RegisterFunctionNoOverride(const Key &key_of_function,
                                  const PreprocessFunction &func_ptr) {
  if (_map_to_function.find(key_of_function) == _map_to_function.end()) {
    return false; // function already exists for this Key
  } else {
    _map_to_function[key_of_function] = func_ptr;
    return true;
  }
}

template <typename IDType, typename NNZType, typename ValueType, typename ReturnType, class Preprocess, typename Key, typename KeyHash,
          typename KeyEqualTo>
void FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, Preprocess, Key, KeyHash, KeyEqualTo>::
    RegisterFunction(const Key &key_of_function, const PreprocessFunction &func_ptr) {
  _map_to_function[key_of_function] = func_ptr;
}
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType, class Preprocess, typename Key, typename KeyHash,
          typename KeyEqualTo>
bool FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, Preprocess, Key, KeyHash, KeyEqualTo>::
    UnregisterFunction(const Key &key_of_function) {
  if (_map_to_function.find(key_of_function) == _map_to_function.end()) {
    return false; // function already exists for this Key
  } else {
    _map_to_function.erase(key_of_function);
    return true;
  }
}
template <class Parent, typename IDType, typename NNZType, typename ValueType>
void ConverterMixin<Parent, IDType, NNZType, ValueType>::SetConverter(
    const utils::Converter<IDType, NNZType, ValueType> &new_sc) {
  sc_ = new_sc;
}
template <class Parent, typename IDType, typename NNZType, typename ValueType>
void ConverterMixin<Parent, IDType, NNZType, ValueType>::ResetConverter() {
  utils::Converter<IDType, NNZType, ValueType> new_sc;
  sc_ = new_sc;
}
template <typename IDType, typename NNZType, typename ValueType>
ReorderPreprocessType<IDType, NNZType, ValueType>::~ReorderPreprocessType(){};

template <typename IDType, typename NNZType, typename ValueType,
          typename ReturnType, class PreprocessingImpl, typename Key,
          typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<
    IDType, NNZType, ValueType, ReturnType, PreprocessingImpl, Key, KeyHash,
    KeyEqualTo>::CheckIfKeyMatches(ConversionMap map, Key key,
                                   std::vector<format::Format *> packed_sfs,
                                   std::vector<context::Context *> contexts) {
  bool match = true;
  if (map.find(key) != map.end()) {
    for (auto sf : packed_sfs) {
      bool found_context = false;
      for (auto context : contexts){
        if (sf->get_context()->IsEquivalent(context)){
          found_context = true;
        }
      }
      if (!found_context) match = false;
    }
  } else {
    match = false;
  }
  std::cout << "Match :" << match << std::endl;
  return match;
}
  //! Return the correct function for the operation and a conversion schema to convert the input formats
  /*!
   * \param key defines the types of input objects (default is vector of format types)
   * \param map the map between keys and functions
   * \param sc utils::Converter object to query possible conversions
   * \return the function to be executed and the conversion schema the conversions to carry out on inputs 
   */
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
std::tuple<ReturnType (*)(std::vector<Format *>, PreprocessParams *), utils::ConversionSchemaConditional>
FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, PreprocessingImpl,
                   Key, KeyHash, KeyEqualTo>::
    GetFunction(std::vector<format::Format*>packed_sfs, Key key, ConversionMap map, std::vector<context::Context*> contexts,
                utils::Converter<IDType, NNZType, ValueType> &sc) {
  utils::ConversionSchemaConditional cs;
  PreprocessFunction func = nullptr;
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
    std::vector<std::tuple<unsigned int, utils::ConversionSchemaConditional, Key>>
        usable_keys;
    for (auto potential_key : all_keys) {
      if (potential_key.size() == key.size()) {
        utils::ConversionSchemaConditional temp_cs;
        int conversions = 0;
        bool is_usable = true;
        for (int i = 0; i < potential_key.size(); i++) {
          if (key[i] == potential_key[i]) {
            temp_cs.push_back(std::make_tuple(false, potential_key[i], nullptr));
          } else {//  if (sc.CanConvert(key[i], potential_key[i])) {
            auto convertable = sc.CanConvert(key[i], packed_sfs[i]->get_context(), potential_key[i], contexts);    
            if (std::get<0>(convertable)){
              temp_cs.push_back(std::make_tuple(true, potential_key[i], std::get<1>(convertable)));
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
      throw 1; // TODO: add a custom exception type
    }
    std::tuple<PreprocessFunction, utils::ConversionSchemaConditional> best_conversion;
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
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F>
std::vector<std::type_index> 
FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo>::PackFormats(F sf) {
  return {sf->get_format_id()};
}
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::vector<std::type_index> 
FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo>::PackFormats(F sf, SF... sfs) {
  std::vector<std::type_index> f = {sf->get_format()};
  std::vector<std::type_index> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F>
std::vector<F>
FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo>::PackSFS(F sf) {
  return {sf};
}
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::vector<F>
FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo>::PackSFS(F sf, SF... sfs) {
  std::vector<F> f = {sf};
  std::vector<F> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
    class PreprocessingImpl,
    typename Key, typename KeyHash,
    typename KeyEqualTo>
template <typename F, typename... SF>
std::tuple<std::vector<format::Format*>, ReturnType>
FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, PreprocessingImpl,
    Key, KeyHash, KeyEqualTo>::
CachedExecute(PreprocessParams * params, utils::Converter<IDType, NNZType, ValueType>& sc, std::vector<context::Context*> contexts, F sf,
        SF... sfs) {
  ConversionMap map = this->_map_to_function;
  // pack the SFs into a vector
  std::vector<format::Format *> packed_sfs = PackSFS(sf, sfs...);
  // pack the SF formats into a vector
  std::vector<std::type_index> formats = PackFormats(sf, sfs...);
  // get conversion schema
  std::tuple<PreprocessFunction, utils::ConversionSchemaConditional> ret =
      GetFunction(packed_sfs, formats, map, contexts, sc);
  PreprocessFunction func = std::get<0>(ret);
  utils::ConversionSchemaConditional cs = std::get<1>(ret);
  // carry out conversion
  // ready_formats contains the format to use in preprocessing
  std::vector<Format *> ready_formats = sc.ApplyConversionSchema(cs, packed_sfs);
  // `converted` contains the results of conversions
  std::vector<Format *> converted;
  for (int i = 0; i < ready_formats.size(); i++){
    auto conversion = cs[i];
    if (std::get<0>(conversion)){
      converted.push_back(ready_formats[i]);
    } else {
      converted.push_back(nullptr);
    }
  }
  // carry out the correct call
  return std::make_tuple(converted, func(ready_formats , params));
}
template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
ReturnType
FunctionMatcherMixin<IDType, NNZType, ValueType, ReturnType, PreprocessingImpl,
                   Key, KeyHash, KeyEqualTo>::
    Execute(PreprocessParams * params, utils::Converter<IDType, NNZType, ValueType>& sc, std::vector<context::Context*> contexts, F sf,
            SF... sfs) {
  auto cached_output = CachedExecute(params, sc, contexts, sf, sfs...);
  auto converted_formats = std::get<0>(cached_output);
  auto return_object = std::get<1>(cached_output);
  for (auto* converted_format : converted_formats){
    if (converted_format != nullptr)
      delete converted_format;
  }
  return return_object;
}
template <typename IDType, typename NNZType, typename ValueType>
GenericReorder<IDType, NNZType, ValueType>::GenericReorder() {}
template <typename IDType, typename NNZType, typename ValueType>
DegreeReorder<IDType, NNZType, ValueType>::DegreeReorder(int hyperparameter) {
  // this->map[{kCSRFormat}]= calculate_order_csr;
  // this->RegisterFunction({kCSRFormat}, CalculateReorderCSR);
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()},
      CalculateReorderCSR);
  this->params_ = std::unique_ptr<DegreeReorderParams>(
      new DegreeReorderParams(hyperparameter));
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorder(Format * format, std::vector<context::Context*> contexts){
  return this->Execute(this->params_.get(), this->sc_, contexts, format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorder(Format * format, PreprocessParams* params, std::vector<context::Context*> contexts){
  return this->Execute(params, this->sc_, contexts, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format*>,IDType *> ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorderCached(Format * format, std::vector<context::Context*> contexts){
  return this->CachedExecute(this->params_.get(), this->sc_, contexts, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format*>,IDType *> ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorderCached(Format * format, PreprocessParams* params, std::vector<context::Context*> contexts){
  return this->CachedExecute(params, this->sc_, contexts, format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *DegreeReorder<IDType, NNZType, ValueType>::CalculateReorderCSR(
    std::vector<format::Format *> formats,
    PreprocessParams *params) {
  CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  DegreeReorderParams *cast_params = static_cast<DegreeReorderParams *>(params);
  std::cout << cast_params->hyperparameter;
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
  IDType *inv_sorted = new IDType[n];
  for (IDType i = 0; i < n; i++)
    inv_sorted[sorted[i]] = i;
  delete[] mr;
  delete[] counts;
  delete[] sorted;
  return inv_sorted;
}
template <typename IDType, typename NNZType, typename ValueType>
RCMReorder<IDType, NNZType, ValueType>::RCMReorder(float a, float b) {
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetReorderCSR);
  this->params_ = std::unique_ptr<RCMReorderParams>(new RCMReorderParams(a, b));
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
    std::vector<format::Format *> formats,
    PreprocessParams *params) {
  CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  RCMReorderParams *params_ = static_cast<RCMReorderParams *>(params);
  std::cout << "using the parameters " << params_->alpha << " and "
            << params_->beta << std::endl;
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

//template <typename IDType, typename NNZType, typename ValueType>
//IDType *ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorder(
//    SparseFormat<IDType, NNZType, ValueType> *csr) {
//  std::tuple<ReorderFunction<IDType, NNZType, ValueType>,
//             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
//      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
//  ReorderFunction<IDType, NNZType, ValueType> func = std::get<0>(func_formats);
//  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
//  return func(sfs, this->params_.get());
//}
//template <typename IDType, typename NNZType, typename ValueType>
//IDType *ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorder(
//    SparseFormat<IDType, NNZType, ValueType> *csr, ReorderParams *params) {
//  std::tuple<ReorderFunction<IDType, NNZType, ValueType>,
//             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
//      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
//  ReorderFunction<IDType, NNZType, ValueType> func = std::get<0>(func_formats);
//  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
//  return func(sfs, params);
//}

template <typename IDType, typename NNZType, typename ValueType>
Transform<IDType, NNZType, ValueType>::Transform(IDType* order){
  this->RegisterFunction({CSR<IDType, NNZType, ValueType>::get_format_id_static()}, TransformCSR);
  this->params_ = std::unique_ptr<TransformParams>(new TransformParams(order));
}
template <typename IDType, typename NNZType, typename ValueType>
TransformPreprocessType<IDType, NNZType,
                        ValueType>::~TransformPreprocessType(){};
template <typename IDType, typename NNZType, typename ValueType>
Format *Transform<IDType, NNZType, ValueType>::TransformCSR(
    std::vector<Format *> formats, PreprocessParams *params) {
  auto *sp = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto order = static_cast<TransformParams*>(params)->order;
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

  IDType *inverse_order = new IDType[n]();
  for (IDType i = 0; i < n; i++)
    inverse_order[order[i]] = i;
  NNZType c = 0;
  for (IDType i = 0; i < n; i++) {
    IDType u = inverse_order[i];
    nxadj[i + 1] = nxadj[i] + (xadj[u + 1] - xadj[u]);
    for (NNZType v = xadj[u]; v < xadj[u + 1]; v++) {
      nadj[c] = order[adj[v]];
      if constexpr (!std::is_same_v<void, ValueType>) {
        if (sp->get_vals() != nullptr)
          nvals[c] = vals[v];
      }
      c++;
    }
  }
  delete[] inverse_order;
  CSR<IDType, NNZType, ValueType> *csr = new CSR(n, m, nxadj, nadj, nvals);
  return csr;
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format*>, format::Format*>
TransformPreprocessType<IDType, NNZType, ValueType>::GetTransformationCached(
    Format *csr, std::vector<context::Context*>contexts) {
  //  std::tuple<TransformFunction<IDType, NNZType, ValueType, ReturnType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  //  TransformFunction<IDType, NNZType, ValueType, ReturnType> func = std::get<0>(func_formats);
  //  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
  //  return func(sfs, ordr);
  return this->CachedExecute(this->params_.get(), this->sc_, contexts, csr);
}
template <typename IDType, typename NNZType, typename ValueType>
Format*
TransformPreprocessType<IDType, NNZType, ValueType>::GetTransformation(
    Format *csr, std::vector<context::Context*> contexts) {
//  std::tuple<TransformFunction<IDType, NNZType, ValueType, ReturnType>,
//             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
//      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
//  TransformFunction<IDType, NNZType, ValueType, ReturnType> func = std::get<0>(func_formats);
//  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
//  return func(sfs, ordr);
    return this->Execute(this->params_.get(), this->sc_, contexts, csr);
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights(){
  #ifdef CUDA
    this->RegisterFunction({CUDACSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetJaccardWeightCUDACSR);
  #endif
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::~JaccardWeights(){};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
format::Format* JaccardWeights<IDType, NNZType, ValueType, FeatureType>::GetJaccardWeights(Format * format, std::vector<context::Context*> contexts){
    //std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType>,
    //            std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
    //    func_formats = 
    //DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func = std::get<0>(func_formats);
    //std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
    JaccardParams params;
    return this->Execute(&params, this->sc_, contexts, format); //func(sfs, this->params_.get());
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(){
    this->RegisterFunction({CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetDegreeDistributionCSR);
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::~DegreeDistribution(){};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::tuple<std::vector<format::Format*>, FeatureType*> DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistributionCached(Format * format, std::vector<context::Context*> contexts){
  //std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType>,
  //            std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //    func_formats =
  //DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func = std::get<0>(func_formats);
  //std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->CachedExecute(&params, this->sc_, contexts, format); //func(sfs, this->params_.get());
}
template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType * DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(Format * format, std::vector<context::Context*> contexts){
    //std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType>,
    //            std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
    //    func_formats = 
    //DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func = std::get<0>(func_formats);
    //std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
    DegreeDistributionParams params;
    return this->Execute(&params, this->sc_, contexts, format); //func(sfs, this->params_.get());
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType * DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(object::Graph<IDType, NNZType, ValueType> * obj, std::vector<context::Context*> contexts){
    //std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType>,
    //            std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
    //    func_formats = 
    //DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func = std::get<0>(func_formats);
    //std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
    DegreeDistributionParams params;
    Format * format = obj->get_connectivity();
    return this->Execute(&params, this->sc_, contexts, format); //func(sfs, this->params_.get());
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType * DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDegreeDistributionCSR(std::vector<Format *> formats, PreprocessParams * params){
    auto csr = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
    auto dims = csr->get_dimensions();
    for(auto dim : dims){
      std::cout << dim << std::endl;
    }
    IDType num_vertices = dims[0];
    NNZType num_edges = csr->get_num_nnz();
    FeatureType * dist = new FeatureType[num_vertices]();
    auto * rows = csr->get_row_ptr();
    for(int i = 0; i < num_vertices; i++){
      dist[i] = (rows[i+1] - rows[i]) / (FeatureType)num_edges;
      //std::cout<< dist[i] << std::endl;
    }
    return dist;
}




//template __global__ void jac_binning_gpu_u_per_grid_bst_kernel<int, int, float>(const int* xadj, const int* adj, int n, float* emetrics, int SM_FAC);
template class JaccardWeights<int, int, int, float>;
//template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>

#if !defined(_HEADER_ONLY)
#include "init/preprocess.inc"
#endif

} // namespace preprocess

} // namespace sparsebase
