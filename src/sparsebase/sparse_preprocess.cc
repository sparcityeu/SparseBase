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
template <typename ReturnType, class Preprocess, typename Function, typename Key, typename KeyHash,
          typename KeyEqualTo>
bool FunctionMatcherMixin<ReturnType, Preprocess, Function, Key, KeyHash, KeyEqualTo>::
    RegisterFunctionNoOverride(const Key &key_of_function,
                                  const Function &func_ptr) {
  if (_map_to_function.find(key_of_function) == _map_to_function.end()) {
    return false; // function already exists for this Key
  } else {
    _map_to_function[key_of_function] = func_ptr;
    return true;
  }
}

template <typename ReturnType, class Preprocess, typename Function, typename Key, typename KeyHash,
          typename KeyEqualTo>
void FunctionMatcherMixin<ReturnType, Preprocess, Function, Key, KeyHash, KeyEqualTo>::
    RegisterFunction(const Key &key_of_function, const Function &func_ptr) {
  _map_to_function[key_of_function] = func_ptr;
}
template <typename ReturnType, class Preprocess, typename Function, typename Key, typename KeyHash,
          typename KeyEqualTo>
bool FunctionMatcherMixin<ReturnType, Preprocess, Function, Key, KeyHash, KeyEqualTo>::
    UnregisterFunction(const Key &key_of_function) {
  if (_map_to_function.find(key_of_function) == _map_to_function.end()) {
    return false; // function already exists for this Key
  } else {
    _map_to_function.erase(key_of_function);
    return true;
  }
}
template <class Parent>
void ConverterMixin<Parent>::SetConverter(
    const utils::Converter &new_sc) {
  sc_ = std::unique_ptr<utils::Converter>(new_sc.Clone());
}
template <class Parent>
void ConverterMixin<Parent>::ResetConverter() {
  sc_->Reset();
}
template <typename IDType, typename NNZType, typename ValueType>
ReorderPreprocessType<IDType, NNZType, ValueType>::~ReorderPreprocessType()= default;;

template <typename ReturnType, class PreprocessingImpl, typename Function, typename Key,
          typename KeyHash, typename KeyEqualTo>
bool FunctionMatcherMixin<
    ReturnType, PreprocessingImpl, Function, Key, KeyHash,
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
template <typename ReturnType,
          class PreprocessingImpl,
          typename Function,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
std::tuple<Function, utils::ConversionSchemaConditional>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function,
                   Key, KeyHash, KeyEqualTo>::
    GetFunction(std::vector<format::Format*>packed_sfs, Key key, ConversionMap map, std::vector<context::Context*> contexts,
                utils::Converter &sc) {
  utils::ConversionSchemaConditional cs;
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
    std::tuple<Function, utils::ConversionSchemaConditional> best_conversion;
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
template <typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo,
          typename Function>
template <typename F>
std::vector<std::type_index> 
FunctionMatcherMixin<ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo, Function>::PackFormats(F sf) {
  return {sf->get_format_id()};
}
template <typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo,
          typename Function>
template <typename F, typename... SF>
std::vector<std::type_index> 
FunctionMatcherMixin<ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo, Function>::PackFormats(F sf, SF... sfs) {
  std::vector<std::type_index> f = {sf->get_format()};
  std::vector<std::type_index> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo,
          typename Function>
template <typename F>
std::vector<F>
FunctionMatcherMixin<ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo, Function>::PackSFS(F sf) {
  return {sf};
}
template <typename ReturnType,
          class PreprocessingImpl,
          typename Key, typename KeyHash,
          typename KeyEqualTo,
          typename Function>
template <typename F, typename... SF>
std::vector<F>
FunctionMatcherMixin<ReturnType, PreprocessingImpl,
                   Key, KeyHash,
                   KeyEqualTo, Function>::PackSFS(F sf, SF... sfs) {
  std::vector<F> f = {sf};
  std::vector<F> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename ReturnType,
          class PreprocessingImpl,
          typename Function,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::tuple<std::vector<format::Format*>, ReturnType>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function,
                   Key, KeyHash, KeyEqualTo>::
CachedExecute(PreprocessParams * params, utils::Converter& sc, std::vector<context::Context*> contexts, F sf,
        SF... sfs) {
  ConversionMap map = this->_map_to_function;
  // pack the SFs into a vector
  std::vector<format::Format *> packed_sfs = PackSFS(sf, sfs...);
  // pack the SF formats into a vector
  std::vector<std::type_index> formats = PackFormats(sf, sfs...);
  // get conversion schema
  std::tuple<Function, utils::ConversionSchemaConditional> ret =
      GetFunction(packed_sfs, formats, map, contexts, sc);
  Function func = std::get<0>(ret);
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

template <typename ReturnType,
          class PreprocessingImpl,
          typename Function,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
ReturnType
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function,
                   Key, KeyHash, KeyEqualTo>::
    Execute(PreprocessParams * params, utils::Converter& sc, std::vector<context::Context*> contexts, F sf,
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
  this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()},
      CalculateReorderCSR);
  this->params_ = std::unique_ptr<DegreeReorderParams>(
      new DegreeReorderParams(hyperparameter));
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorder(Format * format, std::vector<context::Context*> contexts){
  return this->Execute(this->params_.get(), *(this->sc_), contexts, format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorder(Format * format, PreprocessParams* params, std::vector<context::Context*> contexts){
  return this->Execute(params, *(this->sc_), contexts, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format*>,IDType *> ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorderCached(Format * format, std::vector<context::Context*> contexts){
  return this->CachedExecute(this->params_.get(), *(this->sc_), contexts, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format*>,IDType *> ReorderPreprocessType<IDType, NNZType, ValueType>::GetReorderCached(Format * format, PreprocessParams* params, std::vector<context::Context*> contexts){
  return this->CachedExecute(params, *(this->sc_), contexts, format);
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
  this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
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
  this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
  this->RegisterFunction({CSR<IDType, NNZType, ValueType>::get_format_id_static()}, TransformCSR);
  this->params_ = std::unique_ptr<TransformParams>(new TransformParams(order));
}
template <typename IDType, typename NNZType, typename ValueType>
TransformPreprocessType<IDType, NNZType,
                        ValueType>::~TransformPreprocessType() = default;
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
  return this->CachedExecute(this->params_.get(), *(this->sc_), contexts, csr);
}

template <typename IDType, typename NNZType, typename ValueType>
Format*
TransformPreprocessType<IDType, NNZType, ValueType>::GetTransformation(Format *csr, std::vector<context::Context*> contexts) {
//  std::tuple<TransformFunction<IDType, NNZType, ValueType, ReturnType>,
//             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
//      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
//  TransformFunction<IDType, NNZType, ValueType, ReturnType> func = std::get<0>(func_formats);
//  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
//  return func(sfs, ordr);
return this->Execute(this->params_.get(), *(this->sc_), contexts, csr);
 //   Format *csr) {
 // auto tp = this->Execute(this->sc_, csr);
 // auto params =  this->params_.get();
 // auto func = std::get<0>(tp);
 // auto formats = std::get<1>(tp);
 // return func(formats, params);
 //   //return this->Execute(this->params_.get(), this->sc_, csr);
}

template <typename FeatureType>
FeaturePreprocessType<FeatureType>::~FeaturePreprocessType()= default;

template <typename FeatureType>
std::shared_ptr<PreprocessParams>
FeaturePreprocessType<FeatureType>::get_params() {
  return this->params_;
}
template <typename FeatureType>
std::shared_ptr<PreprocessParams>
FeaturePreprocessType<FeatureType>::get_params(
    std::type_index t) {
  if(this->pmap_.find(t) != this->pmap_.end()){
    return this->pmap_[t];
  }
  else{
    throw utils::FeatureParamsException(get_feature_id().name(), t.name());
  }
}
template <typename FeatureType>
void FeaturePreprocessType<FeatureType>::set_params(
    std::type_index t, std::shared_ptr<PreprocessParams> p) {
  auto ids = this->get_sub_ids();
  if(std::find(ids.begin(), ids.end(), t) != ids.end()){
    this->pmap_[t] = p;
  }
  else{
    throw utils::FeatureParamsException(get_feature_id().name(), t.name());
  }
}
template <typename FeatureType>
std::type_index FeaturePreprocessType<FeatureType>::get_feature_id() {
  return typeid(*this);
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights(){
  this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
  #ifdef CUDA
    std::vector<std::type_index> formats ={CUDACSR<IDType, NNZType, ValueType>::get_format_id_static()}; 
    this->RegisterFunction(formats, GetJaccardWeightCUDACSR);
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
    return this->Execute(&params, *(this->sc_), contexts, format); //func(sfs, this->params_.get());
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(){
    this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
    Register();
    this->params_ = std::shared_ptr<DegreeDistributionParams>(new DegreeDistributionParams());
    this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(
    const DegreeDistribution & d) {
    Register();
    this->params_ = d.params_;
    this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(
    const std::shared_ptr<DegreeDistributionParams> p) {
  this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
void DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction({CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetDegreeDistributionCSR);
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::unordered_map<std::type_index, std::any> DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Extract(format::Format * format, std::vector<context::Context*> c){
  return {{this->get_feature_id(), std::forward<FeatureType*>(GetDistribution(format, c))}};
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<std::type_index> DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() { return {typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>)}; }

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<ExtractableType*> DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_subs(){ return {new DegreeDistribution<IDType, NNZType, ValueType, FeatureType>(*this)}; }

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::type_index DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() { return typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>); }

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::~DegreeDistribution()= default;

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::tuple<std::vector<format::Format*>, FeatureType*> DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistributionCached(Format * format, std::vector<context::Context*> contexts){
  //std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType>,
  //            std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //    func_formats =
  //DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func = std::get<0>(func_formats);
  //std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->CachedExecute(&params, *(this->sc_), contexts, format); //func(sfs, this->params_.get());
}
template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType * DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(Format * format, std::vector<context::Context*> contexts){
    //std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType>,
    //            std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
    //    func_formats = 
    //DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func = std::get<0>(func_formats);
    //std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
    DegreeDistributionParams params;
    return this->Execute(&params, *(this->sc_), contexts, format); //func(sfs, this->params_.get());
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType * DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(object::Graph<IDType, NNZType, ValueType> * obj, std::vector<context::Context*> contexts){
    //std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType>,
    //            std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
    //    func_formats = 
    //DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func = std::get<0>(func_formats);
    //std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
    Format * format = obj->get_connectivity();
    return this->Execute(this->params_.get(), *(this->sc_), contexts, format); //func(sfs, this->params_.get());
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

template<typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(){
  this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = std::shared_ptr<DegreesParams>(new DegreesParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(const Degrees<IDType, NNZType, ValueType> & d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(const std::shared_ptr<DegreesParams> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_feature_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::~Degrees()= default;

template <typename IDType, typename NNZType, typename ValueType>
void Degrees<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction({CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetDegreesCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index> Degrees<IDType, NNZType, ValueType>::get_sub_ids() { return {typeid(Degrees<IDType, NNZType, ValueType>)}; }

template <typename IDType, typename NNZType, typename ValueType>
std::vector<ExtractableType*> Degrees<IDType, NNZType, ValueType>::get_subs(){ return {new Degrees<IDType, NNZType, ValueType>(*this)}; }

template <typename IDType, typename NNZType, typename ValueType>
std::type_index Degrees<IDType, NNZType, ValueType>::get_feature_id_static() { return typeid(Degrees<IDType, NNZType, ValueType>); }

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any> Degrees<IDType, NNZType, ValueType>::Extract(format::Format * format, std::vector<context::Context*> c){
  return {{this->get_feature_id(), std::forward<IDType*>(GetDegrees(format, c))}};
};

template<typename IDType, typename NNZType, typename ValueType>
IDType * Degrees<IDType, NNZType, ValueType>::GetDegrees(Format * format, std::vector<context::Context*> c){
  return this->Execute(this->params_.get(),*(this->sc_), c, format);
}

template<typename IDType, typename NNZType, typename ValueType>
IDType * Degrees<IDType, NNZType, ValueType>::GetDegreesCSR(std::vector<Format *> formats, PreprocessParams * params){
  auto csr = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto dims = csr->get_dimensions();
  for(auto dim : dims){
    std::cout << dim << std::endl;
  }
  IDType num_vertices = dims[0];
  NNZType num_edges = csr->get_num_nnz();
  IDType * degrees = new IDType[num_vertices]();
  auto * rows = csr->get_row_ptr();
  for(int i = 0; i < num_vertices; i++){
    degrees[i] = rows[i+1] - rows[i];
  }
  return degrees;
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Degrees_DegreeDistribution(){
  this->SetConverter(utils::OrderTwoConverter<IDType, NNZType, ValueType>{});
  this->RegisterFunction({CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GetCSR);
  this->params_ = std::shared_ptr<Params>(new Params());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::~Degrees_DegreeDistribution()=default;

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<std::type_index> Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  std::vector<std::type_index> r = {typeid(Degrees<IDType, NNZType, ValueType>), typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>)};
  std::sort(r.begin(), r.end());
  return r;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<ExtractableType*> Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_subs(){
  auto * f1 = new Degrees<IDType, NNZType, ValueType>();
  if(this->pmap_.find(Degrees<IDType, NNZType, ValueType>::get_feature_id_static()) != this->pmap_.end()){
    f1->set_params(Degrees<IDType, NNZType, ValueType>::get_feature_id_static(), this->pmap_[Degrees<IDType, NNZType, ValueType>::get_feature_id_static()]);
  }

  auto * f2 = new DegreeDistribution<IDType, NNZType, ValueType, FeatureType>();
  if(this->pmap_.find(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static()) != this->pmap_.end()){
    f1->set_params(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static(), this->pmap_[DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static()]);
  }
  return {f1, f2};
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::type_index Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() { return typeid(Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>); }

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::unordered_map<std::type_index, std::any> Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Extract(format::Format * format, std::vector<context::Context*> c){
  return Get(format, c);
};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::unordered_map<std::type_index, std::any> Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Get(Format * format, std::vector<context::Context*> c){
  Params params;
  return this->Execute(this->params_.get(),*(this->sc_), c, format);
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::unordered_map<std::type_index, std::any> Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetCSR(std::vector<Format *> formats, PreprocessParams * params){
  auto csr = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto dims = csr->get_dimensions();
  IDType num_vertices = dims[0];
  NNZType num_edges = csr->get_num_nnz();
  auto * degrees = new IDType[num_vertices]();
  auto * dist = new FeatureType[num_vertices];
  auto * rows = csr->get_row_ptr();
  for(int i = 0; i < num_vertices; i++){
    degrees[i] = rows[i+1] - rows[i];
    dist[i] = (rows[i+1] - rows[i]) / (FeatureType)num_edges;
  }
  return {{Degrees<IDType, NNZType, ValueType>::get_feature_id_static(), std::forward<IDType*>(degrees)},
          {DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static(), std::forward<FeatureType*>(dist)}};
}

#if !defined(_HEADER_ONLY)
#include "init/preprocess.inc"
#endif

} // namespace preprocess

} // namespace sparsebase
