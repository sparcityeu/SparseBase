#include "preprocess.h"
#include "sparsebase/format/format.h"
#include "sparsebase/utils/converter/converter.h"
#ifdef CUDA
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
        message += f->get_format_id().name();
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
                                                    contexts,
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
                                              contexts,
                                          F sf, SF... sfs) {
  auto cached_output = CachedExecute(params, sc, contexts, sf, sfs...);
  auto converted_formats = std::get<0>(cached_output);
  auto return_object = std::get<1>(cached_output);
  for (auto *converted_format : converted_formats) {
    if (converted_format != nullptr)
      delete converted_format;
  }
  return return_object;
}
template <typename IDType, typename NNZType, typename ValueType>
GenericReorder<IDType, NNZType, ValueType>::GenericReorder() {}
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
    std::vector<context::Context *> contexts) {
  return this->Execute(params, (this->sc_.get()), contexts, format);
}

template <typename ReturnType>
std::tuple<std::vector<format::Format *>, int>
GenericPreprocessType<ReturnType>::GetOutputCached(
    Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts) {
  return this->CachedExecute(params, (this->sc_.get()), contexts, format);
}
template <typename IDType>
IDType *ReorderPreprocessType<IDType>::GetReorder(
    Format *format, std::vector<context::Context *> contexts) {
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format);
}

template <typename IDType>
IDType *ReorderPreprocessType<IDType>::GetReorder(
    Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts) {
  return this->Execute(params, (this->sc_.get()), contexts, format);
}

template <typename IDType>
std::tuple<std::vector<format::Format *>, IDType *>
ReorderPreprocessType<IDType>::GetReorderCached(
    Format *format, std::vector<context::Context *> contexts) {
  return this->CachedExecute(this->params_.get(), (this->sc_.get()), contexts,
                             format);
}

template <typename IDType>
std::tuple<std::vector<format::Format *>, IDType *>
ReorderPreprocessType<IDType>::GetReorderCached(
    Format *format, PreprocessParams *params,
    std::vector<context::Context *> contexts) {
  return this->CachedExecute(params, (this->sc_.get()), contexts, format);
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

template <typename IDType, typename NNZType, typename ValueType>
Transform<IDType, NNZType, ValueType>::Transform(IDType *order) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  this->RegisterFunction(
      {CSR<IDType, NNZType, ValueType>::get_format_id_static()}, TransformCSR);
  this->params_ = std::unique_ptr<TransformParams>(new TransformParams(order));
}
template <typename IDType, typename NNZType, typename ValueType>
TransformPreprocessType<IDType, NNZType,
                        ValueType>::~TransformPreprocessType() = default;
template <typename IDType, typename NNZType, typename ValueType>
Format *Transform<IDType, NNZType, ValueType>::TransformCSR(
    std::vector<Format *> formats, PreprocessParams *params) {
  auto *sp = formats[0]->As<CSR<IDType, NNZType, ValueType>>();
  auto order = static_cast<TransformParams *>(params)->order;
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
std::tuple<std::vector<format::Format *>, format::Format *>
TransformPreprocessType<IDType, NNZType, ValueType>::GetTransformationCached(
    Format *csr, std::vector<context::Context *> contexts) {
  return this->CachedExecute(this->params_.get(), (this->sc_.get()), contexts,
                             csr);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format *>, format::Format *>
TransformPreprocessType<IDType, NNZType, ValueType>::GetTransformationCached(
    Format *csr, PreprocessParams *params,
    std::vector<context::Context *> contexts) {
  return this->CachedExecute(params, (this->sc_.get()), contexts, csr);
}

template <typename IDType, typename NNZType, typename ValueType>
Format *TransformPreprocessType<IDType, NNZType, ValueType>::GetTransformation(
    Format *csr, std::vector<context::Context *> contexts) {
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts, csr);
}

template <typename IDType, typename NNZType, typename ValueType>
Format *TransformPreprocessType<IDType, NNZType, ValueType>::GetTransformation(
    Format *csr, PreprocessParams *params,
    std::vector<context::Context *> contexts) {
  return this->Execute(params, (this->sc_.get()), contexts, csr);
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
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
#ifdef CUDA
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
    Format *format, std::vector<context::Context *> contexts) {
  return this->Execute(nullptr, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
#ifdef CUDA
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
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<FeatureType *>(GetDistribution(format, c))}};
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
                          std::vector<context::Context *> contexts) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(
    Format *format, std::vector<context::Context *> contexts) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
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
    format::Format *format, std::vector<context::Context *> c) {
  return {
      {this->get_feature_id(), std::forward<IDType *>(GetDegrees(format, c))}};
};

template <typename IDType, typename NNZType, typename ValueType>
IDType *Degrees<IDType, NNZType, ValueType>::GetDegrees(
    Format *format, std::vector<context::Context *> c) {
  return this->Execute(this->params_.get(), (this->sc_.get()), c, format);
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
    format::Format *format, std::vector<context::Context *> c) {
  return Get(format, c);
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Get(
    Format *format, std::vector<context::Context *> c) {
  Params params;
  return this->Execute(this->params_.get(), (this->sc_.get()), c, format);
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

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumSlices<IDType, NNZType, ValueType, FeatureType>::NumSlices() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<NumSlicesParams>(new NumSlicesParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumSlices<IDType, NNZType, ValueType, FeatureType>::NumSlices(
    const NumSlices &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumSlices<IDType, NNZType, ValueType, FeatureType>::NumSlices(
    const std::shared_ptr<NumSlicesParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
NumSlices<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<FeatureType>(GetNumSlices(format, c))}};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
NumSlices<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(NumSlices<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
NumSlices<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new NumSlices<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index NumSlices<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(NumSlices<IDType, NNZType, ValueType, FeatureType>);
}



template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType>
NumSlices<IDType, NNZType, ValueType, FeatureType>::GetNumSlicesCached(Format *format,
                   std::vector<context::Context *> contexts) {

  NumSlicesParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumSlices<IDType, NNZType, ValueType, FeatureType>::GetNumSlices(
    Format *format, std::vector<context::Context *> contexts) {

  NumSlicesParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumSlices<IDType, NNZType, ValueType, FeatureType>::GetNumSlices(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumSlices<IDType, NNZType, ValueType, FeatureType>::GetNumSlicesCOO(std::vector<Format *> formats,
                PreprocessParams *params) {
  auto coo = formats[0]->As<COO<IDType, NNZType, ValueType>>();

  int order = coo->get_order();
  std::vector<DimensionType> dim = coo->get_dimensions();

  std::cout << order << " " << dim[0];

  FeatureType num_slices = 0;

  for (int i = 0; i < order - 1; i++)
  {
    for (int j = i + 1; j < order; j++)
    {
      FeatureType mult = 1;
      for (int k = 0; k < order; k++)
      {
        if (k != i && k != j)
          mult *= dim.at(k);
      }
      num_slices += mult;
    }
  }

  return num_slices;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
 FeatureType
NumSlices<IDType, NNZType, ValueType, FeatureType>::GetNumSlicesHigherOrderCOO(std::vector<Format *> formats,
                           PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();
  std::vector<DimensionType> dim = hocoo->get_dimensions();

  FeatureType num_slices = 0;

  for (int i = 0; i < order - 1; i++)
  {
    for (int j = i + 1; j < order; j++)
    {
      FeatureType mult = 1;
      for (int k = 0; k < order; k++)
      {
        if (k != i && k != j)
          mult *= dim.at(k);
      }
      num_slices += mult;
    }
  }

  return num_slices;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void NumSlices<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {COO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumSlicesCOO);
  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumSlicesHigherOrderCOO);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumFibers<IDType, NNZType, ValueType, FeatureType>::NumFibers() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<NumFibersParams>(new NumFibersParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumFibers<IDType, NNZType, ValueType, FeatureType>::NumFibers(
    const NumFibers &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumFibers<IDType, NNZType, ValueType, FeatureType>::NumFibers(
    const std::shared_ptr<NumFibersParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
NumFibers<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<FeatureType>(GetNumFibers(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
NumFibers<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(NumFibers<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
NumFibers<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new NumFibers<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index NumFibers<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(NumFibers<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType>
NumFibers<IDType, NNZType, ValueType, FeatureType>::GetNumFibersCached(Format *format,
                   std::vector<context::Context *> contexts) {

  NumFibersParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumFibers<IDType, NNZType, ValueType, FeatureType>::GetNumFibers(
    Format *format, std::vector<context::Context *> contexts) {

  NumFibersParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumFibers<IDType, NNZType, ValueType, FeatureType>::GetNumFibers(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
 FeatureType
NumFibers<IDType, NNZType, ValueType, FeatureType>::GetNumFibersCOO(std::vector<Format *> formats,
                PreprocessParams *params) {
  auto coo = formats[0]->As<COO<IDType, NNZType, ValueType>>();

  int order = coo->get_order();
  std::vector<DimensionType> dim = coo->get_dimensions();

  FeatureType num_fibers = 0;

  for (int i = 0; i < order; i++)
  {
    FeatureType mult = 1;
    for (int j = 0; j < order; j++)
      if (j != i)
        mult *= dim[j];
    num_fibers += mult;
  }
  return num_fibers;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
 FeatureType
NumFibers<IDType, NNZType, ValueType, FeatureType>::GetNumFibersHigherOrderCOO(std::vector<Format *> formats,
                           PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();

  FeatureType num_fibers = 0;

  for (int i = 0; i < order; i++)
  {
    FeatureType mult = 1;
    for (int j = 0; j < order; j++)
      if (j != i)
        mult *= dim[j];

    num_fibers += mult;
  }

  return num_fibers;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void NumFibers<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {COO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumFibersCOO);

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumFibersHigherOrderCOO);
}

// NNZ

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::NnzPerFiber() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<NnzPerFiberParams>(new NnzPerFiberParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::NnzPerFiber(
    const NnzPerFiber &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::NnzPerFiber(
    const std::shared_ptr<NnzPerFiberParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType *>(GetNnzPerFiber(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(NnzPerFiber<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new NnzPerFiber<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(NnzPerFiber<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<format::Format *>, NNZType *>
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::GetNnzPerFiberCached(Format *format,
                                                                           std::vector<context::Context *> contexts) {

  NnzPerFiberParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NNZType *
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::GetNnzPerFiber(
    Format *format, std::vector<context::Context *> contexts) {

  NnzPerFiberParams params;
  return this->Execute(&params, (this->sc_.get()), contexts, format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NNZType *
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::GetNnzPerFiber(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NNZType *
NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::GetNnzPerFiberHigherOrderCOO(std::vector<Format *> formats,
                                                                                   PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  DimensionType order = hocoo->get_order();

  sparsebase::preprocess::NumFibers<IDType, NNZType, ValueType, FeatureType> num_fibers;
  sparsebase::context::CPUContext cpu_context;

  FeatureType num_fibers_feature = num_fibers.GetNumFibers(hocoo, {&cpu_context});

  NNZType * nnz_per_fiber = new NNZType[num_fibers_feature];

  std::vector<DimensionType> dimensions = hocoo->get_dimensions();

  int start_index = 0;
  for (DimensionType i = 0; i < order; i++)
  {
    DimensionType x1 = (i + 1) % order;
    DimensionType x2 = (x1+ 1) % order;

    DimensionType x1_dim = dimensions[x1];
    DimensionType x2_dim = dimensions[x2];

    IDType *x1_indices = hocoo->get_indices()[x1];
    IDType *x2_indices = hocoo->get_indices()[x2];

    NNZType *x1x2_fibers = nnz_per_fiber + start_index;

    for(int j=0; j<hocoo->get_num_nnz(); j++){
      x1x2_fibers[x1_indices[j] + x2_indices[j] * x1_dim]++;
    }
    start_index += x1_dim*x2_dim;
  }

  return nnz_per_fiber;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void NnzPerFiber<IDType, NNZType, ValueType, FeatureType>::Register() {

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNnzPerFiberHigherOrderCOO);
}

//<NnzPerSlice>

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::NnzPerSlice() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<NnzPerSliceParams>(new NnzPerSliceParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::NnzPerSlice(
    const NnzPerSlice &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::NnzPerSlice(
    const std::shared_ptr<NnzPerSliceParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType *>(GetNnzPerSlice(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(NnzPerSlice<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new NnzPerSlice<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(NnzPerSlice<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<format::Format *>, NNZType *>
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetNnzPerSliceCached(Format *format,
                                                                           std::vector<context::Context *> contexts) {

  NnzPerSliceParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NNZType *
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetNnzPerSlice(
    Format *format, std::vector<context::Context *> contexts) {

  NnzPerSliceParams params;
  return this->Execute(&params, (this->sc_.get()), contexts, format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NNZType *
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetNnzPerSlice(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NNZType *
NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetNnzPerSliceHigherOrderCOO(std::vector<Format *> formats,
                                                                                   PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  DimensionType order = hocoo->get_order();

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, FeatureType> num_slices;
  sparsebase::context::CPUContext cpu_context;

  FeatureType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  NNZType * nnz_per_slice = new NNZType[num_slices_feature];

  std::vector<DimensionType> dimensions = hocoo->get_dimensions();

  int start_index = 0;
  for (int i = 0; i < order; i++){
    IDType *curr_indices = hocoo->get_indices()[i];
    NNZType *curr_slices = nnz_per_slice + start_index;

    for(int j=0; j < hocoo->get_num_nnz(); j++){
      curr_slices[curr_indices[j]]++;
    }
    start_index += dimensions[i];
  }

  return nnz_per_slice;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void NnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Register() {

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNnzPerSliceHigherOrderCOO);
}

//</NnzPerSlice>

//<NumNnzFibers>

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::NumNnzFibers() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<NumNnzFibersParams>(new NumNnzFibersParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::NumNnzFibers(
    const NumNnzFibers &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::NumNnzFibers(
    const std::shared_ptr<NumNnzFibersParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<FeatureType>(GetNumNnzFibers(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(NumNnzFibers<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new NumNnzFibers<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(NumNnzFibers<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType>
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::GetNumNnzFibersCached(Format *format,
                                                                       std::vector<context::Context *> contexts) {

  NumNnzFibersParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::GetNumNnzFibers(
    Format *format, std::vector<context::Context *> contexts) {

  NumNnzFibersParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::GetNumNnzFibers(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::GetNumNnzFibersCOO(std::vector<Format *> formats,
                                                                    PreprocessParams *params) {
  auto coo = formats[0]->As<COO<IDType, NNZType, ValueType>>();

  int order = coo->get_order();
  std::vector<DimensionType> dim = coo->get_dimensions();

  FeatureType num_fibers = 0;

  for (int i = 0; i < order; i++)
  {
    FeatureType mult = 1;
    for (int j = 0; j < order; j++)
      if (j != i)
        mult *= dim[j];
    num_fibers += mult;
  }
  return num_fibers;
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::GetNumNnzFibersHigherOrderCOO(std::vector<Format *> formats,
                                                                               PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumFibers<IDType, NNZType, ValueType, FeatureType> num_fibers;
  FeatureType num_fibers_feature = num_fibers.GetNumFibers(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerFiber<IDType, NNZType, ValueType, FeatureType> nnz_per_fiber;
  NNZType * nnz_per_fiber_feature = nnz_per_fiber.GetNnzPerFiber(hocoo, {&cpu_context});

  return std::transform_reduce(nnz_per_fiber_feature, nnz_per_fiber_feature + num_fibers_feature, 0, std::plus<>(), []( auto val) { return val != 0; });
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void NumNnzFibers<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {COO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumNnzFibersCOO);

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumNnzFibersHigherOrderCOO);
}

//</NumNnzFibers>

//<NumNnzSlices>

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::NumNnzSlices() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<NumNnzSlicesParams>(new NumNnzSlicesParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::NumNnzSlices(
    const NumNnzSlices &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::NumNnzSlices(
    const std::shared_ptr<NumNnzSlicesParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<FeatureType>(GetNumNnzSlices(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(NumNnzSlices<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<ExtractableType *>
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new NumNnzSlices<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(NumNnzSlices<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType>
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::GetNumNnzSlicesCached(Format *format,
                                                                             std::vector<context::Context *> contexts) {

  NumNnzSlicesParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::GetNumNnzSlices(
    Format *format, std::vector<context::Context *> contexts) {

  NumNnzSlicesParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::GetNumNnzSlices(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::GetNumNnzSlicesCOO(std::vector<Format *> formats,
                                                                          PreprocessParams *params) {
  auto coo = formats[0]->As<COO<IDType, NNZType, ValueType>>();

  int order = coo->get_order();
  std::vector<DimensionType> dim = coo->get_dimensions();

  FeatureType num_Slices = 0;

  for (int i = 0; i < order; i++)
  {
    FeatureType mult = 1;
    for (int j = 0; j < order; j++)
      if (j != i)
        mult *= dim[j];
    num_Slices += mult;
  }
  return num_Slices;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType
NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::GetNumNnzSlicesHigherOrderCOO(std::vector<Format *> formats,
                                                                                     PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, FeatureType> num_slices;
  FeatureType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerSlice<IDType, NNZType, ValueType, FeatureType> nnz_per_slice;
  NNZType * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(hocoo, {&cpu_context});

  return std::transform_reduce(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature, 0, std::plus<>(), []( auto val) { return val != 0; });
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void NumNnzSlices<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {COO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumNnzSlicesCOO);

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetNumNnzSlicesHigherOrderCOO);
}

//</NumNnzSlices>

//<MaxNnzPerSlice>

template <typename IDType, typename NNZType, typename ValueType>
MaxNnzPerSlice<IDType, NNZType, ValueType>::MaxNnzPerSlice() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<MaxNnzPerSliceParams>(new MaxNnzPerSliceParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
MaxNnzPerSlice<IDType, NNZType, ValueType>::MaxNnzPerSlice(
    const MaxNnzPerSlice &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
MaxNnzPerSlice<IDType, NNZType, ValueType>::MaxNnzPerSlice(
    const std::shared_ptr<MaxNnzPerSliceParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
MaxNnzPerSlice<IDType, NNZType, ValueType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType>(GetMaxNnzPerSlice(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
MaxNnzPerSlice<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(MaxNnzPerSlice<IDType, NNZType, ValueType>)};
}
template <typename IDType, typename NNZType, typename ValueType>
std::vector<ExtractableType *>
MaxNnzPerSlice<IDType, NNZType, ValueType>::get_subs() {
  return {
      new MaxNnzPerSlice<IDType, NNZType, ValueType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType>
std::type_index MaxNnzPerSlice<IDType, NNZType, ValueType>::get_feature_id_static() {
  return typeid(MaxNnzPerSlice<IDType, NNZType, ValueType>);
}
template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format *>, NNZType>
MaxNnzPerSlice<IDType, NNZType, ValueType>::GetMaxNnzPerSliceCached(Format *format,
                                                                    std::vector<context::Context *> contexts) {

  MaxNnzPerSliceParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType
MaxNnzPerSlice<IDType, NNZType, ValueType>::GetMaxNnzPerSlice(
    Format *format, std::vector<context::Context *> contexts) {

  MaxNnzPerSliceParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType
MaxNnzPerSlice<IDType, NNZType, ValueType>::GetMaxNnzPerSlice(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType
MaxNnzPerSlice<IDType, NNZType, ValueType>::GetMaxNnzPerSliceCOO(std::vector<Format *> formats,
                                                                 PreprocessParams *params) {
  auto coo = formats[0]->As<COO<IDType, NNZType, ValueType>>();

  int order = coo->get_order();
  std::vector<DimensionType> dim = coo->get_dimensions();

  NNZType num_Slices = 0;

  for (int i = 0; i < order; i++)
  {
    NNZType mult = 1;
    for (int j = 0; j < order; j++)
      if (j != i)
        mult *= dim[j];
    num_Slices += mult;
  }
  return num_Slices;
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType
MaxNnzPerSlice<IDType, NNZType, ValueType>::GetMaxNnzPerSliceHigherOrderCOO(std::vector<Format *> formats,
                                                                            PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, NNZType> num_slices;
  NNZType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerSlice<IDType, NNZType, ValueType, NNZType> nnz_per_slice;
  NNZType * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(hocoo, {&cpu_context});

  return *std::max_element(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature);
}

template <typename IDType, typename NNZType, typename ValueType>
void MaxNnzPerSlice<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {COO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetMaxNnzPerSliceCOO);

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetMaxNnzPerSliceHigherOrderCOO);
}

//</MaxNnzPerSlice>



//<MinNnzPerSlice>

template <typename IDType, typename NNZType, typename ValueType>
MinNnzPerSlice<IDType, NNZType, ValueType>::MinNnzPerSlice() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<MinNnzPerSliceParams>(new MinNnzPerSliceParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
MinNnzPerSlice<IDType, NNZType, ValueType>::MinNnzPerSlice(
    const MinNnzPerSlice &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
MinNnzPerSlice<IDType, NNZType, ValueType>::MinNnzPerSlice(
    const std::shared_ptr<MinNnzPerSliceParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
MinNnzPerSlice<IDType, NNZType, ValueType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType>(GetMinNnzPerSlice(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
MinNnzPerSlice<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(MinNnzPerSlice<IDType, NNZType, ValueType>)};
}
template <typename IDType, typename NNZType, typename ValueType>
std::vector<ExtractableType *>
MinNnzPerSlice<IDType, NNZType, ValueType>::get_subs() {
  return {
      new MinNnzPerSlice<IDType, NNZType, ValueType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType>
std::type_index MinNnzPerSlice<IDType, NNZType, ValueType>::get_feature_id_static() {
  return typeid(MinNnzPerSlice<IDType, NNZType, ValueType>);
}
template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format *>, NNZType>
MinNnzPerSlice<IDType, NNZType, ValueType>::GetMinNnzPerSliceCached(Format *format,
                                                                    std::vector<context::Context *> contexts) {

  MinNnzPerSliceParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType
MinNnzPerSlice<IDType, NNZType, ValueType>::GetMinNnzPerSlice(
    Format *format, std::vector<context::Context *> contexts) {

  MinNnzPerSliceParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType
MinNnzPerSlice<IDType, NNZType, ValueType>::GetMinNnzPerSlice(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType
MinNnzPerSlice<IDType, NNZType, ValueType>::GetMinNnzPerSliceCOO(std::vector<Format *> formats,
                                                                 PreprocessParams *params) {
  auto coo = formats[0]->As<COO<IDType, NNZType, ValueType>>();

  int order = coo->get_order();
  std::vector<DimensionType> dim = coo->get_dimensions();

  NNZType num_Slices = 0;

  for (int i = 0; i < order; i++)
  {
    NNZType mult = 1;
    for (int j = 0; j < order; j++)
      if (j != i)
        mult *= dim[j];
    num_Slices += mult;
  }
  return num_Slices;
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType
MinNnzPerSlice<IDType, NNZType, ValueType>::GetMinNnzPerSliceHigherOrderCOO(std::vector<Format *> formats,
                                                                            PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, NNZType> num_slices;
  NNZType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerSlice<IDType, NNZType, ValueType, NNZType> nnz_per_slice;
  NNZType * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(hocoo, {&cpu_context});

  return *std::min_element(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature);
}

template <typename IDType, typename NNZType, typename ValueType>
void MinNnzPerSlice<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {COO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetMinNnzPerSliceCOO);

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetMinNnzPerSliceHigherOrderCOO);
}

//</MinNnzPerSlice>

//<AvgNnzPerSlice>

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::AvgNnzPerSlice() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<AvgNnzPerSliceParams>(new AvgNnzPerSliceParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::AvgNnzPerSlice(
    const AvgNnzPerSlice &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::AvgNnzPerSlice(
    const std::shared_ptr<AvgNnzPerSliceParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::unordered_map<std::type_index, std::any>
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType>(GetAvgNnzPerSlice(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<std::type_index>
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(    AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<ExtractableType *>
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new     AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::type_index     AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(    AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType>
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetAvgNnzPerSliceCached(Format *format,
                                                                                 std::vector<context::Context *> contexts) {

  AvgNnzPerSliceParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetAvgNnzPerSlice(
    Format *format, std::vector<context::Context *> contexts) {

  AvgNnzPerSliceParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetAvgNnzPerSlice(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetAvgNnzPerSliceHigherOrderCOO(std::vector<Format *> formats,
                                                                                         PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, NNZType> num_slices;
  NNZType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerSlice<IDType, NNZType, ValueType, NNZType> nnz_per_slice;
  NNZType * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(hocoo, {&cpu_context});

  sparsebase::preprocess::NumNnzSlices<IDType, NNZType, ValueType, NNZType> num_nnz_slices;
  NNZType num_nnz_slices_feature = num_nnz_slices.GetNumNnzSlices(hocoo, {&cpu_context});

  return std::accumulate(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature, 0) / ((FeatureType) num_nnz_slices_feature);
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
void AvgNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Register() {

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetAvgNnzPerSliceHigherOrderCOO);
}

//</AvgNnzPerSlice>

//<DevNnzPerSlice>

template <typename IDType, typename NNZType, typename ValueType>
DevNnzPerSlice<IDType, NNZType, ValueType>::DevNnzPerSlice() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<DevNnzPerSliceParams>(new DevNnzPerSliceParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
DevNnzPerSlice<IDType, NNZType, ValueType>::DevNnzPerSlice(
    const DevNnzPerSlice &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
DevNnzPerSlice<IDType, NNZType, ValueType>::DevNnzPerSlice(
    const std::shared_ptr<DevNnzPerSliceParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
DevNnzPerSlice<IDType, NNZType, ValueType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType>(GetDevNnzPerSlice(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
DevNnzPerSlice<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(DevNnzPerSlice<IDType, NNZType, ValueType>)};
}
template <typename IDType, typename NNZType, typename ValueType>
std::vector<ExtractableType *>
DevNnzPerSlice<IDType, NNZType, ValueType>::get_subs() {
  return {
      new DevNnzPerSlice<IDType, NNZType, ValueType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType>
std::type_index DevNnzPerSlice<IDType, NNZType, ValueType>::get_feature_id_static() {
  return typeid(DevNnzPerSlice<IDType, NNZType, ValueType>);
}
template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<format::Format *>, NNZType>
DevNnzPerSlice<IDType, NNZType, ValueType>::GetDevNnzPerSliceCached(Format *format,
                                                                    std::vector<context::Context *> contexts) {

  DevNnzPerSliceParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType
DevNnzPerSlice<IDType, NNZType, ValueType>::GetDevNnzPerSlice(
    Format *format, std::vector<context::Context *> contexts) {

  DevNnzPerSliceParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType
DevNnzPerSlice<IDType, NNZType, ValueType>::GetDevNnzPerSlice(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType
DevNnzPerSlice<IDType, NNZType, ValueType>::GetDevNnzPerSliceCOO(std::vector<Format *> formats,
                                                                 PreprocessParams *params) {
  auto coo = formats[0]->As<COO<IDType, NNZType, ValueType>>();

  int order = coo->get_order();
  std::vector<DimensionType> dim = coo->get_dimensions();

  NNZType num_Slices = 0;

  for (int i = 0; i < order; i++)
  {
    NNZType mult = 1;
    for (int j = 0; j < order; j++)
      if (j != i)
        mult *= dim[j];
    num_Slices += mult;
  }
  return num_Slices;
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType
DevNnzPerSlice<IDType, NNZType, ValueType>::GetDevNnzPerSliceHigherOrderCOO(std::vector<Format *> formats,
                                                                            PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, NNZType> num_slices;
  NNZType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerSlice<IDType, NNZType, ValueType, NNZType> nnz_per_slice;
  NNZType * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(hocoo, {&cpu_context});

  return (*std::max_element(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature))
         - (*std::min_element(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature));
}

template <typename IDType, typename NNZType, typename ValueType>
void DevNnzPerSlice<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {COO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetDevNnzPerSliceCOO);

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetDevNnzPerSliceHigherOrderCOO);
}

//</DevNnzPerSlice>


//<StdNnzPerSlice>

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::StdNnzPerSlice() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<StdNnzPerSliceParams>(new StdNnzPerSliceParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::StdNnzPerSlice(
    const StdNnzPerSlice &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::StdNnzPerSlice(
    const std::shared_ptr<StdNnzPerSliceParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::unordered_map<std::type_index, std::any>
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType>(GetStdNnzPerSlice(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<std::type_index>
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(    StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<ExtractableType *>
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new     StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::type_index     StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(    StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType>
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetStdNnzPerSliceCached(Format *format,
                                                                                 std::vector<context::Context *> contexts) {

  StdNnzPerSliceParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetStdNnzPerSlice(
    Format *format, std::vector<context::Context *> contexts) {

  StdNnzPerSliceParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetStdNnzPerSlice(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetStdNnzPerSliceHigherOrderCOO(std::vector<Format *> formats,
                                                                                         PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, NNZType> num_slices;
  NNZType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerSlice<IDType, NNZType, ValueType, NNZType> nnz_per_slice;
  NNZType * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(hocoo, {&cpu_context});

  sparsebase::preprocess::NumNnzSlices<IDType, NNZType, ValueType, NNZType> num_nnz_slices;
  NNZType num_nnz_slices_feature = num_nnz_slices.GetNumNnzSlices(hocoo, {&cpu_context});

  if (num_slices_feature == 1) {
    return (FeatureType) 0.0;
  }
  // Calculate the mean
  const FeatureType mean = std::accumulate(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature , 0.0) / num_nnz_slices_feature;

  // Now calculate the variance
  auto variance_func = [&mean, &num_nnz_slices_feature](FeatureType accumulator, const FeatureType& val) {
    return accumulator + (val > 0) * ((val - mean)*(val - mean) / (num_nnz_slices_feature - 1));
  };

  return sqrt(std::accumulate(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature, 0.0, variance_func));
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
void StdNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Register() {

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetStdNnzPerSliceHigherOrderCOO);
}

//</StdNnzPerSlice>


//<CovNnzPerSlice>

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::CovNnzPerSlice() {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ =
      std::shared_ptr<CovNnzPerSliceParams>(new CovNnzPerSliceParams());
  this->pmap_.insert({get_feature_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::CovNnzPerSlice(
    const CovNnzPerSlice &d) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::CovNnzPerSlice(
    const std::shared_ptr<CovNnzPerSliceParams> p) {
  this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
  Register();
  this->params_ = p;
  this->pmap_[get_feature_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::unordered_map<std::type_index, std::any>
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c) {
  return {{this->get_feature_id(),
           std::forward<NNZType>(GetCovNnzPerSlice(format, c))}};
};
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<std::type_index>
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(    CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>)};
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<ExtractableType *>
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new     CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>(*this)};
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::type_index     CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::get_feature_id_static() {
  return typeid(    CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>);
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::tuple<std::vector<format::Format *>, FeatureType>
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetCovNnzPerSliceCached(Format *format,
                                                                                 std::vector<context::Context *> contexts) {

  CovNnzPerSliceParams params;
  return this->CachedExecute(&params, (this->sc_.get()), contexts,
                             format);
}
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetCovNnzPerSlice(
    Format *format, std::vector<context::Context *> contexts) {

  CovNnzPerSliceParams params;
  return this->Execute(&params, (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetCovNnzPerSlice(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts) {

  Format *format = obj->get_connectivity_to_coo();
  return this->Execute(this->params_.get(), (this->sc_.get()), contexts,
                       format); // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
FeatureType
CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::GetCovNnzPerSliceHigherOrderCOO(std::vector<Format *> formats,
                                                                                         PreprocessParams *params) {
  auto hocoo = formats[0]->As<HigherOrderCOO<IDType, NNZType, ValueType>>();

  int order = hocoo->get_order();

  std::vector<DimensionType> dim = hocoo->get_dimensions();
  sparsebase::context::CPUContext cpu_context;

  sparsebase::preprocess::NumSlices<IDType, NNZType, ValueType, NNZType> num_slices;
  NNZType num_slices_feature = num_slices.GetNumSlices(hocoo, {&cpu_context});

  sparsebase::preprocess::NnzPerSlice<IDType, NNZType, ValueType, NNZType> nnz_per_slice;
  NNZType * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(hocoo, {&cpu_context});

  sparsebase::preprocess::NumNnzSlices<IDType, NNZType, ValueType, NNZType> num_nnz_slices;
  NNZType num_nnz_slices_feature = num_nnz_slices.GetNumNnzSlices(hocoo, {&cpu_context});

  if (num_slices_feature == 1) {
    return (FeatureType) 0.0;
  }
  // Calculate the mean
  const FeatureType mean = std::accumulate(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature , 0.0) / num_nnz_slices_feature;

  // Now calculate the variance
  auto variance_func = [&mean, &num_nnz_slices_feature](FeatureType accumulator, const FeatureType& val) {
    return accumulator + (val > 0) * ((val - mean)*(val - mean) / (num_nnz_slices_feature - 1));
  };

  return sqrt(std::accumulate(nnz_per_slice_feature, nnz_per_slice_feature + num_slices_feature, 0.0, variance_func)) / mean;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
void CovNnzPerSlice<IDType, NNZType, ValueType, FeatureType>::Register() {

  this->RegisterFunction(
      {HigherOrderCOO<IDType, NNZType, ValueType>::get_format_id_static()},
      GetCovNnzPerSliceHigherOrderCOO);
}

//</CovNnzPerSlice>




#if !defined(_HEADER_ONLY)
#include "init/preprocess.inc"
#endif

} // namespace preprocess

} // namespace sparsebase
