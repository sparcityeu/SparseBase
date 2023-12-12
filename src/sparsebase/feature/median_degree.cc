#include "sparsebase/feature/median_degree.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegree<IDType, NNZType, ValueType,
                   FeatureType>::MedianDegree() {
  Register();
  this->params_ =
      std::shared_ptr<MedianDegreeParams>(new MedianDegreeParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegree<IDType, NNZType, ValueType, FeatureType>::MedianDegree(
    MedianDegreeParams params) {
  MedianDegree();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegree<IDType, NNZType, ValueType, FeatureType>::MedianDegree(
    const MedianDegree &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegree<IDType, NNZType, ValueType, FeatureType>::MedianDegree(
    const std::shared_ptr<MedianDegreeParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void MedianDegree<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetMedianDegreeCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
MedianDegree<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetMedianDegree(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
MedianDegree<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(MedianDegree<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
MedianDegree<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new MedianDegree<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
MedianDegree<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(MedianDegree<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegree<IDType, NNZType, ValueType,
                   FeatureType>::~MedianDegree() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
MedianDegree<IDType, NNZType, ValueType, FeatureType>::
    GetMedianDegreeCached(format::Format *format,
                          std::vector<context::Context *> contexts,
                          bool convert_input) {
  MedianDegreeParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
MedianDegree<IDType, NNZType, ValueType, FeatureType>::GetMedianDegree(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  MedianDegreeParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
MedianDegree<IDType, NNZType, ValueType, FeatureType>::GetMedianDegree(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *MedianDegree<IDType, NNZType, ValueType, FeatureType>::
    GetMedianDegreeCSR(std::vector<format::Format *> formats,
                             utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType num_vertices =  csr->get_dimensions()[0];
  auto *rows = csr->get_row_ptr();
  
  std::vector<NNZType> degrees;
  degrees.reserve(num_vertices);
  for (IDType i = 0; i < num_vertices; i++) {
    NNZType degree = rows[i + 1] - rows[i];
    degrees.push_back(degree);
  }
  std::sort(degrees.begin(), degrees.end());  
  if (num_vertices % 2 == 0) {
    return new FeatureType((FeatureType)(degrees[num_vertices / 2 - 1] + degrees[num_vertices / 2]) / 2.0);
  } 
  else {
    return new FeatureType(degrees[num_vertices / 2]);
  }
}

#if !defined(_HEADER_ONLY)
#include "init/median_degree.inc"
#endif
}  // namespace sparsebase::feature