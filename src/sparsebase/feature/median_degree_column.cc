//
// Created by Sinan Ekmekcibasi on 8.11.2023.
//

#include "sparsebase/feature/median_degree_column.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegreeColumn<IDType, NNZType, ValueType,
             FeatureType>::MedianDegreeColumn() {
  Register();
  this->params_ =
      std::shared_ptr<MedianDegreeColumnParams>(new MedianDegreeColumnParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::MedianDegreeColumn(
    MedianDegreeColumnParams params) {
  MedianDegreeColumn();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::MedianDegreeColumn(
    const MedianDegreeColumn &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::MedianDegreeColumn(
    const std::shared_ptr<MedianDegreeColumnParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetMedianDegreeColumnCSC);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetMedianDegreeColumn(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MedianDegreeColumn<IDType, NNZType, ValueType,
             FeatureType>::~MedianDegreeColumn() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetMedianDegreeColumnCached(format::Format *format,
                          std::vector<context::Context *> contexts,
                          bool convert_input) {
  MedianDegreeColumnParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetMedianDegreeColumn(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  MedianDegreeColumnParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetMedianDegreeColumn(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *MedianDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetMedianDegreeColumnCSC(std::vector<format::Format *> formats,
                       utils::Parameters *params) {
  auto csc = formats[0]->AsAbsolute<format::CSC<IDType, NNZType, ValueType>>();
  IDType num_col =  csc->get_dimensions()[0];
  auto *cols = csc->get_col_ptr();

  std::vector<NNZType> degrees;
  degrees.reserve(num_col);
  for (IDType i = 0; i < num_col; i++) {
    NNZType degree = cols[i + 1] - cols[i];
    degrees.push_back(degree);
  }
  std::sort(degrees.begin(), degrees.end());
  if (num_col % 2 == 0) {
    return new FeatureType((FeatureType)(degrees[num_col / 2 - 1] + degrees[num_col / 2]) / 2.0);
  }
  else {
    return new FeatureType(degrees[num_col / 2]);
  }
}

#if !defined(_HEADER_ONLY)
#include "init/median_degree_column.inc"
#endif
}  // namespace sparsebase::feature