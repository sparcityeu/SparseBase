#include "sparsebase/feature/avg_degree_column.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegreeColumn<IDType, NNZType, ValueType,
                   FeatureType>::AvgDegreeColumn() {
  Register();
  this->params_ =
      std::shared_ptr<AvgDegreeColumnParams>(new AvgDegreeColumnParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::AvgDegreeColumn(
    AvgDegreeColumnParams params) {
  AvgDegreeColumn();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::AvgDegreeColumn(
    const AvgDegreeColumn &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::AvgDegreeColumn(
    const std::shared_ptr<AvgDegreeColumnParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSC<IDType, NNZType, ValueType>::get_id_static()},
      GetAvgDegreeColumnCSC);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetAvgDegreeColumn(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegreeColumn<IDType, NNZType, ValueType,
                   FeatureType>::~AvgDegreeColumn() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetAvgDegreeColumnCached(format::Format *format,
                          std::vector<context::Context *> contexts,
                          bool convert_input) {
  AvgDegreeColumnParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetAvgDegreeColumn(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  AvgDegreeColumnParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetAvgDegreeColumn(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *AvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetAvgDegreeColumnCSC(std::vector<format::Format *> formats,
                             utils::Parameters *params) {
  auto csc = formats[0]->AsAbsolute<format::CSC<IDType, NNZType, ValueType>>();
  IDType num_col = csc->get_dimensions()[0];
  FeatureType *avg_degree_column = new FeatureType;
  auto *cols = csc->get_col_ptr();
  NNZType degree_sum = cols[num_col] - cols[0];
  *avg_degree_column = degree_sum / (FeatureType)num_col;
  return avg_degree_column;
}

#if !defined(_HEADER_ONLY)
#include "init/avg_degree_column.inc"
#endif
}  // namespace sparsebase::feature
