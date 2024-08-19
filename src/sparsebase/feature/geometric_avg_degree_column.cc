#include "sparsebase/feature/geometric_avg_degree_column.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType,
                   FeatureType>::GeometricAvgDegreeColumn() {
  Register();
  this->params_ =
      std::shared_ptr<GeometricAvgDegreeColumnParams>(new GeometricAvgDegreeColumnParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GeometricAvgDegreeColumn(
    GeometricAvgDegreeColumnParams params) {
  GeometricAvgDegreeColumn();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GeometricAvgDegreeColumn(
    const GeometricAvgDegreeColumn &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GeometricAvgDegreeColumn(
    const std::shared_ptr<GeometricAvgDegreeColumnParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSC<IDType, NNZType, ValueType>::get_id_static()},
      GetGeometricAvgDegreeColumnCSC);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetGeometricAvgDegreeColumn(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType,
                   FeatureType>::~GeometricAvgDegreeColumn() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetGeometricAvgDegreeColumnCached(format::Format *format,
                                std::vector<context::Context *> contexts,
                                bool convert_input) {
  GeometricAvgDegreeColumnParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetGeometricAvgDegreeColumn(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  GeometricAvgDegreeColumnParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetGeometricAvgDegreeColumn(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *GeometricAvgDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetGeometricAvgDegreeColumnCSC(std::vector<format::Format *> formats,
                             utils::Parameters *params) {
  auto CSC = formats[0]->AsAbsolute<format::CSC<IDType, NNZType, ValueType>>();
  IDType num_cols = CSC->get_dimensions()[0];
  auto *cols = CSC->get_col_ptr();
  FeatureType sum = 0.0;
  for (int i = 0; i < num_cols; i++) {
    sum += log(cols[i + 1] - cols[i]);
  }
  return new FeatureType(exp(sum/num_cols));
}

#if !defined(_HEADER_ONLY)
#include "init/geometric_avg_degree_column.inc"
#endif
}  // namespace sparsebase::feature