#include "sparsebase/feature/geometric_avg_degree.h"

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
GeometricAvgDegree<IDType, NNZType, ValueType,
                   FeatureType>::GeometricAvgDegree() {
  Register();
  this->params_ =
      std::shared_ptr<GeometricAvgDegreeParams>(new GeometricAvgDegreeParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::GeometricAvgDegree(
    GeometricAvgDegreeParams params) {
  GeometricAvgDegree();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::GeometricAvgDegree(
    const GeometricAvgDegree &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::GeometricAvgDegree(
    const std::shared_ptr<GeometricAvgDegreeParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetGeometricAvgDegreeCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetGeometricAvgDegree(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
GeometricAvgDegree<IDType, NNZType, ValueType,
                   FeatureType>::~GeometricAvgDegree() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::
    GetGeometricAvgDegreeCached(format::Format *format,
                          std::vector<context::Context *> contexts,
                          bool convert_input) {
  GeometricAvgDegreeParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::GetGeometricAvgDegree(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  GeometricAvgDegreeParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::GetGeometricAvgDegree(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *GeometricAvgDegree<IDType, NNZType, ValueType, FeatureType>::
    GetGeometricAvgDegreeCSR(std::vector<format::Format *> formats,
                             utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType num_vertices = csr->get_dimensions()[0];
  auto *rows = csr->get_row_ptr();
  FeatureType sum = 0.0;
  for (int i = 0; i < num_vertices; i++) {
    sum += log(rows[i + 1] - rows[i]);
  }
  return new FeatureType(exp(sum/num_vertices));
}

#if !defined(_HEADER_ONLY)
#include "init/geometric_avg_degree.inc"
#endif
}  // namespace sparsebase::feature
