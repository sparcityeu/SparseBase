#include "sparsebase/feature/avg_degree.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegree<IDType, NNZType, ValueType,
                   FeatureType>::AvgDegree() {
  Register();
  this->params_ =
      std::shared_ptr<AvgDegreeParams>(new AvgDegreeParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegree<IDType, NNZType, ValueType, FeatureType>::AvgDegree(
    AvgDegreeParams params) {
  AvgDegree();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegree<IDType, NNZType, ValueType, FeatureType>::AvgDegree(
    const AvgDegree &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegree<IDType, NNZType, ValueType, FeatureType>::AvgDegree(
    const std::shared_ptr<AvgDegreeParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void AvgDegree<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetAvgDegreeCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
AvgDegree<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetAvgDegree(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
AvgDegree<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(AvgDegree<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
AvgDegree<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new AvgDegree<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
AvgDegree<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(AvgDegree<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
AvgDegree<IDType, NNZType, ValueType,
                   FeatureType>::~AvgDegree() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
AvgDegree<IDType, NNZType, ValueType, FeatureType>::
    GetAvgDegreeCached(format::Format *format,
                          std::vector<context::Context *> contexts,
                          bool convert_input) {
  AvgDegreeParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
AvgDegree<IDType, NNZType, ValueType, FeatureType>::GetAvgDegree(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  AvgDegreeParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
AvgDegree<IDType, NNZType, ValueType, FeatureType>::GetAvgDegree(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *AvgDegree<IDType, NNZType, ValueType, FeatureType>::
    GetAvgDegreeCSR(std::vector<format::Format *> formats,
                             utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType num_vertices = csr->get_dimensions()[0];
  FeatureType *avg_degree = new FeatureType;
  auto *rows = csr->get_row_ptr();
  NNZType degree_sum = rows[num_vertices] - rows[0];
  *avg_degree = degree_sum / (FeatureType)num_vertices;
  return avg_degree;
}

#if !defined(_HEADER_ONLY)
#include "init/avg_degree.inc"
#endif
}  // namespace sparsebase::feature
