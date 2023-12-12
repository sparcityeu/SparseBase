#include "sparsebase/feature/coefficient_of_variation_degree.h"

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
CoefficientOfVariationDegree<IDType, NNZType, ValueType,
                   FeatureType>::CoefficientOfVariationDegree() {
  Register();
  this->params_ =
      std::shared_ptr<CoefficientOfVariationDegreeParams>(new CoefficientOfVariationDegreeParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::CoefficientOfVariationDegree(
    CoefficientOfVariationDegreeParams params) {
  CoefficientOfVariationDegree();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::CoefficientOfVariationDegree(
    const CoefficientOfVariationDegree &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::CoefficientOfVariationDegree(
    const std::shared_ptr<CoefficientOfVariationDegreeParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetCoefficientOfVariationDegreeCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetCoefficientOfVariationDegree(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
CoefficientOfVariationDegree<IDType, NNZType, ValueType,
                   FeatureType>::~CoefficientOfVariationDegree() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::
    GetCoefficientOfVariationDegreeCached(format::Format *format,
                          std::vector<context::Context *> contexts,
                          bool convert_input) {
  CoefficientOfVariationDegreeParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::GetCoefficientOfVariationDegree(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  CoefficientOfVariationDegreeParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::GetCoefficientOfVariationDegree(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *CoefficientOfVariationDegree<IDType, NNZType, ValueType, FeatureType>::
    GetCoefficientOfVariationDegreeCSR(std::vector<format::Format *> formats,
                             utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType num_vertices = csr->get_dimensions()[0];
  FeatureType avg_degree;
  auto *rows = csr->get_row_ptr();
  NNZType degree_sum = rows[num_vertices] - rows[0];
  avg_degree = degree_sum / (FeatureType)num_vertices; 
  FeatureType standard_deviation_degree = 0;
  for (int i = 0; i < num_vertices; i++) {
    standard_deviation_degree += (rows[i + 1] - rows[i] - avg_degree)*(rows[i + 1] - rows[i] - avg_degree);
  }
  return new FeatureType(sqrt(standard_deviation_degree)/avg_degree);
}

#if !defined(_HEADER_ONLY)
#include "init/coefficient_of_variation_degree.inc"
#endif
}  // namespace sparsebase::feature
