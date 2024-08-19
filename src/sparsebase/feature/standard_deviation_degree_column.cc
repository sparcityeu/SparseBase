#include "sparsebase/feature/standard_deviation_degree_column.h"

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
StandardDeviationDegreeColumn<IDType, NNZType, ValueType,
                        FeatureType>::StandardDeviationDegreeColumn() {
  Register();
  this->params_ =
      std::shared_ptr<StandardDeviationDegreeColumnParams>(new StandardDeviationDegreeColumnParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::StandardDeviationDegreeColumn(
    StandardDeviationDegreeColumnParams params) {
  StandardDeviationDegreeColumn();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::StandardDeviationDegreeColumn(
    const StandardDeviationDegreeColumn &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::StandardDeviationDegreeColumn(
    const std::shared_ptr<StandardDeviationDegreeColumnParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSC<IDType, NNZType, ValueType>::get_id_static()},
      GetStandardDeviationDegreeColumnCSC);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetStandardDeviationDegreeColumn(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType,
                        FeatureType>::~StandardDeviationDegreeColumn() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetStandardDeviationDegreeColumnCached(format::Format *format,
                                     std::vector<context::Context *> contexts,
                                     bool convert_input) {
  StandardDeviationDegreeColumnParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetStandardDeviationDegreeColumn(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  StandardDeviationDegreeColumnParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::GetStandardDeviationDegreeColumn(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *StandardDeviationDegreeColumn<IDType, NNZType, ValueType, FeatureType>::
    GetStandardDeviationDegreeColumnCSC(std::vector<format::Format *> formats,
                                  utils::Parameters *params) {
  auto CSC = formats[0]->AsAbsolute<format::CSC<IDType, NNZType, ValueType>>();
  IDType num_cols = CSC->get_dimensions()[0];
  FeatureType avg_degree;
  auto *cols = CSC->get_col_ptr();
  NNZType degree_sum = cols[num_cols] - cols[0];
  avg_degree = degree_sum / (FeatureType)num_cols;
  FeatureType standard_deviation_degree = 0;
  for (int i = 0; i < num_cols; i++) {
    standard_deviation_degree += (cols[i + 1] - cols[i] - avg_degree)*(cols[i + 1] - cols[i] - avg_degree);
  }
  return new FeatureType(sqrt(standard_deviation_degree));
}

#if !defined(_HEADER_ONLY)
#include "init/standard_deviation_degree_column.inc"
#endif
}  // namespace sparsebase::feature

