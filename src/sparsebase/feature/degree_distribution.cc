#include "sparsebase/feature/degree_distribution.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType,
                   FeatureType>::DegreeDistribution() {
  Register();
  this->params_ =
      std::shared_ptr<DegreeDistributionParams>(new DegreeDistributionParams());
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(
    DegreeDistributionParams params) {
  DegreeDistribution();
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(
    const DegreeDistribution &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::DegreeDistribution(
    const std::shared_ptr<DegreeDistributionParams> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetDegreeDistributionCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<FeatureType *>(
                               GetDistribution(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  return {typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_subs() {
  return {
      new DegreeDistribution<IDType, NNZType, ValueType, FeatureType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
DegreeDistribution<IDType, NNZType, ValueType,
                   FeatureType>::~DegreeDistribution() = default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
    GetDistributionCached(format::Format *format,
                          std::vector<context::Context *> contexts,
                          bool convert_input) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->CachedExecute(&params, contexts, convert_input, false,
                             format);  // func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  DegreeDistributionParams params;
  return this->Execute(&params, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *
DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetDistribution(
    object::Graph<IDType, NNZType, ValueType> *obj,
    std::vector<context::Context *> contexts, bool convert_input) {
  // std::tuple<DegreeDistributionFunction<IDType, NNZType, ValueType,
  // FeatureType>,
  //             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  //     func_formats =
  // DegreeDistributionFunction<IDType, NNZType, ValueType, FeatureType> func =
  // std::get<0>(func_formats); std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> sfs = std::get<1>(func_formats);
  format::Format *format = obj->get_connectivity();
  return this->Execute(this->params_.get(), contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureType *DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
    GetDegreeDistributionCSR(std::vector<format::Format *> formats,
                             utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
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

#if !defined(_HEADER_ONLY)
#include "init/degree_distribution.inc"
#endif
}  // namespace sparsebase::feature
