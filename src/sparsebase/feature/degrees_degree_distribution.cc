#include "sparsebase/feature/degrees.h"
#include "sparsebase/feature/degree_distribution.h"
#include "sparsebase/feature/degrees_degree_distribution.h"
#include "sparsebase/utils/parameterizable.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
    FeatureType>::Degrees_DegreeDistribution(Params) {
  Degrees_DegreeDistribution();
}
template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
    FeatureType>::Degrees_DegreeDistribution() {
  this->Register();
  // this->RegisterFunction(
  //     {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, GetCSR);
  this->params_ = std::shared_ptr<Params>(new Params());
  this->pmap_.insert({get_id_static(), this->params_});
  std::shared_ptr<utils::Parameters> deg_dist_param(
      new DegreeDistributionParams);
  std::shared_ptr<utils::Parameters> degs_param(new DegreesParams);
  this->pmap_[DegreeDistribution<IDType, NNZType, ValueType,
      FeatureType>::get_id_static()] =
      deg_dist_param;
  this->pmap_[Degrees<IDType, NNZType, ValueType>::get_id_static()] =
      degs_param;
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
void Degrees_DegreeDistribution<IDType, NNZType, ValueType,
    FeatureType>::Register() {
  this->RegisterFunction(
      std::vector<std::type_index>({format::CSR<IDType, NNZType, ValueType>::get_id_static()}), GetCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
Degrees_DegreeDistribution(const Degrees_DegreeDistribution<
IDType, NNZType, ValueType, FeatureType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::
Degrees_DegreeDistribution(const std::shared_ptr<Params> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
Degrees_DegreeDistribution<IDType, NNZType, ValueType,
    FeatureType>::~Degrees_DegreeDistribution() =
default;

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
std::vector<std::type_index> Degrees_DegreeDistribution<
    IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  std::vector<std::type_index> r = {
      typeid(Degrees<IDType, NNZType, ValueType>),
      typeid(DegreeDistribution<IDType, NNZType, ValueType, FeatureType>)};
  std::sort(r.begin(), r.end());
  return r;
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
std::vector<utils::Extractable *> Degrees_DegreeDistribution<
    IDType, NNZType, ValueType, FeatureType>::get_subs() {
  auto *f1 = new Degrees<IDType, NNZType, ValueType>();
  if (this->pmap_.find(
      Degrees<IDType, NNZType, ValueType>::get_id_static()) !=
      this->pmap_.end()) {
    f1->set_params(Degrees<IDType, NNZType, ValueType>::get_id_static(),
                   this->pmap_[Degrees<IDType, NNZType,
                       ValueType>::get_id_static()]);
  }

  auto *f2 = new DegreeDistribution<IDType, NNZType, ValueType, FeatureType>();
  if (this->pmap_.find(
      DegreeDistribution<IDType, NNZType, ValueType,
          FeatureType>::get_id_static()) !=
      this->pmap_.end()) {
    f2->set_params(
        DegreeDistribution<IDType, NNZType, ValueType,
            FeatureType>::get_id_static(),
        this->pmap_[DegreeDistribution<IDType, NNZType, ValueType,
            FeatureType>::get_id_static()]);
  }

  auto ids = this->get_sub_ids();
  if (ids[0] == f1->get_id())
    return {f1, f2};
  else
    return {f2, f1};
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
std::type_index Degrees_DegreeDistribution<
    IDType, NNZType, ValueType, FeatureType>::get_id_static() {
  return typeid(
      Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
std::unordered_map<std::type_index, std::any>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return Get(format, c, convert_input);
};

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
std::unordered_map<std::type_index, std::any>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::Get(
    format::Format *format, std::vector<context::Context *> c, bool convert_input) {
  Params params;
  return this->Execute(this->params_.get(), c, convert_input,
                       format);
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
std::unordered_map<std::type_index, std::any>
Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::GetCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
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
  return {{Degrees<IDType, NNZType, ValueType>::get_id_static(),
              std::forward<IDType *>(degrees)},
          {DegreeDistribution<IDType, NNZType, ValueType,
              FeatureType>::get_id_static(),
              std::forward<FeatureType *>(dist)}};
}

#if !defined(_HEADER_ONLY)
#include "init/degrees_degree_distribution.inc"
#endif
}
