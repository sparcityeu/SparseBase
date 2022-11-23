#include "preprocess.h"

#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/coo.h"

#include "sparsebase/converter/converter.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#include "sparsebase/utils/logger.h"
#ifdef USE_CUDA
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#endif
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

#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif
namespace sparsebase {

namespace preprocess {



template <typename FeatureType>
FeaturePreprocessType<FeatureType>::~FeaturePreprocessType() = default;

template <typename FeatureType>
std::shared_ptr<utils::Parameters>
FeaturePreprocessType<FeatureType>::get_params() {
  return this->params_;
}
template <typename FeatureType>
std::shared_ptr<utils::Parameters>
FeaturePreprocessType<FeatureType>::get_params(std::type_index t) {
  if (this->pmap_.find(t) != this->pmap_.end()) {
    return this->pmap_[t];
  } else {
    throw utils::FeatureParamsException(get_id().name(), t.name());
  }
}
template <typename FeatureType>
void FeaturePreprocessType<FeatureType>::set_params(
    std::type_index t, std::shared_ptr<utils::Parameters> p) {
  auto ids = this->get_sub_ids();
  if (std::find(ids.begin(), ids.end(), t) != ids.end()) {
    this->pmap_[t] = p;
  } else {
    throw utils::FeatureParamsException(get_id().name(), t.name());
  }
}
template <typename FeatureType>
std::type_index FeaturePreprocessType<FeatureType>::get_id() {
  return typeid(*this);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights(
    ParamsType) {}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights() {
#ifdef USE_CUDA
  std::vector<std::type_index> formats = {
      format::CUDACSR<IDType, NNZType,
                            ValueType>::get_id_static()};
  this->RegisterFunction(formats, GetJaccardWeightCUDACSR);
#endif
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::~JaccardWeights(){};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::Format *
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::GetJaccardWeights(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  return this->Execute(nullptr, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}
#ifdef USE_CUDA
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::Format *preprocess::JaccardWeights<
    IDType, NNZType, ValueType,
    FeatureType>::GetJaccardWeightCUDACSR(std::vector<format::Format *> formats,
                                          utils::Parameters *params) {
  auto cuda_csr =
      formats[0]
          ->AsAbsolute<format::CUDACSR<IDType, NNZType, ValueType>>();
  return preprocess::cuda::RunJaccardKernel<IDType, NNZType, ValueType,
                                            FeatureType>(cuda_csr);
}
#endif

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
  return {{this->get_id(), std::forward<FeatureType *>(GetDistribution(
                                       format, c, convert_input))}};
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
std::type_index DegreeDistribution<IDType, NNZType, ValueType,
                                   FeatureType>::get_id_static() {
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
  return this->CachedExecute(&params, contexts,
                             convert_input, false,
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
  return this->Execute(this->params_.get(), contexts,
                       convert_input,
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

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(DegreesParams) {
  Degrees();
}
template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees() {
  Register();
  this->params_ = std::shared_ptr<DegreesParams>(new DegreesParams());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(
    const Degrees<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(
    const std::shared_ptr<DegreesParams> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::~Degrees() = default;

template <typename IDType, typename NNZType, typename ValueType>
void Degrees<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, GetDegreesCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
Degrees<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(Degrees<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *> Degrees<IDType, NNZType, ValueType>::get_subs() {
  return {new Degrees<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index Degrees<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(Degrees<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
Degrees<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                             std::vector<context::Context *> c,
                                             bool convert_input) {
  return {{this->get_id(),
           std::forward<IDType *>(GetDegrees(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
IDType *Degrees<IDType, NNZType, ValueType>::GetDegrees(
    format::Format *format, std::vector<context::Context *> c, bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input,
                       format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
Degrees<IDType, NNZType, ValueType>::GetDegreesCached(
    format::Format *format, std::vector<context::Context *> c, bool convert_input) {
  return this->CachedExecute(this->params_.get(), c,
                             convert_input, false, format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *Degrees<IDType, NNZType, ValueType>::GetDegreesCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  auto dims = csr->get_dimensions();
  IDType num_vertices = dims[0];
  NNZType num_edges = csr->get_num_nnz();
  IDType *degrees = new IDType[num_vertices]();
  auto *rows = csr->get_row_ptr();
  for (int i = 0; i < num_vertices; i++) {
    degrees[i] = rows[i + 1] - rows[i];
  }
  return degrees;
}

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
#include "init/preprocess.inc"
#endif

}  // namespace preprocess

}  // namespace sparsebase
