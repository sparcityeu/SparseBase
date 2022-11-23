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


template <typename IDType, typename ValueType>
PermuteOrderOne<IDType, ValueType>::PermuteOrderOne(ParamsType params) {
  PermuteOrderOne(params.order);
}
template <typename IDType, typename ValueType>
PermuteOrderOne<IDType, ValueType>::PermuteOrderOne(IDType *order) {
  this->RegisterFunction({format::Array<ValueType>::get_id_static()},
                         PermuteArray);
  this->params_ = std::make_unique<PermuteOrderOneParams<IDType>>(
      order);
}
template <typename IDType, typename ValueType>
format::FormatOrderOne<ValueType> *
PermuteOrderOne<IDType, ValueType>::PermuteArray(std::vector<format::Format *> formats, utils::Parameters *params) {
  auto *sp = formats[0]->AsAbsolute<format::Array<ValueType>>();
  auto order = static_cast<PermuteOrderOneParams<IDType> *>(params)->order;
  std::vector<format::DimensionType> dimensions = sp->get_dimensions();
  IDType length = dimensions[0];
  ValueType *vals = sp->get_vals();
  ValueType *nvals = new ValueType[length]();
  IDType *inv_order = new IDType[length];
  for (IDType i = 0; i < length; i++) {
    inv_order[order[i]] = i;
  }

  for (IDType i = 0; i < length; i++) {
    nvals[i] = vals[inv_order[i]];
  }
  format::Array<ValueType> *arr = new format::Array<ValueType>(length, nvals, format::kOwned);
  return arr;
}
template <typename IDType, typename NNZType, typename ValueType>
PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwo(
    IDType *row_order, IDType *col_order) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      PermuteOrderTwoCSR);
  this->params_ = std::make_unique<PermuteOrderTwoParams<IDType>>(
      row_order, col_order);
}
template <typename IDType, typename NNZType, typename ValueType>
PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwo(
    PermuteOrderTwoParams<IDType> params) {
  PermuteOrderTwo(params.row_order, params.col_order);
}
template <typename InputFormatType, typename ReturnFormtType>
TransformPreprocessType<InputFormatType,
                        ReturnFormtType>::~TransformPreprocessType() = default;
template <typename IDType, typename NNZType, typename ValueType>
format::FormatOrderTwo<IDType, NNZType, ValueType>
    *PermuteOrderTwo<IDType, NNZType, ValueType>::PermuteOrderTwoCSR(
        std::vector<format::Format *> formats, utils::Parameters *params) {
  auto *sp = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  auto row_order =
      static_cast<PermuteOrderTwoParams<IDType> *>(params)->row_order;
  auto col_order =
      static_cast<PermuteOrderTwoParams<IDType> *>(params)->col_order;
  std::vector<format::DimensionType> dimensions = sp->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = sp->get_num_nnz();
  NNZType *xadj = sp->get_row_ptr();
  IDType *adj = sp->get_col();
  ValueType *vals = sp->get_vals();
  NNZType *nxadj = new NNZType[n + 1]();
  IDType *nadj = new IDType[nnz]();
  ValueType *nvals = nullptr;
  if constexpr (!std::is_same_v<void, ValueType>) {
    if (sp->get_vals() != nullptr) nvals = new ValueType[nnz]();
  }
  std::function<IDType(IDType)> get_i_row_order;
  std::function<IDType(IDType)> get_col_order;
  IDType *inverse_row_order;
  if (row_order != nullptr) {
    inverse_row_order = new IDType[n]();
    for (IDType i = 0; i < n; i++) inverse_row_order[row_order[i]] = i;
    get_i_row_order = [&inverse_row_order](IDType i) -> IDType {
      return inverse_row_order[i];
    };
  } else {
    get_i_row_order = [&inverse_row_order](IDType i) -> IDType { return i; };
  }
  if (col_order != nullptr) {
    get_col_order = [&col_order](IDType i) -> IDType { return col_order[i]; };
  } else {
    get_col_order = [](IDType i) -> IDType { return i; };
  }
  // IDType *inverse_col_order = new IDType[n]();
  // for (IDType i = 0; i < n; i++)
  //  inverse_col_order[col_order[i]] = i;
  NNZType c = 0;
  for (IDType i = 0; i < n; i++) {
    IDType u = get_i_row_order(i);
    nxadj[i + 1] = nxadj[i] + (xadj[u + 1] - xadj[u]);
    for (NNZType v = xadj[u]; v < xadj[u + 1]; v++) {
      nadj[c] = get_col_order(adj[v]);
      if constexpr (!std::is_same_v<void, ValueType>) {
        if (sp->get_vals() != nullptr) nvals[c] = vals[v];
      }
      c++;
    }
  }
  if (row_order == nullptr) delete[] inverse_row_order;
  format::CSR<IDType, NNZType, ValueType> *csr = new format::CSR(n, m, nxadj, nadj, nvals);
  return csr;
}

template <typename InputFormatType, typename ReturnFormatType>
std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
TransformPreprocessType<InputFormatType, ReturnFormatType>::
    GetTransformationCached(format::Format *format,
                            std::vector<context::Context *> contexts,
                            bool convert_input) {
  //if (dynamic_cast<InputFormatType *>(format) == nullptr)
  //  throw utils::TypeException(format->get_name(),
  //                             InputFormatType::get_name_static());
  return this->CachedExecute(this->params_.get(), contexts,
                             convert_input, false, format);
}

template <typename InputFormatType, typename ReturnFormatType>
std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
TransformPreprocessType<InputFormatType, ReturnFormatType>::
    GetTransformationCached(format::Format *format, utils::Parameters *params,
                            std::vector<context::Context *> contexts,
                            bool convert_input) {
  //if (dynamic_cast<InputFormatType *>(format) == nullptr)
  //  throw utils::TypeException(format->get_name(),
  //                             InputFormatType::get_name_static());
  return this->CachedExecute(params, contexts, convert_input,
                             false, format);
}

template <typename InputFormatType, typename ReturnFormatType>
ReturnFormatType *
TransformPreprocessType<InputFormatType, ReturnFormatType>::GetTransformation(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  ////if (dynamic_cast<InputFormatType *>(format) == nullptr)
  ////  throw utils::TypeException(format->get_name(),
  ////                             InputFormatType::get_name_static());
  return this->Execute(this->params_.get(), contexts,
                       convert_input, format);
}

template <typename InputFormatType, typename ReturnFormatType>
ReturnFormatType *
TransformPreprocessType<InputFormatType, ReturnFormatType>::GetTransformation(
    format::Format *format, utils::Parameters *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  ////if (dynamic_cast<InputFormatType *>(format) == nullptr)
  ////  throw utils::TypeException(format->get_name(),
  ////                             InputFormatType::get_name_static());
  return this->Execute(params, contexts, convert_input,
                       format);
}

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


template <typename IDType>
PartitionPreprocessType<IDType>::PartitionPreprocessType() = default;

template <typename IDType>
IDType *PartitionPreprocessType<IDType>::Partition(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  return this->Execute(this->params_.get(), contexts,
                       convert_input, format);
}

template <typename IDType>
IDType *PartitionPreprocessType<IDType>::Partition(
    format::Format *format, utils::Parameters *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->Execute(params, contexts, convert_input,
                       format);
}

template <typename IDType>
PartitionPreprocessType<IDType>::~PartitionPreprocessType() = default;


#ifdef USE_METIS

template <typename IDType, typename NNZType, typename ValueType>
MetisPartition<IDType, NNZType, ValueType>::MetisPartition() {

  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::make_unique<MetisPartitionParams>();
}

template <typename IDType, typename NNZType, typename ValueType>
MetisPartition<IDType, NNZType, ValueType>::MetisPartition(
    MetisPartitionParams params) {

  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::make_unique<MetisPartitionParams>(params);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *MetisPartition<IDType, NNZType, ValueType>::PartitionCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  MetisPartitionParams *mparams = static_cast<MetisPartitionParams *>(params);

  metis::idx_t n = (metis::idx_t)csr->get_dimensions()[0];

  IDType *partition = new IDType[n];

  metis::idx_t options[METIS_NOPTIONS];
  options[metis::METIS_OPTION_OBJTYPE] = (metis::idx_t)mparams->objtype;
  options[metis::METIS_OPTION_CTYPE] = (metis::idx_t)mparams->ctype;
  options[metis::METIS_OPTION_IPTYPE] = (metis::idx_t)mparams->iptype;
  options[metis::METIS_OPTION_RTYPE] = (metis::idx_t)mparams->rtype;
  options[metis::METIS_OPTION_NO2HOP] = (metis::idx_t)mparams->no2hop;
  options[metis::METIS_OPTION_NCUTS] = (metis::idx_t)mparams->ncuts;
  options[metis::METIS_OPTION_NITER] = (metis::idx_t)mparams->niter;
  options[metis::METIS_OPTION_UFACTOR] = (metis::idx_t)mparams->ufactor;
  options[metis::METIS_OPTION_MINCONN] = (metis::idx_t)mparams->minconn;
  options[metis::METIS_OPTION_CONTIG] = (metis::idx_t)mparams->contig;
  options[metis::METIS_OPTION_SEED] = (metis::idx_t)mparams->seed;
  options[metis::METIS_OPTION_NUMBERING] = (metis::idx_t)mparams->numbering;
  options[metis::METIS_OPTION_DBGLVL] = (metis::idx_t)0;


  metis::idx_t np = (metis::idx_t)mparams->num_partitions;
  metis::idx_t nw = (metis::idx_t)1;
  metis::idx_t objval;

  if constexpr (std::is_same_v<IDType, metis::idx_t> && std::is_same_v<NNZType, metis::idx_t>) {
    if (mparams->ptype == metis::METIS_PTYPE_RB) {
      metis::METIS_PartGraphRecursive(&n, &nw, (metis::idx_t *)csr->get_row_ptr(),
                               (metis::idx_t *)csr->get_col(), nullptr, nullptr,
                               nullptr, &np, nullptr, nullptr, options, &objval,
                               partition);

    } else {
      metis::METIS_PartGraphKway(&n, &nw, (metis::idx_t *)csr->get_row_ptr(),
                          (metis::idx_t *)csr->get_col(), nullptr, nullptr, nullptr,
                          &np, nullptr, nullptr, options, &objval, partition);
    }
  } else {
    throw utils::TypeException("Metis Partitioner supports only " +
                               std::to_string(sizeof(metis::idx_t) * 8) +
                               "-bit signed integers for ids");
  }
  return partition;
}


#endif

#ifdef USE_PULP

#include <pulp.h>

template <typename IDType, typename NNZType, typename ValueType>
PulpPartition<IDType, NNZType, ValueType>::PulpPartition() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<PulpPartitionParams>(new PulpPartitionParams);
}

template <typename IDType, typename NNZType, typename ValueType>
PulpPartition<IDType, NNZType, ValueType>::PulpPartition(
    PulpPartitionParams params) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<PulpPartitionParams>(new PulpPartitionParams(params));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *PulpPartition<IDType, NNZType, ValueType>::PartitionCSR(
    std::vector<format::Format *> formats, PreprocessParams *params) {
  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  PulpPartitionParams *pparams = static_cast<PulpPartitionParams *>(params);

  format::DimensionType n = csr->get_dimensions()[0];
  format::DimensionType m = csr->get_num_nnz();

  pulp_part_control_t con;
  con.vert_balance = pparams->vert_balance;
  con.edge_balance = pparams->edge_balance;
  con.pulp_seed = pparams->seed;
  con.do_lp_init = pparams->do_lp_init;
  con.do_bfs_init = pparams->do_bfs_init;
  con.do_repart = pparams->do_repart;
  con.do_edge_balance = pparams->do_edge_balance;
  con.do_maxcut_balance = pparams->do_maxcut_balance;

  int np = pparams->num_partitions;
  IDType* partition = new IDType[n];

  if constexpr (std::is_same_v<IDType, int> && std::is_same_v<NNZType, long>) {
    pulp_graph_t graph;
    graph.n = n;
    graph.m = m;
    graph.out_array = csr->get_col();
    graph.out_degree_list = csr->get_row_ptr();
    graph.vertex_weights = nullptr;
    graph.edge_weights = nullptr;
    graph.vertex_weights_sum = 0;
    pulp_run(&graph, &con, partition, np);
  } else {
    throw utils::TypeException("Pulp Partitioner requires IDType=int, NNZType=long");
  }
  return partition;
}
#endif

#ifdef USE_PATOH

#include <patoh.h>

template <typename IDType, typename NNZType, typename ValueType>
PatohPartition<IDType, NNZType, ValueType>::PatohPartition() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<PatohPartitionParams>(new PatohPartitionParams);
}

template <typename IDType, typename NNZType, typename ValueType>
PatohPartition<IDType, NNZType, ValueType>::PatohPartition(
    PatohPartitionParams params) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<PatohPartitionParams>(new PatohPartitionParams(params));
}


template <typename IDType, typename NNZType, typename ValueType>
IDType *PatohPartition<IDType, NNZType, ValueType>::PartitionCSR(
    std::vector<format::Format *> formats, PreprocessParams *params){

  if constexpr (!(std::is_same_v<IDType, int> && std::is_same_v<NNZType, int>)) {
    throw utils::TypeException("Patoh Partitioner requires IDType=int, NNZType=int");
  }

  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  int* ptrs = (int*) csr->get_row_ptr();
  int* js = (int*) csr->get_col();
  int m = csr->get_dimensions()[0];
  int n = csr->get_dimensions()[1];


  int *xpins, *pins, *cwghts, *nwghts;
  int i, p;

  cwghts = (int *) malloc(sizeof(int) * n);
  memset(cwghts,0,sizeof(int) *n);
  for(i = 0; i < m; i++) {
    for(p = ptrs[i]; p < ptrs[i+1]; p++) {
      cwghts[js[p]]++;
    }
  }

  nwghts = (int *)malloc(sizeof(int) * m);
  for(i = 0; i < m; i++) nwghts[i] = 1;

  xpins = (int *) malloc(sizeof(int) * (m+1));
  memcpy(xpins, ptrs, sizeof(int) * (m+1));

  pins = (int*) malloc(sizeof(int) * xpins[m]);
  for(i = 0; i < m; i++) {
    memcpy(pins + xpins[i], js + ptrs[i], sizeof(int) * (ptrs[i+1] - ptrs[i]));
  }

  PatohPartitionParams *concrete_params = static_cast<PatohPartitionParams *>(params);
  PaToH_Parameters patoh_params;
  PaToH_Initialize_Parameters(&patoh_params, concrete_params->objective, concrete_params->param_init);
  patoh_params._k = concrete_params->num_partitions;
  patoh_params.MemMul_Pins += 3;
  patoh_params.MemMul_CellNet += 3;
  patoh_params.final_imbal = concrete_params->final_imbalance;
  patoh_params.seed = concrete_params->seed;

  auto alloc_res = PaToH_Alloc(&patoh_params, m, n, 1, cwghts, nwghts, xpins, pins);

  if(alloc_res) {
    throw utils::AllocationException();
  }

  int* partition = new int[m];
  int* partwghts = new int[concrete_params->num_partitions];
  int cut = -1;

  PaToH_Part(&patoh_params, m, n, 1, 0, cwghts, nwghts, xpins, pins, nullptr, partition, partwghts, &cut);

  delete[] partwghts;
  free(xpins);
  free(pins);
  free(cwghts);
  free(nwghts);

  return (IDType*) partition;
}
#endif


#if !defined(_HEADER_ONLY)
#include "init/preprocess.inc"
#endif

}  // namespace preprocess

}  // namespace sparsebase
