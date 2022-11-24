#include "sparsebase/feature/degrees.h"
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

template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees(ParamsType) {
  Degrees();
}
template <typename IDType, typename NNZType, typename ValueType>
Degrees<IDType, NNZType, ValueType>::Degrees() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
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
    const std::shared_ptr<ParamsType> r) {
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

#if !defined(_HEADER_ONLY)
#include "init/degrees.inc"
#endif
}