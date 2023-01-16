#include "sparsebase/feature/min_degree.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
MinDegree<IDType, NNZType, ValueType>::MinDegree(ParamsType) {
  MinDegree();
}
template <typename IDType, typename NNZType, typename ValueType>
MinDegree<IDType, NNZType, ValueType>::MinDegree() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
MinDegree<IDType, NNZType, ValueType>::MinDegree(
    const MinDegree<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
MinDegree<IDType, NNZType, ValueType>::MinDegree(
    const std::shared_ptr<ParamsType> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
MinDegree<IDType, NNZType, ValueType>::~MinDegree() = default;

template <typename IDType, typename NNZType, typename ValueType>
void MinDegree<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetMinDegreeCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
MinDegree<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(MinDegree<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
MinDegree<IDType, NNZType, ValueType>::get_subs() {
  return {new MinDegree<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index MinDegree<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(MinDegree<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
MinDegree<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                             std::vector<context::Context *> c,
                                             bool convert_input) {
  return {{this->get_id(),
           std::forward<NNZType *>(GetMinDegree(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MinDegree<IDType, NNZType, ValueType>::GetMinDegree(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, NNZType *>
MinDegree<IDType, NNZType, ValueType>::GetMinDegreeCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MinDegree<IDType, NNZType, ValueType>::GetMinDegreeCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType num_vertices =  csr->get_dimensions()[0];
  auto *rows = csr->get_row_ptr();
  NNZType *min_degree = new NNZType;
  *min_degree = rows[1] - rows[0];
  for (int i = 1; i < num_vertices; i++) {
    *min_degree = std::min(*min_degree, rows[i + 1] - rows[i]);
  }
  return min_degree;
}

#if !defined(_HEADER_ONLY)
#include "init/min_degree.inc"
#endif
}  // namespace sparsebase::feature