#include "sparsebase/feature/max_degree.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
MaxDegree<IDType, NNZType, ValueType>::MaxDegree(ParamsType) {
  MaxDegree();
}
template <typename IDType, typename NNZType, typename ValueType>
MaxDegree<IDType, NNZType, ValueType>::MaxDegree() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
MaxDegree<IDType, NNZType, ValueType>::MaxDegree(
    const MaxDegree<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
MaxDegree<IDType, NNZType, ValueType>::MaxDegree(
    const std::shared_ptr<ParamsType> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
MaxDegree<IDType, NNZType, ValueType>::~MaxDegree() = default;

template <typename IDType, typename NNZType, typename ValueType>
void MaxDegree<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetMaxDegreeCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
MaxDegree<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(MaxDegree<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
MaxDegree<IDType, NNZType, ValueType>::get_subs() {
  return {new MaxDegree<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index MaxDegree<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(MaxDegree<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
MaxDegree<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                             std::vector<context::Context *> c,
                                             bool convert_input) {
  return {{this->get_id(),
           std::forward<NNZType *>(GetMaxDegree(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MaxDegree<IDType, NNZType, ValueType>::GetMaxDegree(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, NNZType *>
MaxDegree<IDType, NNZType, ValueType>::GetMaxDegreeCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MaxDegree<IDType, NNZType, ValueType>::GetMaxDegreeCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType num_vertices =  csr->get_dimensions()[0];
  auto *rows = csr->get_row_ptr();
  NNZType *max_degree = new NNZType;
  *max_degree = rows[1] - rows[0];
  for (int i = 1; i < num_vertices; i++) {
    *max_degree = std::max(*max_degree, rows[i + 1] - rows[i]);
  }
  return max_degree;
}

#if !defined(_HEADER_ONLY)
#include "init/max_degree.inc"
#endif
}  // namespace sparsebase::feature