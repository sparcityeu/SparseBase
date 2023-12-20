#include "min_degree_column.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
MinDegreeColumn<IDType, NNZType, ValueType>::MinDegreeColumn(ParamsType) {
  MinDegreeColumn();
}
template <typename IDType, typename NNZType, typename ValueType>
MinDegreeColumn<IDType, NNZType, ValueType>::MinDegreeColumn() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
MinDegreeColumn<IDType, NNZType, ValueType>::MinDegreeColumn(
    const MinDegreeColumn<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
MinDegreeColumn<IDType, NNZType, ValueType>::MinDegreeColumn(
    const std::shared_ptr<ParamsType> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
MinDegreeColumn<IDType, NNZType, ValueType>::~MinDegreeColumn() = default;

template <typename IDType, typename NNZType, typename ValueType>
void MinDegreeColumn<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSC<IDType, NNZType, ValueType>::get_id_static()},
      GetMinDegreeColumnCSC);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
MinDegreeColumn<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(MinDegreeColumn<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
MinDegreeColumn<IDType, NNZType, ValueType>::get_subs() {
  return {new MinDegreeColumn<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index MinDegreeColumn<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(MinDegreeColumn<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
MinDegreeColumn<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                             std::vector<context::Context *> c,
                                             bool convert_input) {
  return {{this->get_id(),
           std::forward<NNZType *>(GetMinDegreeColumn(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MinDegreeColumn<IDType, NNZType, ValueType>::GetMinDegreeColumn(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, NNZType *>
MinDegreeColumn<IDType, NNZType, ValueType>::GetMinDegreeColumnCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MinDegreeColumn<IDType, NNZType, ValueType>::GetMinDegreeColumnCSC(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csc = formats[0]->AsAbsolute<format::CSC<IDType, NNZType, ValueType>>();
  IDType num_col =  csc->get_dimensions()[0];
  auto *cols = csc->get_col_ptr();
  NNZType *min_degree = new NNZType;
  *min_degree = cols[1] - cols[0];
  for (int i = 1; i < num_col; i++) {
    *min_degree = std::min(*min_degree, cols[i + 1] - cols[i]);
  }
  return min_degree;
}

#if !defined(_HEADER_ONLY)
#include "init/min_degree_column.inc"
#endif
}  // namespace sparsebase::feature