#include "max_degree_column.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
MaxDegreeColumn<IDType, NNZType, ValueType>::MaxDegreeColumn(ParamsType) {
  MaxDegreeColumn();
}
template <typename IDType, typename NNZType, typename ValueType>
MaxDegreeColumn<IDType, NNZType, ValueType>::MaxDegreeColumn() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
MaxDegreeColumn<IDType, NNZType, ValueType>::MaxDegreeColumn(
    const MaxDegreeColumn<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
MaxDegreeColumn<IDType, NNZType, ValueType>::MaxDegreeColumn(
    const std::shared_ptr<ParamsType> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
MaxDegreeColumn<IDType, NNZType, ValueType>::~MaxDegreeColumn() = default;

template <typename IDType, typename NNZType, typename ValueType>
void MaxDegreeColumn<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSC<IDType, NNZType, ValueType>::get_id_static()},
      GetMaxDegreeColumnCSC);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
MaxDegreeColumn<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(MaxDegreeColumn<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
MaxDegreeColumn<IDType, NNZType, ValueType>::get_subs() {
  return {new MaxDegreeColumn<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index MaxDegreeColumn<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(MaxDegreeColumn<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
MaxDegreeColumn<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                             std::vector<context::Context *> c,
                                             bool convert_input) {
  return {{this->get_id(),
           std::forward<NNZType *>(GetMaxDegreeColumn(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MaxDegreeColumn<IDType, NNZType, ValueType>::GetMaxDegreeColumn(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, NNZType *>
MaxDegreeColumn<IDType, NNZType, ValueType>::GetMaxDegreeColumnCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType *MaxDegreeColumn<IDType, NNZType, ValueType>::GetMaxDegreeColumnCSC(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csc = formats[0]->AsAbsolute<format::CSC<IDType, NNZType, ValueType>>();
  IDType num_col =  csc->get_dimensions()[0];
  auto *cols = csc->get_csc_ptr();
  NNZType *max_degree = new NNZType;
  *max_degree = cols[1] - cols[0];
  for (int i = 1; i < num_col; i++) {
    *max_degree = std::max(*max_degree, cols[i + 1] - cols[i]);
  }
  return max_degree;
}

#if !defined(_HEADER_ONLY)
#include "init/max_degree_column.inc"
#endif
}  // namespace sparsebase::feature