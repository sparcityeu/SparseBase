#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/feature/profile.h"
#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
Profile<IDType, NNZType, ValueType>::Profile(ParamsType params) {
  Profile();
}
template <typename IDType, typename NNZType, typename ValueType>
Profile<IDType, NNZType, ValueType>::Profile() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
Profile<IDType, NNZType, ValueType>::Profile(
    const Profile<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
Profile<IDType, NNZType, ValueType>::Profile(
    const std::shared_ptr<ParamsType> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType>
Profile<IDType, NNZType, ValueType>::~Profile() = default;

template <typename IDType, typename NNZType, typename ValueType>
void Profile<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetProfileCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
Profile<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(Profile<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
Profile<IDType, NNZType, ValueType>::get_subs() {
  return {new Profile<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index Profile<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(Profile<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
Profile<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                              std::vector<context::Context *> c,
                                              bool convert_input) {
  return {{this->get_id(),
           std::forward<IDType *>(GetProfile(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
IDType *Profile<IDType, NNZType, ValueType>::GetProfile(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
Profile<IDType, NNZType, ValueType>::GetProfileCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *Profile<IDType, NNZType, ValueType>::GetProfileCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  auto row_ptr = csr->get_row_ptr();
  auto col = csr->get_col();
  IDType sum = 0;
  for (int i = 0; i < csr->get_dimensions()[0]; ++i) {
    IDType j = i;
    for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
      if (j > col[k]) j = col[k];
    }
    sum += i-j;
  }
  return new IDType(sum);
}

#if !defined(_HEADER_ONLY)
#include "init/profile.inc"
#endif
}  // namespace sparsebase::feature