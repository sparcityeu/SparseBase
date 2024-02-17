#include "sparsebase/feature/bandwidth.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
Bandwidth<IDType, NNZType, ValueType>::Bandwidth(ParamsType) {
  Bandwidth();
}
template <typename IDType, typename NNZType, typename ValueType>
Bandwidth<IDType, NNZType, ValueType>::Bandwidth() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
Bandwidth<IDType, NNZType, ValueType>::Bandwidth(
    const Bandwidth<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
Bandwidth<IDType, NNZType, ValueType>::Bandwidth(
    const std::shared_ptr<ParamsType> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
Bandwidth<IDType, NNZType, ValueType>::~Bandwidth() = default;

template <typename IDType, typename NNZType, typename ValueType>
void Bandwidth<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetBandwidthCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
Bandwidth<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(Bandwidth<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
Bandwidth<IDType, NNZType, ValueType>::get_subs() {
  return {new Bandwidth<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index Bandwidth<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(Bandwidth<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
Bandwidth<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                             std::vector<context::Context *> c,
                                             bool convert_input) {
  return {{this->get_id(),
           std::forward<int *>(GetBandwidth(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
int *Bandwidth<IDType, NNZType, ValueType>::GetBandwidth(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, int *>
Bandwidth<IDType, NNZType, ValueType>::GetBandwidthCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
int *Bandwidth<IDType, NNZType, ValueType>::GetBandwidthCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
    auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
    auto dims = csr->get_dimensions();
    IDType num_rows = dims[0]; //??
    NNZType *rows = csr->get_row_ptr();
    IDType *columns = csr->get_col();
    int bandwidth = 0;
    for (NNZType i = 0; i < num_rows; i++) {
        for (NNZType k = rows[i]; k < rows[i + 1]; k++) {
            IDType j = columns[k];
            if (i >= j && bandwidth < i - j + 1)
                bandwidth = i - j + 1;
            else if (i < j && bandwidth < j - i + 1)
                bandwidth = j - i + 1;
        }
    }

    return new int(bandwidth);
}

#if !defined(_HEADER_ONLY)
#include "init/bandwidth.inc"
#endif
}  // namespace sparsebase::feature
