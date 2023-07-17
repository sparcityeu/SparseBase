#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/feature/off_diag_block_nnz.h"
#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::OffDiagBlockNNZ(ParamsType params) {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType(params.blockrowsize, params.blockcolsize));
  this->pmap_.insert({get_id_static(), this->params_});
}
template <typename IDType, typename NNZType, typename ValueType>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::OffDiagBlockNNZ() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::OffDiagBlockNNZ(
    const OffDiagBlockNNZ<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::OffDiagBlockNNZ(
    const std::shared_ptr<ParamsType> p) {
  Register();
  this->params_ = p;
  this->pmap_[get_id_static()] = p;
}

template <typename IDType, typename NNZType, typename ValueType>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::~OffDiagBlockNNZ() = default;

template <typename IDType, typename NNZType, typename ValueType>
void OffDiagBlockNNZ<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetOffDiagBlockNNZCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(OffDiagBlockNNZ<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::get_subs() {
  return {new OffDiagBlockNNZ<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index OffDiagBlockNNZ<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(OffDiagBlockNNZ<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::Extract(format::Format *format,
                                              std::vector<context::Context *> c,
                                              bool convert_input) {
  return {{this->get_id(),
           std::forward<IDType *>(GetOffDiagBlockNNZ(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
IDType *OffDiagBlockNNZ<IDType, NNZType, ValueType>::GetOffDiagBlockNNZ(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
OffDiagBlockNNZ<IDType, NNZType, ValueType>::GetOffDiagBlockNNZCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *OffDiagBlockNNZ<IDType, NNZType, ValueType>::GetOffDiagBlockNNZCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  OffDiagBlockNNZParams* param = static_cast<OffDiagBlockNNZParams*>(params);
  int h = param->blockrowsize, w = param->blockcolsize;
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType cnt = 0;
  int num_rows = csr->get_dimensions()[0];
  int num_cols = csr->get_dimensions()[1];
  auto row_ptr = csr->get_row_ptr();
  auto col = csr->get_col();
  for (int p = 0; p < h; p++) {
    IDType rowstart = std::min(num_rows, p*(num_rows/h) + std::min(p, num_rows%h));
    IDType rowend = std::min(num_rows, (p+1)*(num_rows/h) + std::min(p+1, num_rows%h));
    IDType colstart = std::min(num_cols, p*(num_cols/w) + std::min(p, num_cols%w));
    IDType colend = std::min(num_cols, (p+1)*(num_cols/w) + std::min(p+1, num_cols%w));
    for (IDType i = rowstart; i < rowend; i++) {
      for (IDType k = row_ptr[i]; k < row_ptr[i+1]; k++) {
        if (col[k] < colstart || col[k] >= colend) ++cnt;
      }
    }
  }
  return new IDType(cnt);
}

#if !defined(_HEADER_ONLY)
#include "init/off_diag_block_nnz.inc"
#endif
}  // namespace sparsebase::feature