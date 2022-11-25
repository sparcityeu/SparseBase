#include "sparsebase/reorder/metis_reorder.h"

#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/logger.h"

namespace sparsebase::reorder {
#ifdef USE_METIS

template <typename IDType, typename NNZType, typename ValueType>
MetisReorder<IDType, NNZType, ValueType>::MetisReorder() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCSR);
  this->params_ = std::make_unique<ParamsType>();
}

template <typename IDType, typename NNZType, typename ValueType>
MetisReorder<IDType, NNZType, ValueType>::MetisReorder(
    MetisReorderParams params) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCSR);
  this->params_ = std::make_unique<ParamsType>(params);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *MetisReorder<IDType, NNZType, ValueType>::GetReorderCSR(
    std::vector<format::Format *> formats,
    sparsebase::utils::Parameters *params) {
  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  auto *mparams = static_cast<MetisReorderParams *>(params);
  auto n = (metis::idx_t)csr->get_dimensions()[0];

  metis::idx_t options[METIS_NOPTIONS];
  options[metis::METIS_OPTION_OBJTYPE] = metis::METIS_OBJTYPE_NODE;
  options[metis::METIS_OPTION_CTYPE] = (metis::idx_t)mparams->ctype;
  options[metis::METIS_OPTION_IPTYPE] = metis::METIS_IPTYPE_NODE;
  options[metis::METIS_OPTION_RTYPE] = (metis::idx_t)mparams->rtype;
  options[metis::METIS_OPTION_NO2HOP] = (metis::idx_t)mparams->no2hop;
  options[metis::METIS_OPTION_NITER] = (metis::idx_t)mparams->niter;
  options[metis::METIS_OPTION_UFACTOR] = (metis::idx_t)mparams->ufactor;
  options[metis::METIS_OPTION_SEED] = (metis::idx_t)mparams->seed;
  options[metis::METIS_OPTION_NUMBERING] = (metis::idx_t)mparams->numbering;
  options[metis::METIS_OPTION_COMPRESS] = (metis::idx_t)mparams->compress;
  options[metis::METIS_OPTION_CCORDER] = (metis::idx_t)mparams->ccorder;
  options[metis::METIS_OPTION_PFACTOR] = (metis::idx_t)mparams->pfactor;
  options[metis::METIS_OPTION_NSEPS] = (metis::idx_t)mparams->nseps;
  options[metis::METIS_OPTION_DBGLVL] = (metis::idx_t)0;

  if constexpr (std::is_same_v<IDType, metis::idx_t> &&
                std::is_same_v<NNZType, metis::idx_t>) {
    auto *perm = new metis::idx_t[n];
    auto *inv_perm = new metis::idx_t[n];
    metis::METIS_NodeND(&n, csr->get_row_ptr(), csr->get_col(), nullptr,
                        options, perm, inv_perm);
    delete[] perm;
    return inv_perm;
  } else {
    throw utils::TypeException("MetisReorder supports only " +
                               std::to_string(sizeof(metis::idx_t) * 8) +
                               "-bit signed integers for ids");
  }
}

#endif
#if !defined(_HEADER_ONLY)
#include "init/metis_reorder.inc"
#endif
}  // namespace sparsebase::reorder
