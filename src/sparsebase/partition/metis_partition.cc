#include "sparsebase/partition/partitioner.h"
#include "sparsebase/partition/metis_partition.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/logger.h"

#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif
namespace sparsebase::partition {
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
#if !defined(_HEADER_ONLY)
#include "init/metis_partition.inc"
#endif
}
