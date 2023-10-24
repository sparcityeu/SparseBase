#include "sparsebase/reorder/metis_reorder.h"
#include "sparsebase/reorder/rcm_reorder.h"
#include "sparsebase/reorder/amd_reorder.h"
#include "sparsebase/bases/iobase.h"
#include "sparsebase/bases/reorder_base.h"

#include "sparsebase/reorder/reorderer.h"

using namespace sparsebase;
using namespace bases;

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
  auto *rptr = csr->get_row_ptr();
  auto *col = csr->get_col();

  /* Initialize the METIS options vector */
  metis::idx_t options[METIS_NOPTIONS];
  metis::METIS_SetDefaultOptions(options);
  options[metis::METIS_OPTION_OBJTYPE] = metis::METIS_OBJTYPE_CUT;
  options[metis::METIS_OPTION_CTYPE] = (metis::idx_t)mparams->ctype;
  options[metis::METIS_OPTION_IPTYPE] = metis::METIS_IPTYPE_GROW;
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
  options[metis::METIS_OPTION_NCUTS] = (metis::idx_t)mparams->ncuts;
  options[metis::METIS_OPTION_MINCONN] = (metis::idx_t)0;
  options[metis::METIS_OPTION_CONTIG] = (metis::idx_t)0;

  if constexpr (std::is_same_v<IDType, metis::idx_t> &&
                std::is_same_v<NNZType, metis::idx_t>) {
    auto *perm = new metis::idx_t[n];
    auto *inv_perm = new metis::idx_t[n];
    metis::idx_t k = (metis::idx_t) mparams->nparts;
    metis::idx_t ordering = (metis::idx_t) mparams->ordering;

    sparsebase::context::CPUContext cpu_context;

    /* If only 1 partition, apply RCM or AMD or ND */
    if (k == 1) {
      if (ordering == 0) {
        RCMReorderParams rcm_params = {};
        inv_perm = ReorderBase::Reorder<RCMReorder>(rcm_params, csr, {&cpu_context}, true);
      } else if (ordering == 1) {
        AMDReorderParams amd_params = {};
        inv_perm = ReorderBase::Reorder<AMDReorder>(amd_params, csr, {&cpu_context}, true);
      } else if (ordering == 2) {
        options[metis::METIS_OPTION_OBJTYPE] = metis::METIS_OBJTYPE_NODE;
        options[metis::METIS_OPTION_IPTYPE] = metis::METIS_IPTYPE_NODE;
        metis::METIS_NodeND(&n, csr->get_row_ptr(), csr->get_col(), nullptr, options, perm, inv_perm);
      }
      
      delete[] perm;
      return inv_perm;
    }

    metis::idx_t order_track = 0;
    metis::idx_t objval;
    metis::idx_t ncon = 1;
    auto *part = new metis::idx_t[n];

    /* Partition the graph into k parts */
    int ret = metis::METIS_PartGraphKway(&n, &ncon, csr->get_row_ptr(), csr->get_col(),
      nullptr, nullptr, nullptr, &k, nullptr,
      nullptr, options, &objval, part);

    /* Go through all the k parts and reorder them */
    for (int i = 0; i < k; i++) {
      auto *csr_row = new metis::idx_t[n + 1];
      auto *csr_col = new metis::idx_t[csr->get_num_nnz()];
      auto *flag = new metis::idx_t[n];
      auto *inv_flag = new metis::idx_t[n];
      metis::idx_t nodesP = 0;
      metis::idx_t nonZerosP = 0;
      csr_row[0] = 0;

      /* Flag the nodes belonging to the current partition */
      for (int j = 0; j < n; j++) {
        if (part[j] == i) {
          flag[j] = nodesP;
          inv_flag[nodesP] = j;
          nodesP++;
        } else {
          flag[j] = -1;
        }
      }
      nodesP = 0;

      /* Create CSR format for the nodes in the current partition */
      for (int j = 0; j < n; j++) {
        if (part[j] == i) {
          csr_row[nodesP] = nonZerosP;
          for (metis::idx_t ptr = rptr[j]; ptr < rptr[j + 1]; ptr++) {
            IDType node_id = col[ptr];
            if (part[node_id] == i) {
              csr_col[nonZerosP] = flag[node_id];
              nonZerosP++;
            }
          }
          nodesP++;
        }
      }
      csr_row[nodesP] = nonZerosP;
      sparsebase::format::CSR<NNZType, IDType, IDType> csr_part(nodesP, nodesP, csr_row, csr_col, nullptr);
      sparsebase::format::CSR<NNZType, IDType, IDType> * csr_part_ptr = &csr_part;

      auto *perm_part = new metis::idx_t[nodesP];
      auto *inv_perm_part = new metis::idx_t[nodesP];

      printf("nodes per partition: %d %d\n", nodesP, nonZerosP);

      /* Apply reordering (RCM or AMD or ND) to the partition */
      if (ordering == 0) {
        RCMReorderParams rcm_params = {};
        inv_perm_part = ReorderBase::Reorder<RCMReorder>(rcm_params, csr_part_ptr, {&cpu_context}, true);
      } else if (ordering == 1) {
        AMDReorderParams amd_params = {};
        inv_perm_part = ReorderBase::Reorder<AMDReorder>(amd_params, csr_part_ptr, {&cpu_context}, true);
      } else if (ordering == 2) {
        options[metis::METIS_OPTION_OBJTYPE] = metis::METIS_OBJTYPE_NODE;
        options[metis::METIS_OPTION_IPTYPE] = metis::METIS_IPTYPE_NODE;
        metis::METIS_NodeND(&nodesP, csr_row, csr_col, nullptr, options, perm_part, inv_perm_part);
      }

      /* Insert permutation order of the partition in the permutation vector of the final graph */
      for (int j = 0; j < nodesP; j++)
        perm_part[inv_perm_part[j]] = j;
      for (int j = 0; j < nodesP; j++)
        perm[order_track + j] = inv_flag[perm_part[j]];
      order_track += nodesP;

      delete[] perm_part;
      delete[] inv_perm_part;
    }

    /* Create the inverse permutation vector */
    for (int i = 0; i < n; i++) {
      inv_perm[perm[i]] = i;
    }

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
