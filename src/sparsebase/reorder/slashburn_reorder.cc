#include "sparsebase/reorder/slashburn_reorder.h"

#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/logger.h"

#include "sparsebase/format/csr.h"

#include <queue>
#include <stack>

bool GREEDY;
bool HUB_ORDER;

/*SQ*/
/* 1ºst imp - greedy k-hub selection, update degree of nodes */
/* after removing one of the nodes.                          */
/* 2ºnd imp - order spokes by hube order and not by size     */

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
SlashburnReorder<IDType, NNZType, ValueType>::SlashburnReorder(int k_size, bool greedy, bool hub_order) {
  auto params_struct = new SlashburnReorderParams;
  params_struct->k_size = k_size;
  params_struct->greedy = greedy;
  params_struct->hub_order = hub_order;
  this->params_ = std::unique_ptr<SlashburnReorderParams>(params_struct);

  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCSR);
}
template <typename IDType, typename NNZType, typename ValueType>
SlashburnReorder<IDType, NNZType, ValueType>::SlashburnReorder(SlashburnReorderParams p)
    : SlashburnReorder(p.k_size, p.greedy, p.hub_order) {}
 /*{
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCSR);
}*/

template <typename IDType, typename NNZType, typename ValueType>
IDType *SlashburnReorder<IDType, NNZType, ValueType>::computeDegree(NNZType *rptr,
                                                              IDType *col,
                                                              IDType n, 
                                                              IDType *v_flag,
                                                              IDType level) {
  
  IDType *degree = new IDType[n];

  /* Compute the degree of all nodes in the GCC */
  for (IDType i = 0; i < n; i++) {
    degree[i] = 0;
    if (v_flag[i] == level) {
      for (IDType ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
        IDType node_id = col[ptr];
        if (v_flag[node_id] == level) degree[i]++;
      }
    } else {
      degree[i] = -1;
    }
  }
  return degree;
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *SlashburnReorder<IDType, NNZType, ValueType>::removeKHubsetGreedy(NNZType *rptr,
                                                              IDType *col,
                                                              IDType n, IDType k, 
                                                              IDType *v_flag,
                                                              IDType *order,
                                                              IDType *degree,
                                                              IDType level,
                                                              IDType min_id) {
  std::priority_queue<std::pair<IDType, IDType>, std::vector<std::pair<IDType, IDType>>, std::greater<std::pair<IDType, IDType>>> PQ;
  IDType *k_hub = new IDType[k];

  /* Removes the highest degree node k times*/
  for (IDType i = 0; i < k; i++) {
    /* Finds node of highest degree */
    PQ.push(std::make_pair(degree[0], 0));
    for (IDType i = 0; i < n; i++) {
      if (v_flag[i] == level) {
        if (degree[i] > PQ.top().first) {
            PQ.pop();
            PQ.push(std::make_pair(degree[i], i));
        }
      }
    }
    
    /* Remove highest degree node from the graph (flag = 0) */
    IDType u = PQ.top().second;
    order[min_id + i] = u;
    v_flag[u] = 0;
    k_hub[i] = u;
    degree[u] = -1;
    PQ.pop();
    
    /* Update degree of nodes */
    for (IDType ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
      IDType node_id = col[ptr];
      if (v_flag[node_id] == level) degree[node_id]--;
    }
  }

  return k_hub;
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *SlashburnReorder<IDType, NNZType, ValueType>::removeKHubset(NNZType *rptr,
                                                              IDType *col,
                                                              IDType n, IDType k, 
                                                              IDType *v_flag,
                                                              IDType *order,
                                                              IDType level,
                                                              IDType min_id) {
  std::priority_queue<std::pair<IDType, IDType>, std::vector<std::pair<IDType, IDType>>, std::greater<std::pair<IDType, IDType>>> PQ;
  IDType *k_hub = new IDType[k];
  IDType i = 0, j = 0, qwp1 = 0;

  /* Place first k nodes in the stack */
  for (i = 0; i < n; i++) {
    if (v_flag[i] == level) {
      IDType degree = 0;
      for (IDType ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
        IDType node_id = col[ptr];
        if (v_flag[node_id] == level) degree++;
      }
      PQ.push(std::make_pair(degree, i));
      j++;
    }
    if (j == k) break;
  }
  /* Check rest of the nodes to find high degree ones */
  for (i = i+1; i < n; i++) {
    if (v_flag[i] == level) {
      IDType degree = 0;
      for (IDType ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
        IDType node_id = col[ptr];
        if (v_flag[node_id] == level) degree++;
      }
      if (degree > PQ.top().first) {
          PQ.pop();
          PQ.push(std::make_pair(degree, i));
      }
    }
  }
  /* Remove them from the graph (flag = 0) */
  while (!PQ.empty()) {
    IDType node_id = PQ.top().second;
    order[min_id + k - 1 - qwp1] = node_id;
    v_flag[node_id] = 0;
    k_hub[k - 1 - qwp1] = node_id;
    qwp1++;
    ;
    PQ.pop();
  }
  return k_hub;
}



template <typename IDType, typename NNZType, typename ValueType>
IDType SlashburnReorder<IDType, NNZType, ValueType>::findCC(NNZType *rptr,
                                                        IDType *col,
                                                        IDType *v_flag,
                                                        IDType level,
                                                        IDType root) {
  IDType cc_count = 1;
  std::stack<IDType> DFS;

  DFS.push(root);
  v_flag[root] = level + 1;
  
  /* Goes through all the nodes in the connected component */
  while (!DFS.empty()) {
    IDType u = DFS.top();
    DFS.pop();
    
    for (IDType ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
      IDType node_id = col[ptr];
      if (v_flag[node_id] == level) {
        DFS.push(node_id);
        v_flag[node_id] = level + 1;
        cc_count++;
      } 
    }
  }
  return cc_count;
}

template <typename IDType, typename NNZType, typename ValueType>
IDType SlashburnReorder<IDType, NNZType, ValueType>::orderCC(NNZType *rptr,
                                                        IDType *col,
                                                        IDType *v_flag,
                                                        IDType *order,
                                                        IDType level,
                                                        IDType root,
                                                        IDType max_id) {
  IDType qwp2 = 0;
  std::queue<IDType> DFS;

  DFS.push(root);
  order[max_id - qwp2] = root;
  v_flag[root] = -level;
  qwp2++;

  /* Goes through all the nodes in the connected component and orders them */
  while (!DFS.empty()) {
    IDType u = DFS.front();
    DFS.pop();
    for (IDType ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
      IDType node_id = col[ptr];
      if (v_flag[node_id] == level + 1) {
        v_flag[node_id] = -level;
        DFS.push(node_id);
        order[max_id - qwp2] = node_id;
        qwp2++;
      }
    }
  }
  return qwp2;
}

template <typename IDType, typename NNZType, typename ValueType>
void SlashburnReorder<IDType, NNZType, ValueType>::slashloop(NNZType *rptr,
                                                          IDType *col,
                                                          IDType n, IDType k,
                                                          IDType *v_flag,
                                                          IDType *order,
                                                          IDType level,
                                                          IDType max_id) {

  std::priority_queue<std::tuple<IDType, IDType, IDType>, std::vector<std::tuple<IDType, IDType, IDType>>, std::greater<std::tuple<IDType, IDType, IDType>>> PQ_CC_hub;
  IDType cmp_counter = 0;
  IDType *k_hub = NULL;

  while (true) {

    /* Removes k-hubset */
    if (GREEDY) {
      IDType *degree = computeDegree(rptr, col, n, v_flag, level);
      k_hub = removeKHubsetGreedy(rptr, col, n, k, v_flag, order, degree, level, (level-2)*k);
    } else {
      k_hub = removeKHubset(rptr, col, n, k, v_flag, order, level, (level-2)*k);
    }
    IDType gcc_count = 0, gcc_id = -1;
    cmp_counter = 0;

    /* Finds strong connected components */
    for (IDType i = k - 1; i >= 0; i--) {
      IDType u = k_hub[i];
      for (IDType ptr = rptr[u]; ptr < rptr[u + 1]; ptr++) {
        IDType node_id = col[ptr];
        if (v_flag[node_id] == level) {
          IDType n_cc = findCC(rptr, col, v_flag, level, node_id);
          if (n_cc > gcc_count) {
            gcc_count = n_cc;
            gcc_id = node_id;
          }
          if (HUB_ORDER) {
            PQ_CC_hub.push(std::make_tuple(i, n_cc, node_id));
          } else {
            PQ_CC_hub.push(std::make_tuple(0, n_cc, node_id));
          }
          cmp_counter++;
        }
      }
    }
    /* When there are no more nodes left to reorder */
    if (cmp_counter == 0)
      break;
    
    /* Places spokes in the permutation vector */
    for (IDType i = 0; i < cmp_counter; i++) {
      std::tuple<IDType, IDType, IDType> root = PQ_CC_hub.top();
      if (std::get<2>(root) == gcc_id) {
        PQ_CC_hub.pop();
        continue;
      }
      PQ_CC_hub.pop();
      IDType n_cc = orderCC(rptr, col, v_flag, order, level, std::get<2>(root), n-1-max_id);
      max_id += n_cc;
    }

    /* Checks size of GCC */
    if (gcc_count < k) {
      IDType n_cc = orderCC(rptr, col, v_flag, order, level, gcc_id, n-1-max_id);
      break;
    } else {
      level++;
    }
  }
  return;
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *SlashburnReorder<IDType, NNZType, ValueType>::GetReorderCSR(
    std::vector<format::Format *> formats, utils::Parameters *poly_params) {
  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  context::CPUContext *cpu_context =
      static_cast<context::CPUContext *>(csr->get_context());

  std::priority_queue<std::pair<IDType, IDType>, std::vector<std::pair<IDType, IDType>>, std::greater<std::pair<IDType, IDType>>> PQ;
  SlashburnReorderParams *params = static_cast<SlashburnReorderParams *>(poly_params);
  int k = params->k_size;

  if (params->greedy)
    GREEDY = true;
  if (params->hub_order)
    HUB_ORDER = true;
  
  NNZType *rptr = csr->get_row_ptr();
  IDType *col = csr->get_col();
  IDType nodes = csr->get_dimensions()[0];

  NNZType *t_row = new NNZType[nodes + 1];
  IDType *t_count = new IDType[nodes];
  IDType *t_col = new IDType[csr->get_num_nnz()];

  IDType *order = new IDType[nodes];
  IDType *order2 = new IDType[nodes];
  IDType *v_flag = new IDType[nodes];

  IDType cmp_counter = 0, max_id = 0;

  for (IDType i = 0; i < nodes; i++) {
    v_flag[i] = 1;
    t_count[i] = 0;
  }

  /* Create CSC format matrix of the input CSR */
  for (IDType i = 0; i < csr->get_num_nnz(); i++) {
    IDType col_id = col[i];
    t_count[col_id]++;
  }
  
  t_row[0] = 0;
  for (IDType i = 1; i < nodes; i++) {
    t_row[i] = t_row[i-1] + t_count[i-1];
  }
  t_row[nodes] = csr->get_num_nnz();
  
  for (IDType i = 0; i < nodes; i++) {
    for (IDType ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
      IDType node_id = col[ptr];
      t_col[t_row[node_id] + t_count[node_id] - 1] = i;
      t_count[node_id]--;
    }
  }
  
  /* Create the symmetric version of the input matrix */
  NNZType *last_row = new NNZType[nodes + 1];
  IDType *last_col = new IDType[csr->get_num_nnz() * 2];
  IDType *s_flag = new IDType[nodes]();
  IDType last_c = 0;
  
  for (IDType i = 0; i < nodes; i++) {
    for (IDType ptr = rptr[i]; ptr < rptr[i + 1]; ptr++) {
      IDType node_id = col[ptr];
      last_col[last_c] = node_id;
      s_flag[node_id] = i+1;
      last_c++;
    }
    for (IDType ptr = t_row[i]; ptr < t_row[i + 1]; ptr++) {
      IDType node_id = t_col[ptr];
      if (s_flag[node_id] != i+1) {
        last_col[last_c] = node_id;
        last_c++;
      }
    }
    last_row[i + 1] = last_c;
  }

  last_row[0] = 0;

  /* Free the auxiliary CSC format matrix */
  delete[] s_flag;
  delete[] t_row;
  delete[] t_count;
  delete[] t_col;

  /* Orders the spokes in the original graph */
  for (IDType i = 0; i < nodes; i++) {
    if (v_flag[i] == 1) {
      IDType n_cc = findCC(last_row, last_col, v_flag, 1, i);
      PQ.push(std::make_pair(n_cc, i));
      cmp_counter++;
    }
  }
  for (IDType i = 0; i < cmp_counter - 1; i++) {
    IDType root = PQ.top().second;
    PQ.pop();
    IDType n_cc = orderCC(last_row, last_col, v_flag, order, 1, root, nodes-1-max_id);
    max_id += n_cc;
  }

  if (PQ.top().first < k) {
    IDType root = PQ.top().second;
    PQ.pop();
    IDType n_cc = orderCC(last_row, last_col, v_flag, order, 1, root, nodes-1-max_id);
  } else {
    PQ.pop();
    /* Call the slashburn loop algorithm to order the GCC */
    slashloop(last_row, last_col, nodes, k, v_flag, order, 2, max_id);
  }

  /* Convert to the sparsebase permutation format */
  for (IDType i = 0; i < nodes; i++) {
    order2[order[i]] = i;
  }

  /* Free rest of allocated memory */
  delete[] order;
  delete[] v_flag;
  
  return order2;
}

#if !defined(_HEADER_ONLY)
#include "init/slashburn_reorder.inc"
#endif
}  // namespace sparsebase::reorder
