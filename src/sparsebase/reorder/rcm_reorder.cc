#include "sparsebase/reorder/rcm_reorder.h"

#include <queue>

#include "sparsebase/format/csr.h"

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
RCMReorder<IDType, NNZType, ValueType>::RCMReorder() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
RCMReorder<IDType, NNZType, ValueType>::RCMReorder(RCMReorderParams p) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCSR);
}
template <typename IDType, typename NNZType, typename ValueType>
IDType RCMReorder<IDType, NNZType, ValueType>::peripheral(NNZType *xadj,
                                                          IDType *adj, IDType n,
                                                          IDType start,
                                                          SignedID *distance,
                                                          IDType *Q) {
  IDType r = start;
  SignedID rlevel = -1;
  SignedID qlevel = 0;
  SignedID deg = -1, flag = -1;

  /* Repeat with the new found root.                          */
  /* If the distance is the same then return the current root */
  while (rlevel != qlevel) {
    rlevel = qlevel;

    IDType qrp = 0, qwp = 0;
    distance[r] = 0;
    Q[qwp++] = r;

    /* Computes the distance to the root of every node in the connected component */
    while (qrp < qwp) {
      IDType u = Q[qrp++];
      for (NNZType ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
        IDType v = adj[ptr];
        if (distance[v] == (IDType)-1) {
          distance[v] = distance[u] + 1;
          Q[qwp++] = v;
          if (distance[v] > qlevel) {
            qlevel = distance[v];
          }
        }
      }
    }

    /* If the number of levels is the same as the number of nodes in the */
    /* connected component (N-N-N-N-N) then we already have the root     */
    if (qrp == qlevel + 1) {return r;}

    /* Goes through nodes in the connected component.                */
    /* Root will be the node with maximum distance and lowest degree */
    flag = -1;
    if (rlevel != qlevel) {
      for (IDType i = 0; i < qrp; i++) {
        if (qlevel == distance[Q[i]]) {
          if (flag == -1) {
            deg = xadj[Q[i] + 1] - xadj[Q[i]] + 1;
            flag = 0;
          }
          if (xadj[Q[i] + 1] - xadj[Q[i]] < deg) {
            qlevel = distance[Q[i]];
            r = Q[i];
            deg = xadj[Q[i] + 1] - xadj[Q[i]];
          }
        } 
        distance[Q[i]] = -1;
      }
    }
  }
  return r;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *RCMReorder<IDType, NNZType, ValueType>::GetReorderCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  NNZType *xadj = csr->get_row_ptr();
  IDType *adj = csr->get_col();
  IDType n = csr->get_dimensions()[0];
  IDType *Q = new IDType[n];
    
  IDType *Qp = new IDType[n];
  IDType *Qp2 = new IDType[n];
  SignedID *distance = new SignedID[n];
  IDType *V = new IDType[n];
  for (IDType i = 0; i < n; i++) {
    distance[i] = -1;
    V[i] = 0;
  } 
  std::priority_queue<std::pair<IDType, IDType>, std::vector<std::pair<IDType, IDType>>, std::greater<std::pair<IDType, IDType>>> PQ;
  int qrp = 0, qwp = 0, qst = 0;
  IDType reverse = n - 1;

  /* Go through every node in the graph */
  for (IDType i = 0; i < n; i++) {

    /* If the node has not been visited yet then it belongs to a new connected component */
    if (V[i] == 0) {

      /* Connected components that consist of a single node don't need reordering */
      if (xadj[i] == xadj[i + 1]) {
        Q[qwp] = i;
        Qp2[qwp++] = i;
        V[i] = 1;
        continue;
      }

      /* Find approximate peripheral node */
      IDType perv = peripheral(xadj, adj, n, i, distance, Qp);
      qst = qwp;
      V[perv] = 1;
      Q[qwp++] = perv;

      /* BFS search of the connected component */
      while (qrp < qwp) {
        IDType u = Q[qrp++];

        /* Visit all unvisited nodes neighbouring the current node. */
        /* Order based on their degree                              */
        for (IDType ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
          IDType v = adj[ptr];
          if (V[v] == 0) {
            PQ.push(std::make_pair(xadj[v + 1] - xadj[v], v));
            V[v] = 1;
          }
        }

        /* Place the neighbouring nodes in the queue with their new degree order */
        while (!PQ.empty()) {
          Q[qwp++] = PQ.top().second;
          ;
          PQ.pop();
        }
      }

      /* Reverse connected component */
      for (IDType j = qst; j < qst + (qwp-qst)/2; j++) {
        Qp2[j] = Q[qwp-1 - (j - qst)];
        Qp2[qwp-1 - (j - qst)] = Q[j];
      }
      if ((qwp-qst)%2 != 0) {
        Qp2[qst + (qwp-1 - qst)/2] = Q[qst + (qwp-1 - qst)/2];
      }
    }
  }
  
  /* Place it in the form that the transform function takes */
  for (IDType i = 0; i < n; i++) {
    Q[Qp2[i]] = i;
  }

  delete[] Qp;
  delete[] distance;
  delete[] V;
  return Q;
}

#if !defined(_HEADER_ONLY)
#include "init/rcm_reorder.inc"
#endif
}  // namespace sparsebase::reorder