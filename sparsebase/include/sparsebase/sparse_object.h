#ifndef _SPARSEOBJECT_HPP
#define _SPARSEOBJECT_HPP

#include "sparse_format.h"
#include "sparse_reader.h"

namespace sparsebase {

class SparseObject {
public:
  virtual ~SparseObject();
  virtual void VerifyStructure() = 0;
};

template <typename IDType, typename NNZType, typename ValueType>
class AbstractSparseObject : public SparseObject {
protected:
  SparseFormat<IDType, NNZType, ValueType> *connectivity_;

public:
  virtual ~AbstractSparseObject();
  SparseFormat<IDType, NNZType, ValueType> *get_connectivity();
};

template <typename VertexID, typename NumEdges, typename Weight>
class Graph : public AbstractSparseObject<VertexID, NumEdges, Weight> {
public:
  Graph(SparseFormat<VertexID, NumEdges, Weight> *connectivity);
  Graph();
  void ReadConnectivityToCSR(const ReadsCSR<VertexID, NumEdges, Weight> &);
  void ReadConnectivityToCOO(const ReadsCOO<VertexID, NumEdges, Weight> &);
  void ReadConnectivityFromMTXToCOO(std::string filename);
  void ReadConnectivityFromEdgelistToCSR(std::string filename);
  void InitializeInfoFromConnection();
  virtual ~Graph();
  void VerifyStructure();
  VertexID n_;
  NumEdges m_;
};

// template<typename VertexID, typename NumEdges, typename t_t>
// class TemporalGraph : public AbstractSparseObject<VertexID, NumEdges>{
//   public:
//     TemporalGraph(SparseFormat<VertexID, NumEdges, t_t> * _connectivity){
//       // init temporal graph
//     }
//     TemporalGraph(SparseReader<VertexID, NumEdges, t_t> * r){
//       // init temporal graph from file
//     }
//     virtual ~TemporalGraph(){};
//     void VerifyStructure(){
//       // check order
//       if (this->connectivity->get_order() != 2) //throw error
//       // check dimensions
//     }
//     VertexID n;
//     NumEdges m;
//     // ...
// };

} // namespace sparsebase

#endif
