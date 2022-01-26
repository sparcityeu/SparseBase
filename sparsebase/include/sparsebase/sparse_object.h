#ifndef _SPARSEOBJECT_HPP
#define _SPARSEOBJECT_HPP

#include <memory>
#include <functional>
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
  std::unique_ptr<SparseFormat<IDType, NNZType, ValueType>, std::function<void (SparseFormat<IDType, NNZType, ValueType>*)>> connectivity_;

public:
  virtual ~AbstractSparseObject();
  AbstractSparseObject();
  AbstractSparseObject(const AbstractSparseObject<IDType, NNZType, ValueType>&);
  AbstractSparseObject(AbstractSparseObject<IDType, NNZType, ValueType>&&);
  SparseFormat<IDType, NNZType, ValueType> *get_connectivity() const;
  SparseFormat<IDType, NNZType, ValueType> *release_connectivity();
  void set_connectivity(SparseFormat<IDType, NNZType, ValueType>*, bool);
  bool ConnectivityIsOwned() const;

};

template <typename VertexID, typename NumEdges, typename Weight>
class Graph : public AbstractSparseObject<VertexID, NumEdges, Weight> {
public:
  Graph(SparseFormat<VertexID, NumEdges, Weight> *connectivity);
  Graph();
  Graph(const Graph<VertexID, NumEdges, Weight>&);
  Graph(Graph<VertexID, NumEdges, Weight>&&);
  Graph<VertexID, NumEdges, Weight>& operator=(const Graph<VertexID, NumEdges, Weight>&);
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
