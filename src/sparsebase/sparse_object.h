#ifndef _SPARSEOBJECT_HPP
#define _SPARSEOBJECT_HPP

#include "config.h"
#include "sparse_format.h"
#include "sparse_reader.h"
#include <functional>
#include <memory>

namespace sparsebase {

namespace object {

class Object {
public:
  virtual ~Object();
  virtual void VerifyStructure() = 0;
};

template <typename IDType, typename NNZType, typename ValueType>
class AbstractObject : public Object {
protected:
  std::unique_ptr<format::Format, std::function<void(format::Format *)>> connectivity_;

public:
  virtual ~AbstractObject();
  AbstractObject();
  AbstractObject(const AbstractObject<IDType, NNZType, ValueType> &);
  AbstractObject(AbstractObject<IDType, NNZType, ValueType> &&);
  format::Format *get_connectivity() const;
  format::Format *release_connectivity();
  void set_connectivity(format::Format *, bool);
  bool ConnectivityIsOwned() const;
};

template <typename VertexID, typename NumEdges, typename Weight>
class Graph : public AbstractObject<VertexID, NumEdges, Weight> {
public:
  Graph(format::Format *connectivity);
  Graph();
  Graph(const Graph<VertexID, NumEdges, Weight> &);
  Graph(Graph<VertexID, NumEdges, Weight> &&);
  Graph<VertexID, NumEdges, Weight> &
  operator=(const Graph<VertexID, NumEdges, Weight> &);
  void ReadConnectivityToCSR(const utils::ReadsCSR<VertexID, NumEdges, Weight> &);
  void ReadConnectivityToCOO(const utils::ReadsCOO<VertexID, NumEdges, Weight> &);
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
//     TemporalGraph(Reader<VertexID, NumEdges, t_t> * r){
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

} // namespace object

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparse_object.cc"
#endif
#endif
