#ifndef _SPARSEOBJECT_HPP
#define _SPARSEOBJECT_HPP

#include "sparse_format.hpp"
#include "sparse_reader.hpp"

namespace sparsebase {

class SparseObject {
public:
  virtual ~SparseObject();
  virtual void verify_structure() = 0;
};

template <typename ID, typename NumNonZeros, typename Value>
class AbstractSparseObject : public SparseObject {
protected:
  SparseFormat<ID, NumNonZeros, Value> *connectivity_;

public:
  virtual ~AbstractSparseObject();
  SparseFormat<ID, NumNonZeros, Value> *get_connectivity();
};

template <typename VertexID, typename NumEdges, typename Weight>
class Graph : public AbstractSparseObject<VertexID, NumEdges, Weight> {
public:
  Graph(SparseFormat<VertexID, NumEdges, Weight> *connectivity);
  Graph();
  void read_connectivity_to_csr(const ReadsCSR<VertexID, NumEdges, Weight> &);
  void read_connectivity_to_coo(const ReadsCOO<VertexID, NumEdges, Weight> &);
  void read_connectivity_from_mtx_to_coo(std::string filename);
  void read_connectivity_from_edgelist_to_csr(std::string filename);
  void initialize_info_from_connection();
  virtual ~Graph();
  void verify_structure();
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
//     void verify_structure(){
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
