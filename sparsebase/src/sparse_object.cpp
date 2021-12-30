#include <iostream>

#include "sparsebase/sparse_format.hpp"
#include "sparsebase/sparse_object.hpp"
#include "sparsebase/sparse_reader.hpp"

namespace sparsebase {

SparseObject::~SparseObject(){};

template <typename ID, typename NumNonZeros, typename Value>
AbstractSparseObject<ID, NumNonZeros, Value>::~AbstractSparseObject(){};
template <typename ID, typename NumNonZeros, typename Value>
SparseFormat<ID, NumNonZeros, Value> *
AbstractSparseObject<ID, NumNonZeros, Value>::get_connectivity() {
  return connectivity_;
}

template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph(SparseFormat<VertexID, NumEdges, Weight> *connectivity) {
  this->connectivity_ = connectivity;
  this->VerifyStructure();
  InitializeInfoFromConnection();
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCOO(
    const ReadsCOO<VertexID, NumEdges, Weight> &reader) {
  this->connectivity_ = reader.ReadCOO();
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCSR(
    const ReadsCSR<VertexID, NumEdges, Weight> &reader) {
  this->connectivity_ = reader.ReadCSR();
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromEdgelistToCSR(
    std::string filename) {
  UedgelistReader<VertexID, NumEdges, Weight> reader(filename);
  this->connectivity_ = reader.ReadCSR();
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromMTXToCOO(std::string filename) {
  MTXReader<VertexID, NumEdges, Weight> reader(filename);
  this->connectivity_ = reader.ReadCOO();
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph() {}
template <typename VertexID, typename NumEdges, typename Value>
void Graph<VertexID, NumEdges, Value>::InitializeInfoFromConnection() {
  auto dimensions = this->connectivity_->get_dimensions();
  n_ = dimensions[0];
  m_ = this->connectivity_->get_num_nnz();
}
template <typename VertexID, typename NumEdges, typename Value>
Graph<VertexID, NumEdges, Value>::~Graph(){};
template <typename VertexID, typename NumEdges, typename Value>
void Graph<VertexID, NumEdges, Value>::VerifyStructure() {
  // check order
  if (this->connectivity_->get_order() != 2)
    throw -1;
  // check dimensions
}

template class AbstractSparseObject<unsigned int, unsigned int, unsigned int>;
template class AbstractSparseObject<unsigned int, unsigned int, void>;
template class Graph<unsigned int, unsigned int, unsigned int>;
template class Graph<unsigned int, unsigned int, void>;
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