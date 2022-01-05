#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_object.h"
#include "sparsebase/sparse_reader.h"

namespace sparsebase {

SparseObject::~SparseObject(){};

template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseObject<IDType, NNZType, ValueType>::~AbstractSparseObject(){};
template <typename IDType, typename NNZType, typename ValueType>
SparseFormat<IDType, NNZType, ValueType> *
AbstractSparseObject<IDType, NNZType, ValueType>::get_connectivity() {
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
template <typename VertexID, typename NumEdges, typename ValueType>
void Graph<VertexID, NumEdges, ValueType>::InitializeInfoFromConnection() {
  auto dimensions = this->connectivity_->get_dimensions();
  n_ = dimensions[0];
  m_ = this->connectivity_->get_num_nnz();
}
template <typename VertexID, typename NumEdges, typename ValueType>
Graph<VertexID, NumEdges, ValueType>::~Graph(){};
template <typename VertexID, typename NumEdges, typename ValueType>
void Graph<VertexID, NumEdges, ValueType>::VerifyStructure() {
  // check order
  if (this->connectivity_->get_order() != 2)
    throw -1;
  // check dimensions
}

#ifdef NDEBUG
#include "init/sparse_object.inc"
#else
template class AbstractSparseObject<unsigned int, unsigned int, unsigned int>;
template class AbstractSparseObject<unsigned int, unsigned int, void>;
template class Graph<unsigned int, unsigned int, unsigned int>;
template class Graph<unsigned int, unsigned int, void>;
#endif
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