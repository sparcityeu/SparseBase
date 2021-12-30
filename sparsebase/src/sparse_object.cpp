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
  this->verify_structure();
  initialize_info_from_connection();
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::read_connectivity_to_coo(
    const ReadsCOO<VertexID, NumEdges, Weight> &reader) {
  this->connectivity_ = reader.read_coo();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::read_connectivity_to_csr(
    const ReadsCSR<VertexID, NumEdges, Weight> &reader) {
  this->connectivity_ = reader.read_csr();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::read_connectivity_from_edgelist_to_csr(
    std::string filename) {
  UedgelistReader<VertexID, NumEdges, Weight> reader(filename);
  this->connectivity_ = reader.read_csr();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::read_connectivity_from_mtx_to_coo(std::string filename) {
  MTXReader<VertexID, NumEdges, Weight> reader(filename);
  this->connectivity_ = reader.read_coo();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph() {}
template <typename VertexID, typename NumEdges, typename Value>
void Graph<VertexID, NumEdges, Value>::initialize_info_from_connection() {
  auto dimensions = this->connectivity_->get_dimensions();
  n_ = dimensions[0];
  m_ = this->connectivity_->get_num_nnz();
}
template <typename VertexID, typename NumEdges, typename Value>
Graph<VertexID, NumEdges, Value>::~Graph(){};
template <typename VertexID, typename NumEdges, typename Value>
void Graph<VertexID, NumEdges, Value>::verify_structure() {
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