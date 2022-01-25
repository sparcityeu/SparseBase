#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_object.h"
#include "sparsebase/sparse_reader.h"

namespace sparsebase {

template <typename T>
struct BlankDeleter{
  void operator()(T*){

  }
};
template <typename T>
struct Deleter{
  void operator()(T*ptr){
    if (ptr!=nullptr) delete ptr;
  }
};
SparseObject::~SparseObject(){};

template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseObject<IDType, NNZType, ValueType>::~AbstractSparseObject(){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseObject<IDType, NNZType, ValueType>::AbstractSparseObject(): connectivity_(nullptr, BlankDeleter<SparseFormat<IDType, NNZType, ValueType>>()){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseObject<IDType, NNZType, ValueType>::AbstractSparseObject(AbstractSparseObject<IDType, NNZType, ValueType>&& rhs): connectivity_(std::move(rhs.connectivity_)){};
//template <typename IDType, typename NNZType, typename ValueType>
//AbstractSparseObject<IDType, NNZType, ValueType>::AbstractSparseObject(const AbstractSparseObject<IDType, NNZType, ValueType>& rhs): connectivity_(new SparseFormat<IDType, NNZType, ValueType>(rhs.connectivity_.get()), BlankDeleter<SparseFormat<IDType, NNZType, ValueType>>()){};
template <typename IDType, typename NNZType, typename ValueType>
SparseFormat<IDType, NNZType, ValueType> *
AbstractSparseObject<IDType, NNZType, ValueType>::get_connectivity() const {
  return connectivity_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
SparseFormat<IDType, NNZType, ValueType> *
AbstractSparseObject<IDType, NNZType, ValueType>::release_connectivity() {
  auto ptr = connectivity_.release();
  connectivity_ = std::unique_ptr<SparseFormat<IDType, NNZType, ValueType>, std::function<void (SparseFormat<IDType, NNZType, ValueType>*)>>(ptr, BlankDeleter<SparseFormat<IDType, NNZType, ValueType>>());
  return ptr;
}
template <typename IDType, typename NNZType, typename ValueType>
void AbstractSparseObject<IDType, NNZType, ValueType>::set_connectivity(SparseFormat<IDType, NNZType, ValueType>*conn, bool own) {

  if (own)
    connectivity_ = std::unique_ptr<SparseFormat<IDType, NNZType, ValueType>, std::function<void (SparseFormat<IDType, NNZType, ValueType>*)>>(conn, Deleter<SparseFormat<IDType, NNZType, ValueType>>());
  else
    connectivity_ = std::unique_ptr<SparseFormat<IDType, NNZType, ValueType>, std::function<void (SparseFormat<IDType, NNZType, ValueType>*)>>(conn, BlankDeleter<SparseFormat<IDType, NNZType, ValueType>>());
}

template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph(SparseFormat<VertexID, NumEdges, Weight> *connectivity) {
  //this->connectivity_ = connectivity;
  this->set_connectivity(connectivity, true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCOO(
    const ReadsCOO<VertexID, NumEdges, Weight> &reader) {
  this->set_connectivity(reader.ReadCOO(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCSR(
    const ReadsCSR<VertexID, NumEdges, Weight> &reader) {
  this->set_connectivity(reader.ReadCSR(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromEdgelistToCSR(
    std::string filename) {
  UedgelistReader<VertexID, NumEdges, Weight> reader(filename);
  this->set_connectivity(reader.ReadCSR(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromMTXToCOO(std::string filename) {
  MTXReader<VertexID, NumEdges, Weight> reader(filename);
  this->set_connectivity(reader.ReadCOO(), true);
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