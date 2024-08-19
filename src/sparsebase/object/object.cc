#include "object.h"

#include <iostream>

#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/io/edge_list_reader.h"
#include "sparsebase/io/mtx_reader.h"

namespace sparsebase::object {

Object::~Object(){};

template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::~AbstractObject(){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::AbstractObject()
    : connectivity_(nullptr, format::BlankDeleter<format::Format>()){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::AbstractObject(
    AbstractObject<IDType, NNZType, ValueType> &&rhs)
    : connectivity_(std::move(rhs.connectivity_)){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::AbstractObject(
    const AbstractObject<IDType, NNZType, ValueType> &rhs)
    : connectivity_((format::Format *)rhs.connectivity_->Clone(),
                    format::BlankDeleter<format::Format>()){};
template <typename IDType, typename NNZType, typename ValueType>
format::Format *AbstractObject<IDType, NNZType, ValueType>::get_connectivity()
    const {
  return connectivity_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
bool AbstractObject<IDType, NNZType, ValueType>::ConnectivityIsOwned() const {
  return (connectivity_.get_deleter().target_type() !=
          typeid(format::BlankDeleter<format::Format>));
}
template <typename IDType, typename NNZType, typename ValueType>
format::Format *
AbstractObject<IDType, NNZType, ValueType>::release_connectivity() {
  auto ptr = connectivity_.release();
  connectivity_ =
      std::unique_ptr<format::Format, std::function<void(format::Format *)>>(
          ptr, format::BlankDeleter<format::Format>());
  return ptr;
}
template <typename IDType, typename NNZType, typename ValueType>
void AbstractObject<IDType, NNZType, ValueType>::set_connectivity(
    format::Format *conn, bool own) {
  if (own)
    connectivity_ =
        std::unique_ptr<format::Format, std::function<void(format::Format *)>>(
            conn, format::Deleter<format::Format>());
  else
    connectivity_ =
        std::unique_ptr<format::Format, std::function<void(format::Format *)>>(
            conn, format::BlankDeleter<format::Format>());
}

template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph(
    Graph<VertexID, NumEdges, Weight> &&rhs) {
  this->set_connectivity(rhs.release_connectivity(), true);
  InitializeInfoFromConnection();
  rhs.set_connectivity(nullptr, false);
}
template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph(
    const Graph<VertexID, NumEdges, Weight> &rhs) {
  this->set_connectivity(
      static_cast<format::Format *>(rhs.connectivity_->Clone()), true);
  InitializeInfoFromConnection();
}
template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight> &Graph<VertexID, NumEdges, Weight>::operator=(
    const Graph<VertexID, NumEdges, Weight> &rhs) {
  this->set_connectivity(
      static_cast<format::Format *>(rhs.connectivity_->Clone()), true);
  InitializeInfoFromConnection();
  return *this;
}
template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph(format::Format *connectivity) {
  // this->connectivity_ = connectivity;
  this->set_connectivity(connectivity, true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
}

template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph(format::Format *connectivity, NumEdges ncon, format::Array<Weight>** vertexWeights) {
  this->set_connectivity(connectivity, true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  this->ncon_ = ncon;
  this->vertexWeights_ = vertexWeights;
}

template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCOO(
    const io::ReadsCOO<VertexID, NumEdges, Weight> &reader) {
  this->set_connectivity(reader.ReadCOO(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  // std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] <<
  // ", "
  //          << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCSR(
    const io::ReadsCSR<VertexID, NumEdges, Weight> &reader) {
  this->set_connectivity(reader.ReadCSR(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  /// std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] <<
  /// ", "
  ///          << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromEdgelistToCSR(
    std::string filename) {
  io::EdgeListReader<VertexID, NumEdges, Weight> reader(filename, false, false,
                                                        false, true, true);
  this->set_connectivity(reader.ReadCSR(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  /// std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] <<
  /// ", "
  ///          << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromMTXToCOO(
    std::string filename) {
  io::MTXReader<VertexID, NumEdges, Weight> reader(filename);
  this->set_connectivity(reader.ReadCOO(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  /// std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] <<
  /// ", "
  ///          << this->connectivity_->get_dimensions()[1] << std::endl;
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
  if (this->connectivity_->get_order() != 2) throw -1;
  // check dimensions
}

template <typename VertexID, typename NumEdges, typename Weight>
HyperGraph<VertexID,NumEdges,Weight>::HyperGraph(){}

template <typename VertexID, typename NumEdges, typename Weight>
HyperGraph<VertexID,NumEdges,Weight>::HyperGraph(format::Format *connectivity, VertexID base_type, VertexID constraint_num, format::CSR<VertexID,NumEdges,Weight> *xNetCSR){
  this->set_connectivity(connectivity, true);
  this->VerifyStructure();
  this->InitializeInfoFromConnection();
  this->base_type_ = base_type;
  this->constraint_num_ = constraint_num;
  this->xNetCSR_ = xNetCSR;
}

template <typename VertexID, typename NumEdges, typename Weight>
HyperGraph<VertexID,NumEdges,Weight>::HyperGraph(format::Format *connectivity,format::Array<Weight> *netWeights,format::Array<Weight> *cellWeights, VertexID base_type, VertexID constraint_num, format::CSR<VertexID,NumEdges,Weight> *xNetCSR){
  this->set_connectivity(connectivity, true);
  this->VerifyStructure();
  this->InitializeInfoFromConnection();
  this->base_type_ = base_type;
  this->constraint_num_ = constraint_num;
  this->xNetCSR_ = xNetCSR;
  this->netWeights_ = netWeights;
  this->cellWeights_ = cellWeights;
}

template <typename VertexID, typename NumEdges, typename ValueType>
HyperGraph<VertexID, NumEdges, ValueType>::~HyperGraph(){};


#if !defined(_HEADER_ONLY)
#include "init/object.inc"
#endif
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

}  // namespace sparsebase::object
