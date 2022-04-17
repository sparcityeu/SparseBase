#include <iostream>

#include "object.h"
#include "sparsebase/format/format.h"
#include "sparsebase/utils/io/reader.h"

using namespace sparsebase::format;
using namespace sparsebase::utils;

namespace sparsebase::object {

Object::~Object(){};

template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::~AbstractObject(){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::AbstractObject()
    : connectivity_(nullptr, BlankDeleter<Format>()){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::AbstractObject(
    AbstractObject<IDType, NNZType, ValueType> &&rhs)
    : connectivity_(std::move(rhs.connectivity_)){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractObject<IDType, NNZType, ValueType>::AbstractObject(
    const AbstractObject<IDType, NNZType, ValueType> &rhs)
    : connectivity_((Format *)rhs.connectivity_->Clone(),
                    BlankDeleter<Format>()){};
template <typename IDType, typename NNZType, typename ValueType>
Format *AbstractObject<IDType, NNZType, ValueType>::get_connectivity() const {
  return connectivity_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
bool AbstractObject<IDType, NNZType, ValueType>::ConnectivityIsOwned() const {
  return (connectivity_.get_deleter().target_type() !=
          typeid(BlankDeleter<Format>));
}
template <typename IDType, typename NNZType, typename ValueType>
Format *AbstractObject<IDType, NNZType, ValueType>::release_connectivity() {
  auto ptr = connectivity_.release();
  connectivity_ = std::unique_ptr<Format, std::function<void(Format *)>>(
      ptr, BlankDeleter<Format>());
  return ptr;
}
template <typename IDType, typename NNZType, typename ValueType>
void AbstractObject<IDType, NNZType, ValueType>::set_connectivity(Format *conn,
                                                                  bool own) {

  if (own)
    connectivity_ = std::unique_ptr<Format, std::function<void(Format *)>>(
        conn, Deleter<Format>());
  else
    connectivity_ = std::unique_ptr<Format, std::function<void(Format *)>>(
        conn, BlankDeleter<Format>());
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
  this->set_connectivity(static_cast<Format *>(rhs.connectivity_->Clone()),
                         true);
  InitializeInfoFromConnection();
}
template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight> &Graph<VertexID, NumEdges, Weight>::operator=(
    const Graph<VertexID, NumEdges, Weight> &rhs) {
  this->set_connectivity(static_cast<Format *>(rhs.connectivity_->Clone()),
                         true);
  InitializeInfoFromConnection();
  return *this;
}
template <typename VertexID, typename NumEdges, typename Weight>
Graph<VertexID, NumEdges, Weight>::Graph(Format *connectivity) {
  // this->connectivity_ = connectivity;
  this->set_connectivity(connectivity, true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCOO(
    const utils::io::ReadsCOO<VertexID, NumEdges, Weight> &reader) {
  this->set_connectivity(reader.ReadCOO(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityToCSR(
    const utils::io::ReadsCSR<VertexID, NumEdges, Weight> &reader) {
  this->set_connectivity(reader.ReadCSR(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromEdgelistToCSR(
    std::string filename) {
  utils::io::UedgelistReader<VertexID, NumEdges, Weight> reader(filename);
  this->set_connectivity(reader.ReadCSR(), true);
  this->VerifyStructure();
  InitializeInfoFromConnection();
  std::cout << "dimensions " << this->connectivity_->get_dimensions()[0] << ", "
            << this->connectivity_->get_dimensions()[1] << std::endl;
}
template <typename VertexID, typename NumEdges, typename Weight>
void Graph<VertexID, NumEdges, Weight>::ReadConnectivityFromMTXToCOO(
    std::string filename) {
  utils::io::MTXReader<VertexID, NumEdges, Weight> reader(filename);
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

} // namespace sparsebase::object
