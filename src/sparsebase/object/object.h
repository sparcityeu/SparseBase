/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_OBJECT_OBJECT_H_
#define SPARSEBASE_SPARSEBASE_OBJECT_OBJECT_H_

#include <functional>
#include <memory>

#include "sparsebase/config.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/array.h"
#include "sparsebase/io/reader.h"

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
  std::unique_ptr<format::Format, std::function<void(format::Format *)>>
      connectivity_;

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
  Graph(format::Format *connectivity, NumEdges ncon, format::Array<Weight>** vertexWeights);
  Graph();
  Graph(const Graph<VertexID, NumEdges, Weight> &);
  Graph(Graph<VertexID, NumEdges, Weight> &&);
  Graph<VertexID, NumEdges, Weight> &operator=(
      const Graph<VertexID, NumEdges, Weight> &);
  void ReadConnectivityToCSR(const io::ReadsCSR<VertexID, NumEdges, Weight> &);
  void ReadConnectivityToCOO(const io::ReadsCOO<VertexID, NumEdges, Weight> &);
  void ReadConnectivityFromMTXToCOO(std::string filename);
  void ReadConnectivityFromEdgelistToCSR(std::string filename);
  void InitializeInfoFromConnection();
  virtual ~Graph();
  void VerifyStructure();
  VertexID n_;
  NumEdges m_;
  //! Number of vertex weights
  NumEdges ncon_ = 0;
  format::Array<Weight>** vertexWeights_ = nullptr;
};

template <typename VertexID, typename NumEdges, typename Weight>
class HyperGraph : public Graph<VertexID, NumEdges, Weight> {
 public:
  HyperGraph();
  HyperGraph(format::Format *connectivity,VertexID base_type,VertexID constraint_num,format::CSR<VertexID,NumEdges,Weight> *xNetCSR);
  HyperGraph(format::Format *connectivity, format::Array<Weight> *netWeights, format::Array<Weight> *cellWeights, VertexID base_type, VertexID constraint_num, format::CSR<VertexID,NumEdges,Weight> *xNetCSR);
  virtual ~HyperGraph();
  VertexID constraint_num_ = 1;
  VertexID base_type_;
  format::CSR<VertexID,NumEdges,Weight> *xNetCSR_;
  format::Array<Weight> *netWeights_ = nullptr;
  format::Array<Weight> *cellWeights_ = nullptr;
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

}  // namespace object

}  // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparsebase/object/object.cc"
#endif
#endif  // SPARSEBASE_SPARSEBASE_OBJECT_OBJECT_H_
