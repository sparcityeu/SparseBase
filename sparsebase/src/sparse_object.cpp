#include <iostream>

#include "sparsebase/sparse_format.hpp"
#include "sparsebase/sparse_object.hpp"
#include "sparsebase/sparse_reader.hpp"

namespace sparsebase {

SparseObject::~SparseObject(){};

template <typename ID_t, typename NNZ_t, typename VAL_t>
AbstractSparseObject<ID_t, NNZ_t, VAL_t>::~AbstractSparseObject(){};
template <typename ID_t, typename NNZ_t, typename VAL_t>
SparseFormat<ID_t, NNZ_t, VAL_t> *
AbstractSparseObject<ID_t, NNZ_t, VAL_t>::get_connectivity() {
  return connectivity;
}

template <typename v_t, typename e_t, typename w_t>
Graph<v_t, e_t, w_t>::Graph(SparseFormat<v_t, e_t, w_t> *_connectivity) {
  this->connectivity = _connectivity;
  this->verify_structure();
  initialize_info_from_connection();
}
template <typename v_t, typename e_t, typename w_t>
void Graph<v_t, e_t, w_t>::read_connectivity_to_coo(
    const ReadsCOO<v_t, e_t, w_t> &reader) {
  this->connectivity = reader.read_coo();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity->get_dimensions()[0] << ", "
            << this->connectivity->get_dimensions()[1] << endl;
}
template <typename v_t, typename e_t, typename w_t>
void Graph<v_t, e_t, w_t>::read_connectivity_to_csr(
    const ReadsCSR<v_t, e_t, w_t> &reader) {
  this->connectivity = reader.read_csr();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity->get_dimensions()[0] << ", "
            << this->connectivity->get_dimensions()[1] << endl;
}
template <typename v_t, typename e_t, typename w_t>
void Graph<v_t, e_t, w_t>::read_connectivity_from_edgelist_to_csr(
    string filename) {
  UedgelistReader<v_t, e_t, w_t> reader(filename);
  this->connectivity = reader.read_csr();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity->get_dimensions()[0] << ", "
            << this->connectivity->get_dimensions()[1] << endl;
}
template <typename v_t, typename e_t, typename w_t>
void Graph<v_t, e_t, w_t>::read_connectivity_from_mtx_to_coo(string filename) {
  MTXReader<v_t, e_t, w_t> reader(filename);
  this->connectivity = reader.read_coo();
  this->verify_structure();
  initialize_info_from_connection();
  std::cout << "dimensions " << this->connectivity->get_dimensions()[0] << ", "
            << this->connectivity->get_dimensions()[1] << endl;
}
template <typename v_t, typename e_t, typename w_t>
Graph<v_t, e_t, w_t>::Graph() {}
template <typename v_t, typename e_t, typename VAL_t>
void Graph<v_t, e_t, VAL_t>::initialize_info_from_connection() {
  auto dimensions = this->connectivity->get_dimensions();
  n = dimensions[0];
  m = this->connectivity->get_num_nnz();
}
template <typename v_t, typename e_t, typename VAL_t>
Graph<v_t, e_t, VAL_t>::~Graph(){};
template <typename v_t, typename e_t, typename VAL_t>
void Graph<v_t, e_t, VAL_t>::verify_structure() {
  // check order
  if (this->connectivity->get_order() != 2)
    throw -1;
  // check dimensions
}

template class AbstractSparseObject<unsigned int, unsigned int, unsigned int>;
template class AbstractSparseObject<unsigned int, unsigned int, void>;
template class Graph<unsigned int, unsigned int, unsigned int>;
template class Graph<unsigned int, unsigned int, void>;
// template<typename v_t, typename e_t, typename t_t>
// class TemporalGraph : public AbstractSparseObject<v_t, e_t>{
//   public:
//     TemporalGraph(SparseFormat<v_t, e_t, t_t> * _connectivity){
//       // init temporal graph
//     }
//     TemporalGraph(SparseReader<v_t, e_t, t_t> * r){
//       // init temporal graph from file
//     }
//     virtual ~TemporalGraph(){};
//     void verify_structure(){
//       // check order
//       if (this->connectivity->get_order() != 2) //throw error
//       // check dimensions
//     }
//     v_t n;
//     e_t m;
//     // ...
// };

} // namespace sparsebase