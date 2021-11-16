#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparseObject.hpp"

namespace sparsebase
{

  SparseObject::~SparseObject(){};

  template <typename ID_t, typename NNZ_t>
  AbstractSparseObject<ID_t, NNZ_t>::~AbstractSparseObject(){};
  template <typename ID_t, typename NNZ_t>
  SparseFormat<ID_t, NNZ_t> *AbstractSparseObject<ID_t, NNZ_t>::get_connectivity()
  {
    return connectivity;
  }

  template <typename v_t, typename e_t>
  Graph<v_t, e_t>::Graph(SparseFormat<v_t, e_t> *_connectivity)
  {
    this->connectivity = _connectivity;
    this->verify_structure();
    initialize_info_from_connection();
  }
  template <typename v_t, typename e_t>
  Graph<v_t, e_t>::Graph(SparseReader<v_t, e_t> *r)
  {
    this->connectivity = r->read()[0];
    delete r;
    this->verify_structure();
    initialize_info_from_connection();
    std::cout << "dimensions " << this->connectivity->get_dimensions()[0] << ", " << this->connectivity->get_dimensions()[1] << endl;
  }
  template <typename v_t, typename e_t>
  void Graph<v_t, e_t>::initialize_info_from_connection()
  {
    auto dimensions = this->connectivity->get_dimensions();
    n = dimensions[0];
    m = this->connectivity->get_num_nnz();
  }
  template <typename v_t, typename e_t>
  Graph<v_t, e_t>::~Graph(){};
  template <typename v_t, typename e_t>
  void Graph<v_t, e_t>::verify_structure()
  {
    // check order
    if (this->connectivity->get_order() != 2)
      throw -1;
    // check dimensions
  }

  template class AbstractSparseObject<unsigned int, unsigned int>;
  template class Graph<unsigned int, unsigned int>;
  //template<typename v_t, typename e_t, typename t_t>
  //class TemporalGraph : public AbstractSparseObject<v_t, e_t>{
  //  public:
  //    TemporalGraph(SparseFormat<v_t, e_t, t_t> * _connectivity){
  //      // init temporal graph
  //    }
  //    TemporalGraph(SparseReader<v_t, e_t, t_t> * r){
  //      // init temporal graph from file
  //    }
  //    virtual ~TemporalGraph(){};
  //    void verify_structure(){
  //      // check order
  //      if (this->connectivity->get_order() != 2) //throw error
  //      // check dimensions
  //    }
  //    v_t n;
  //    e_t m;
  //    // ...
  //};

}