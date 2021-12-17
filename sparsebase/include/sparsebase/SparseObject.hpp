#ifndef _SPARSEOBJECT_HPP
#define _SPARSEOBJECT_HPP

#include "SparseFormat.hpp"
#include "SparseReader.hpp"

namespace sparsebase{

  class SparseObject{
    public:
      virtual ~SparseObject();
      virtual void verify_structure() = 0;
  };

  template <typename ID_t, typename NNZ_t, typename VAL_t>
  class AbstractSparseObject : public SparseObject{
    protected:
      SparseFormat<ID_t, NNZ_t, VAL_t> *connectivity;
    public:
      virtual ~AbstractSparseObject();
      SparseFormat<ID_t, NNZ_t, VAL_t> * get_connectivity();
  };

  template<typename v_t, typename e_t, typename w_t>
  class Graph : public AbstractSparseObject<v_t, e_t, w_t>{
    public:
      Graph(SparseFormat<v_t, e_t, w_t> * _connectivity);
      Graph();
      void read_connectivity_to_csr(const ReadsCSR<v_t, e_t, w_t>&);
      void read_connectivity_to_coo(const ReadsCOO<v_t, e_t, w_t>&);
      void read_connectivity_from_mtx_to_coo(string filename);
      void read_connectivity_from_edgelist_to_csr(string filename);
      void initialize_info_from_connection();
      virtual ~Graph();
      void verify_structure();
      v_t n;
      e_t m;
  };

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

#endif
