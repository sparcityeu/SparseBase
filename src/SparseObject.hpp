#ifndef _SPARSEOBJECT_HPP
#define _SPARSEOBJECT_HPP

#include "Tensor.hpp"

namespace sparsebase{

  class SparseObject{
    public:
      virtual ~SparseObject(){};
  };

  template<typename v_t, typename e_t>
  class Graph : public SparseObject{
    Tensor * con;
    public:
      Graph(): SparseObject() {
        //machine learning etc. to choose the most optimal format
        TensorCreator * c = new CSRCreator<v_t, e_t>();
        con = c->create();
      }
      Graph(string filename): SparseObject() {
        //machine learning etc. to choose the most optimal format
        CSRCreator<v_t, e_t> * c = new CSRCreator<v_t, e_t>();
        con = c->create(filename);
      }
    private:
      v_t n;
      e_t m;
      virtual ~Graph(){}; 
  };

}

#endif
