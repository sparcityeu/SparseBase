#ifndef _SPARSEOBJECT_HPP
#define _SPARSEOBJECT_HPP

#include "Tensor.hpp"

namespace sparsebase{

  class SparseObject{
    public:
      virtual ~SparseObject(){};
  };

  class Graph : public SparseObject{
    Tensor * con;
    public:
      Graph(): SparseObject() {
        //machine learning etc. to choose the most optimal format
        TensorCreator c(0);
        con = c.create();
      }
    private:
      virtual ~Graph(){}; 
  };

}

#endif
