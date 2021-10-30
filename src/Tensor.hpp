#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <fstream>

using namespace std;

namespace sparsebase{

  // TENSORS

  class Tensor{
    public:
      virtual ~Tensor(){};
      virtual int get_rank() = 0;
  };

  //abstract class
  class AbstractTensor : public Tensor{
    public:
      unsigned int order;
      //initialize order in the constructor
      virtual ~AbstractTensor(){};
      virtual int get_rank() = 0;
  };

  class CSF : public AbstractTensor{
    public:
      CSF(): AbstractTensor() {cout<<"CSF created."<< endl;}
    private:
      virtual ~CSF(){}; 
      virtual int get_rank(){return 0;}
  };

  class CSR : public AbstractTensor{
    public:
      CSR(): AbstractTensor() {cout<<"CSR created."<< endl;}
    private:
      virtual ~CSR(){}; 
      virtual int get_rank(){return 0;}
  };

  //template<typename i_t, typename c_t>
  class COO : public AbstractTensor{
    public:
      COO(): AbstractTensor() {cout<<"COO created."<< endl;}
      void go_crazy() {cout << "fuck cracy" << endl; } 
    private:
      virtual ~COO(){}; 
      virtual int get_rank(){return 0;}
  };

  // TENSOR CREATORS

  class TensorCreator{ //could be improved
    public:
      virtual ~TensorCreator(){};
      virtual Tensor * create() = 0;
  };

  class CSFCreator : public TensorCreator{
    public:
      CSFCreator(){}
      ~CSFCreator(){}
      Tensor * create(){
        return new CSF();
      }
  };

  class CSRCreator : public TensorCreator{
    public:
      CSRCreator(){}
      ~CSRCreator(){}
      Tensor * create(){
        return new CSR();
      }
  };

  // TENSOR READERS

  class TensorReader{
    public:
      virtual ~TensorReader(){};
  };

  template<typename i_t, typename nnz_t>
  class MtxReader : public TensorReader{
    public:
      MtxReader(string filename): TensorReader(), filename(filename) {}
    private:
      string filename;
      virtual ~MtxReader(){};
  };

}

#endif
