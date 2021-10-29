#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <fstream>

using namespace std;

namespace sparsebase{

  class Tensor{
    public:
      virtual ~Tensor(){};
      virtual int get_rank() = 0;
  };


  class CSF : public Tensor{
    public:
      CSF(): Tensor() {cout<<"CSF created."<< endl;}
    private:
      virtual ~CSF(){}; 
      virtual int get_rank(){return 0;}
  };

  //template<typename i_t, typename c_t>
  class COO : public Tensor{
    public:
      COO(): Tensor() {cout<<"COO created."<< endl;}
    private:
      virtual ~COO(){}; 
      virtual int get_rank(){return 0;}
  };

  class TensorCreator{ //could be improved
    int t;
    public:
      TensorCreator(int t){
        this->t = t;
      };
      ~TensorCreator(){};

      Tensor * create(){
        if(t == 0) return new CSF();
        else if(t == 1) return new COO();
        else return new COO();
      }
  };

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
