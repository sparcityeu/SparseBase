#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

namespace sparsebase{

  //! Enum keeping formats  
  enum Format{
    //! CSR Format
    CSR_f=0, 
    //! COO Format
    COO_f=1 
  };
  // TENSORS

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class SparseFormat{
    public:
      Format format;
      virtual ~SparseFormat(){};
      virtual unsigned int get_order() = 0;
      virtual Format get_format() = 0;
      virtual std::vector<ID_t> get_dimensions() = 0;
      virtual NNZ_t get_num_nnz() = 0;
      virtual NNZ_t * get_xadj() = 0;
      virtual ID_t * get_adj() = 0;
      virtual ID_t * get_is() = 0;
      virtual VAL_t * get_vals() = 0;
      virtual ID_t ** get_ind() = 0;
  };

  //abstract class
  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class AbstractSparseFormat : public SparseFormat<ID_t, NNZ_t, VAL_t>{
    public:
      //initialize order in the constructor
      AbstractSparseFormat();
      virtual ~AbstractSparseFormat();
      unsigned int get_order() override;
      virtual Format get_format() override;
      std::vector<ID_t>get_dimensions() override;
      NNZ_t get_num_nnz() override;
      NNZ_t * get_xadj() override;
      ID_t * get_adj() override;
      ID_t * get_is() override;
      VAL_t * get_vals() override;
      ID_t ** get_ind() override;

      unsigned int order;
      std::vector<ID_t> dimension;
      Format format;
      NNZ_t nnz;
  };

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class COO : public AbstractSparseFormat<ID_t, NNZ_t, VAL_t>{
    public:
      COO();
      COO(ID_t _n, ID_t _m, NNZ_t _nnz, ID_t * _adj, ID_t* _is, VAL_t* _vals);
      virtual ~COO();
      Format get_format() override;
      ID_t * adj;
      ID_t * is;
      VAL_t * vals;
      ID_t * get_adj() override;
      ID_t * get_is() override;
      VAL_t * get_vals() override;
  };
  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class CSR : public AbstractSparseFormat<ID_t, NNZ_t, VAL_t>{
    public:
      CSR();
      CSR(ID_t _n, ID_t _m, NNZ_t *_xadj, ID_t *_adj, VAL_t *_vals);
      Format get_format() override;
      virtual ~CSR();
      NNZ_t * xadj;
      ID_t * adj;
      VAL_t * vals;
      ID_t * get_xadj() override;
      ID_t * get_adj() override;
      VAL_t * get_vals() override;
  };

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class CSF : public AbstractSparseFormat<ID_t, NNZ_t, VAL_t>{
    public:
      CSF(unsigned int order);
      Format get_format() override;
      virtual ~CSF(); 
      NNZ_t ** ind;
      VAL_t * vals;
      ID_t ** get_ind() override;
      VAL_t * get_vals() override;
  };

}
#endif