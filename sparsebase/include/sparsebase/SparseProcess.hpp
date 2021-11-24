#ifndef _ORDERING_HPP
#define _ORDERING_HPP
#include <string>

namespace sparsebase
{
    template <typename ID_t>
    class Order
    {
    public:
      virtual ~Order(){};
    };

  template<typename ID_t>
  class AbstractOrder : public Order<ID_t>{
    public:
      virtual ~AbstractOrder(){};
      //virtual void test_order(){ };
  };

  template<typename ID_t, typename NNZ_t>
  class DegreeOrder : public AbstractOrder<ID_t> {
    public:
      DegreeOrder(){};
      //template <typename ID_t, typename NNZ_t, typename VAL_t>
      template<typename VAL_t>
      ID_t * get_order(SparseFormat<ID_t, NNZ_t> * sp){
        if(sp->format == Format::CSR_f){
          CSR<ID_t, NNZ_t, VAL_t> * csr = dynamic_cast<CSR<ID_t, NNZ_t, VAL_t>*>(sp); 
          cout << "CSR cast successful!" << endl;
          ID_t * r = get_order(csr);
          cout << &r << endl;
          return r; 
        }
        //else if(sp->format == Format::COO_f){
        //  return get_order(dynamic_cast<COO<ID_t, NNZ_t, VAL_t>*>(sp)); 
        //}
        else{
          throw std::invalid_argument( "Format " + to_string(sp->format) + ", not supported!" );
        }
      }
      template<typename VAL_t>
      ID_t * get_order(CSR<ID_t, NNZ_t, VAL_t> * csr){
        ID_t n = csr->get_dimensions()[0];
        ID_t * counts = new ID_t[n]();
        for(ID_t u = 0; u < n; u++){
          counts[csr->xadj[u+1] - csr->xadj[u]+1]++;
        }
        for(ID_t u = 1; u < n; u++){
          counts[u] += counts[u - 1];
        }
        ID_t * sorted = new ID_t[n];
        memset(sorted, -1, sizeof(ID_t) * n);
        ID_t * mr = new ID_t[n]();
        for(ID_t u = 0; u < n; u++){
          ID_t ec = counts[csr->xadj[u+1] - csr->xadj[u]];
          sorted[ec + mr[ec]] = u;
          mr[ec]++;
        }
        delete [] mr;
        delete [] counts;
        return sorted;
      }
      //ID_t * get_order(CSF<ID_t, NNZ_t> * csf, ID_t * sorted){
      //}
      
  };

  template<typename ID_t, typename NNZ_t>
  class RCMOrder : public AbstractOrder<ID_t> {};
} // namespace sparsebase


#endif
