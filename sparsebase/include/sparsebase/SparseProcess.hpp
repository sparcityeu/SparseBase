#ifndef _ORDERING_HPP
#define _ORDERING_HPP
#include "SparseFormat.hpp"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sparsebase
{
  std::string my_to_string(std::vector<Format> v){
    string r = "";
    for (auto e : v){
      r+=std::to_string('a');
    }
    return r;
  }
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

  typedef std::vector<std::tuple<bool, Format>> conversion_schema;
  class SparseConverter{
    public:
    bool can_convert(Format in, Format out){
      return true;
    }  
    // TODO: what about the other templated variables?
    //       can we add a "clone" function to handle this?
    template <typename ID_t, typename NNZ_t>
    std::vector<SparseFormat<ID_t, NNZ_t>*> apply_conversion_schema(conversion_schema sc, std::vector<SparseFormat<ID_t, NNZ_t>*> ptr){
      return ptr;
    }
  };

  template <class ID_t, class NNZ_t, class ProcessingImpl, typename ProcessingFunc, typename ProcessingReturn, typename config_key = std::vector<Format>>
  class ExecutableProcess : public ProcessingImpl {
  protected:
    using ProcessingImpl::ProcessingImpl; 
    std::tuple<ProcessingFunc, conversion_schema> get_function(config_key key, std::unordered_map<config_key, ProcessingFunc> map, SparseConverter sc){
      conversion_schema cs;
      ProcessingFunc func = nullptr;
      // Check if the key is in the map (requires == and != functions for config_key)
      // If it is, 
        //return the function
      if (map.find(key) != map.end()){
        for (auto f : key){
          cs.push_back(make_tuple(false, (Format)f));
        }
        func = map[key];
      } 
      // If it isn't,
      else {
        // check if it can be done
          // If it can, carry out the correct conversions and return a function pointer plus the conversions
          // sort the keys by hamming distance
          // check the keys one by one
          // construct the cs
      }
      return make_tuple(func, cs);
    }
    template <typename F>
    std::vector<Format> pack_formats(F sf){
      SparseFormat<ID_t, NNZ_t>* casted = static_cast<SparseFormat<ID_t, NNZ_t>*>(sf);
      return {casted->get_format()};
    }
    template <typename F, typename... SF>
    std::vector<Format> pack_formats(F sf, SF... sfs){
      SparseFormat<ID_t, NNZ_t>* casted = static_cast<SparseFormat<ID_t, NNZ_t>*>(sf);
      std::vector<Format> f = {casted->get_format()};
      std::vector<Format> remainder = pack_formats(sfs...);
      for (auto i : remainder){
        f.push_back(i);
      }
      return f;
    }
    template <typename F>
    std::vector<F> pack_sfs(F sf){
      return {sf};
    }
    template <typename F, typename... SF>
    std::vector<F> pack_sfs(F sf, SF... sfs){
      std::vector<F> f = {sf};
      std::vector<F> remainder = pack_formats(sfs...);
      for (auto i : remainder){
        f.push_back(i);
      }
      return f;
    }
    template<typename F, typename... SF>
    ProcessingReturn execute(F sf, SF... sfs){
      SparseConverter sc;
      // pack the SFs into a vector
      vector<SparseFormat<ID_t, NNZ_t>*> packed_sfs = pack_sfs(sf, sfs...);
      // pack the SF formats into a vector
      vector<Format> formats = pack_formats(sf, sfs...);
      // get conversion schema
      std::tuple<ProcessingFunc, conversion_schema>  cs = get_function(my_to_string(formats), this->map, sc);
      // carry out conversion
      std::vector<SparseFormat<ID_t, NNZ_t>*> converted = sc.apply_conversion_schema(get<1>(cs), packed_sfs);
      // carry out the correct call using the map
      return get<0>(cs)(packed_sfs);
    }
  };

  template <typename ID_t, typename NNZ_t>
  using OrderingFunction = ID_t* (*)(std::vector<SparseFormat<ID_t, NNZ_t>*>);

  template<typename ID_t, typename NNZ_t, typename V>
  class DegreeOrder : public AbstractOrder<ID_t> {
    public:
      DegreeOrder(int _hyperparameter):hyperparameter(_hyperparameter){
        map.emplace(my_to_string({CSR_f}), calculate_order_csr);
      };
    protected:
      std::unordered_map<std::string, OrderingFunction<ID_t, NNZ_t>> map;
      int hyperparameter;
      static ID_t* calculate_order_csr(std::vector<SparseFormat<ID_t, NNZ_t>*> formats){
        CSR<ID_t, NNZ_t, void>* csr = static_cast<CSR<ID_t, NNZ_t, void>*>(formats[0]);
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
  };

  template <typename ID_t, typename NNZ_t>
  class ExecutableDegreeOrdering : ExecutableProcess<ID_t, NNZ_t, DegreeOrder<ID_t, NNZ_t, void>, OrderingFunction<ID_t, NNZ_t>, ID_t*, std::string> {
    typedef ExecutableProcess<ID_t, NNZ_t, DegreeOrder<ID_t, NNZ_t, void>, OrderingFunction<ID_t, NNZ_t>, ID_t*, std::string> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_order(SparseFormat<ID_t, NNZ_t>* csr){
      return this->execute(csr);
    }
  };

  template<typename ID_t, typename NNZ_t>
  class RCMOrder : public AbstractOrder<ID_t> {
    public:
      RCMOrder() {
        map.emplace(my_to_string({CSR_f}), get_order_csr);
      }
    protected:
      std::unordered_map<std::string, OrderingFunction<ID_t, NNZ_t>> map;
      static ID_t* get_order_csr(std::vector<SparseFormat<ID_t, NNZ_t>*> formats){
        CSR<ID_t, NNZ_t, void>* csr = static_cast<CSR<ID_t, NNZ_t, void>*>(formats[0]);
      }
  };

  template <typename ID_t, typename NNZ_t, typename ORDER_T>
  class ExecutableOrdering : ExecutableProcess<ID_t, NNZ_t, ORDER_T, OrderingFunction<ID_t, NNZ_t>, ID_t*, std::string> {
    typedef ExecutableProcess<ID_t, NNZ_t, ORDER_T, OrderingFunction<ID_t, NNZ_t>, ID_t*, std::string> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_order(SparseFormat<ID_t, NNZ_t>* csr){
      return this->execute(csr);
    }
  };
} // namespace sparsebase


#endif
