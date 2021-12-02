#ifndef _ORDERING_HPP
#define _ORDERING_HPP
#include "SparseFormat.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>

namespace sparsebase
{
  struct  FormatVectorHash {
    std::size_t operator()(std::vector<Format> vf) const{
      int hash = 0;
      for (auto f : vf) hash+=f*19381; 
      return hash;
    }
  };
  //struct  FormatVectorEqual {
  //  bool operator==(const std::vector<Format> & lhs, const std::vector<Format> & rhs) const{
  //    int hash = 0;
  //    for (auto f : vf) hash+=f*19381; 
  //    return hash;
  //  }
  //};
    template <typename ID_t>
    class Order
    {
    public:
      virtual ~Order(){};
    };


  template <typename ID_t, typename NNZ_t>
  using OrderingFunction = ID_t* (*)(std::vector<SparseFormat<ID_t, NNZ_t>*>);

  template<typename ID_t, typename NNZ_t>
  class AbstractOrder : public Order<ID_t>{
    protected:
      std::unordered_map<std::vector<Format>, OrderingFunction<ID_t, NNZ_t>, FormatVectorHash> map;
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

  template <class ID_t, class NNZ_t, class ProcessingImpl, typename ProcessingFunc, typename ProcessingReturn, typename config_key = std::vector<Format>, typename config_key_hash = FormatVectorHash>
  class ExecutableProcess : public ProcessingImpl {
  protected:
    using ProcessingImpl::ProcessingImpl; 
    std::tuple<ProcessingFunc, conversion_schema> get_function(config_key key, std::unordered_map<config_key, ProcessingFunc, config_key_hash> map, SparseConverter sc){
      conversion_schema cs;
      ProcessingFunc func = nullptr;
      if (map.find(key) != map.end()){
        for (auto f : key){
          cs.push_back(make_tuple(false, (Format)f));
        }
        func = map[key];
      } 
      else {
        std::vector<config_key> all_keys;
        for (auto key_func : map){
          all_keys.push_back(key_func.first);
        } 
        std::vector<std::tuple<unsigned int, conversion_schema, config_key>> usable_keys;
        for (auto potential_key : all_keys){
          if (potential_key.size() == key.size()){
            conversion_schema temp_cs;
            int conversions = 0;
            bool is_usable = true;
            for (int i =0; i < potential_key.size(); i){
              if (key[i] == potential_key[i]){
                temp_cs.push_back(make_tuple(false, potential_key[i]));
              }
              else if (sc.can_convert(key[i], potential_key[i])){
                temp_cs.push_back(make_tuple(true, potential_key[i]));
                conversions++;
              } else {
                is_usable = false;
              }
            }
            if (is_usable){
              usable_keys.push_back(make_tuple(conversions,temp_cs,potential_key));
            }
          }
        }
        if (usable_keys.size() == 0){
          throw 1; // TODO: add a custom exception type
        }
        std::tuple<ProcessingFunc, conversion_schema> best_conversion;
        unsigned int num_conversions = (unsigned int)-1;
        for (auto potential_usable_key : usable_keys){
          if (num_conversions < get<0>(potential_usable_key)){
            num_conversions = get<0>(potential_usable_key);
            cs = get<1>(potential_usable_key);
            func = map[get<2>(potential_usable_key)];
          }
        }
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
      std::tuple<ProcessingFunc, conversion_schema>  cs = get_function(formats, this->map, sc);
      // carry out conversion
      std::vector<SparseFormat<ID_t, NNZ_t>*> converted = sc.apply_conversion_schema(get<1>(cs), packed_sfs);
      // carry out the correct call using the map
      return get<0>(cs)(packed_sfs);
    }
  };
  template<typename ID_t, typename NNZ_t, typename V>
  class DegreeOrder : public AbstractOrder<ID_t, NNZ_t> {
    public:
      DegreeOrder(int _hyperparameter):hyperparameter(_hyperparameter){
        map[{CSR_f}]= calculate_order_csr;
      };
    protected:
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
  class ExecutableDegreeOrdering : ExecutableProcess<ID_t, NNZ_t, DegreeOrder<ID_t, NNZ_t, void>, OrderingFunction<ID_t, NNZ_t>, ID_t*> {
    typedef ExecutableProcess<ID_t, NNZ_t, DegreeOrder<ID_t, NNZ_t, void>, OrderingFunction<ID_t, NNZ_t>, ID_t*> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_order(SparseFormat<ID_t, NNZ_t>* csr){
      return this->execute(csr);
    }
  };

  template<typename ID_t, typename NNZ_t>
  class RCMOrder : public AbstractOrder<ID_t, NNZ_t> {
    public:
      RCMOrder() {
        map[{CSR_f}]= get_order_csr;
      }
    protected:
      static ID_t* get_order_csr(std::vector<SparseFormat<ID_t, NNZ_t>*> formats){
        CSR<ID_t, NNZ_t, void>* csr = static_cast<CSR<ID_t, NNZ_t, void>*>(formats[0]);
      }
  };

  template <typename ID_t, typename NNZ_t, typename ORDER_T>
  class ExecutableOrdering : ExecutableProcess<ID_t, NNZ_t, ORDER_T, OrderingFunction<ID_t, NNZ_t>, ID_t*> {
    typedef ExecutableProcess<ID_t, NNZ_t, ORDER_T, OrderingFunction<ID_t, NNZ_t>, ID_t*> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_order(SparseFormat<ID_t, NNZ_t>* csr){
      return this->execute(csr);
    }
  };
} // namespace sparsebase


#endif
