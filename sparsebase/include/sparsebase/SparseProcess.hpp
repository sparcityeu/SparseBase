#ifndef _ORDERING_HPP
#define _ORDERING_HPP
#include "SparseFormat.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>

namespace sparsebase
{
  template <typename ID_t, typename NNZ_t>
  using SparseFormatCont = std::vector<SparseFormat<ID_t, NNZ_t>*>;

  struct  FormatVectorHash {
    std::size_t operator()(std::vector<Format> vf) const{
      int hash = 0;
      for (auto f : vf) hash+=f*19381; 
      return hash;
    }
  };
  typedef std::vector<std::tuple<bool, Format>> conversion_schema;
  class SparseConverter{
    public:
    bool can_convert(Format in, Format out){
      return false;
    }  
    // TODO: what about the other templated variables?
    //       can we add a "clone" function to handle this?
    template <typename ID_t, typename NNZ_t>
    std::vector<SparseFormat<ID_t, NNZ_t>*> apply_conversion_schema(conversion_schema sc, std::vector<SparseFormat<ID_t, NNZ_t>*> ptr){
      return ptr;
    }
  };
  class ProcessType {
  };
  template <class Process, typename func, typename key = std::vector<Format>, typename key_hash = FormatVectorHash, typename key_equal = std::equal_to<std::vector<Format>>>
  class MapToFunctionMixin : public Process {
    using Process::Process;
    protected:
    std::unordered_map<key, func, key_hash, key_equal> _map_to_function;
    bool register_function_no_override(const key& key_of_function, const func & func_ptr){
      if (_map_to_function.find(key_of_function) == _map_to_function.end()){
        return false;  // function already exists for this key
      } else {
        _map_to_function[key_of_function] = func_ptr;
        return true;
      }
    }
    void register_function(const key& key_of_function, const func & func_ptr){
      _map_to_function[key_of_function] = func_ptr;
    }
    bool unregister_function(const key& key_of_function){
      if (_map_to_function.find(key_of_function) == _map_to_function.end()){
        return false;  // function already exists for this key
      } else {
        _map_to_function.erase(key_of_function);
        return true;
      }
    }
  }; 
  template <class Parent>
  class SparseConverterMixin : public Parent {
    using Parent::Parent;
    protected:
      SparseConverter _sc;
    public:
      void set_converter(const SparseConverter & new_sc){
        _sc = new_sc;
      }
      void reset_converter(){
        SparseConverter new_sc;
        _sc = new_sc;
      }
  };

  template <typename ID_t, typename NNZ_t>
  using OrderingFunction = ID_t* (*)(std::vector<SparseFormat<ID_t, NNZ_t>*>);

  template<typename ID_t, typename NNZ_t>
  class ReorderProcessType : public MapToFunctionMixin<SparseConverterMixin<ProcessType>, OrderingFunction<ID_t, NNZ_t>>{
    public:
      virtual ~ReorderProcessType (){};
  };


  template <class ID_t, class NNZ_t, class ProcessingImpl, typename ProcessingFunc, typename config_key = std::vector<Format>, typename config_key_hash = FormatVectorHash, typename config_key_equal_to = std::equal_to<std::vector<Format>>>
  class FormatMatcherMixin : public ProcessingImpl {
    typedef std::unordered_map<config_key, ProcessingFunc, config_key_hash, config_key_equal_to> conversion_map;
  protected:
    using ProcessingImpl::ProcessingImpl; 
    std::tuple<ProcessingFunc, conversion_schema> get_function(config_key key, conversion_map map, SparseConverter sc){
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
            for (int i =0; i < potential_key.size(); i++){
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
          if (num_conversions > get<0>(potential_usable_key)){
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
    std::tuple<ProcessingFunc, std::vector<SparseFormat<ID_t, NNZ_t>*>> execute(conversion_map map, SparseConverter sc, F sf, SF... sfs){
      // pack the SFs into a vector
      vector<SparseFormat<ID_t, NNZ_t>*> packed_sfs = pack_sfs(sf, sfs...);
      // pack the SF formats into a vector
      vector<Format> formats = pack_formats(sf, sfs...);
      // get conversion schema
      std::tuple<ProcessingFunc, conversion_schema> ret = get_function(formats, map, sc);
      ProcessingFunc func = get<0>(ret);
      conversion_schema cs = get<1>(ret);
      // carry out conversion
      std::vector<SparseFormat<ID_t, NNZ_t>*> converted = sc.apply_conversion_schema(cs, packed_sfs);
      // carry out the correct call using the map
      return make_tuple(func, converted);
      //return get<0>(cs)(packed_sfs);
    }
  };
  template<typename ID_t, typename NNZ_t, typename V>
  class DegreeReorder: public ReorderProcessType<ID_t, NNZ_t> {
    public:
      DegreeReorder(int hyperparameter):_hyperparameter(hyperparameter){
        //this->map[{CSR_f}]= calculate_order_csr;
        this->register_function({CSR_f}, calculate_order_csr);
      };
    protected:
      int _hyperparameter;
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
  class DegreeOrderingInterface : FormatMatcherMixin<ID_t, NNZ_t, DegreeReorder<ID_t, NNZ_t, void>, OrderingFunction<ID_t, NNZ_t>> {
    typedef FormatMatcherMixin<ID_t, NNZ_t, DegreeReorder<ID_t, NNZ_t, void>, OrderingFunction<ID_t, NNZ_t>> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_order(SparseFormat<ID_t, NNZ_t>* csr){
      //return this->execute(this->map, this->_sc, csr);
      std::tuple <OrderingFunction<ID_t, NNZ_t>, std::vector<SparseFormat<ID_t, NNZ_t> *>> func_formats = this->execute(this->_map_to_function, this->_sc, csr);
      OrderingFunction<ID_t, NNZ_t> func = get<0>(func_formats);
      std::vector<SparseFormat<ID_t, NNZ_t>*> sfs = get<1>(func_formats);
      return func(sfs);
    }
  };

  template<typename ID_t, typename NNZ_t>
  class RCMOrder : public ReorderProcessType<ID_t, NNZ_t> {
    public:
      RCMOrder() {
        this->map[{CSR_f}]= get_order_csr;
      }
    protected:
      static ID_t* get_order_csr(std::vector<SparseFormat<ID_t, NNZ_t>*> formats){
        CSR<ID_t, NNZ_t, void>* csr = static_cast<CSR<ID_t, NNZ_t, void>*>(formats[0]);
      }
  };

  template <typename ID_t, typename NNZ_t, typename ORDER_T>
  class OrderingInterface : FormatMatcherMixin<ID_t, NNZ_t, ORDER_T, OrderingFunction<ID_t, NNZ_t>> {
    typedef FormatMatcherMixin<ID_t, NNZ_t, ORDER_T, OrderingFunction<ID_t, NNZ_t>> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_order(SparseFormat<ID_t, NNZ_t>* csr){
      //return this->execute(this->map, this->_sc, csr);
      std::tuple <OrderingFunction<ID_t, NNZ_t>, std::vector<SparseFormat<ID_t, NNZ_t> *>> func_formats = this->execute(this->_map_to_function, this->_sc, csr);
      OrderingFunction<ID_t, NNZ_t> func = get<0>(func_formats);
      std::vector<SparseFormat<ID_t, NNZ_t>*> sfs = get<1>(func_formats);
      return func(sfs);
    }
  };
} // namespace sparsebase


#endif
