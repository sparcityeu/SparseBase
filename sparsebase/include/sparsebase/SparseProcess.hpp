#ifndef _Reorder_HPP
#define _Reorder_HPP
#include "SparseFormat.hpp"
#include "SparseConverter.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>

namespace sparsebase
{
  struct  FormatVectorHash {
    std::size_t operator()(std::vector<Format> vf) const;
  };
  class PreprocessType {};
  template <class Preprocess, typename func, typename key = std::vector<Format>, typename key_hash = FormatVectorHash, typename key_equal = std::equal_to<std::vector<Format>>>
  class MapToFunctionMixin : public Preprocess {
    using Preprocess::Preprocess;
    protected:
    std::unordered_map<key, func, key_hash, key_equal> _map_to_function;
    bool register_function_no_override(const key& key_of_function, const func & func_ptr);
    void register_function(const key& key_of_function, const func & func_ptr);
    bool unregister_function(const key& key_of_function);
  }; 
  template <class Parent, typename ID_t, typename NNZ_t, typename VAL_t>
  class SparseConverterMixin : public Parent {
    using Parent::Parent;
    protected:
      SparseConverter<ID_t, NNZ_t, VAL_t> _sc;
    public:
      void set_converter(const SparseConverter<ID_t, NNZ_t, VAL_t> & new_sc);
      void reset_converter();
  };

  template <typename ID_t, typename NNZ_t, typename VAL_t>
  using ReorderFunction = ID_t* (*)(std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*>);

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class ReorderPreprocessType : public MapToFunctionMixin<SparseConverterMixin<PreprocessType, ID_t, NNZ_t, VAL_t>, ReorderFunction<ID_t, NNZ_t, VAL_t>>{
    public:
      virtual ~ReorderPreprocessType ();
  };


  template <typename ID_t, typename NNZ_t, typename VAL_t, class PreprocessingImpl, typename PreprocessFunction, typename config_key = std::vector<Format>, typename config_key_hash = FormatVectorHash, typename config_key_equal_to = std::equal_to<std::vector<Format>>>
  class FormatMatcherMixin : public PreprocessingImpl {
    typedef std::unordered_map<config_key, PreprocessFunction, config_key_hash, config_key_equal_to> conversion_map;
  protected:
    using PreprocessingImpl::PreprocessingImpl; 
    std::tuple<PreprocessFunction, conversion_schema> get_function(config_key key, conversion_map map, SparseConverter<ID_t, NNZ_t, VAL_t> sc);
    template <typename F>
    std::vector<Format> pack_formats(F sf);
    template <typename F, typename... SF>
    std::vector<Format> pack_formats(F sf, SF... sfs);
    template <typename F>
    std::vector<F> pack_sfs(F sf);
    template <typename F, typename... SF>
    std::vector<F> pack_sfs(F sf, SF... sfs);
    template<typename F, typename... SF>
    std::tuple<PreprocessFunction, std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*>> execute(conversion_map map, SparseConverter<ID_t, NNZ_t, VAL_t> sc, F sf, SF... sfs);
  };
  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class DegreeReorder: public ReorderPreprocessType<ID_t, NNZ_t, VAL_t> {
    public:
      DegreeReorder(int hyperparameter);
    protected:
      int _hyperparameter;
      static ID_t* calculate_Reorder_csr(std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> formats);
  };

  template <typename ID_t, typename NNZ_t, typename VAL_t>
  class DegreeReorderInstance : FormatMatcherMixin<ID_t, NNZ_t, VAL_t, DegreeReorder<ID_t, NNZ_t, VAL_t>, ReorderFunction<ID_t, NNZ_t, VAL_t>> {
    typedef FormatMatcherMixin<ID_t, NNZ_t, VAL_t, DegreeReorder<ID_t, NNZ_t, VAL_t>, ReorderFunction<ID_t, NNZ_t, VAL_t>> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_reorder(SparseFormat<ID_t, NNZ_t, VAL_t>* csr);
  };

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class RCMReorder : public ReorderPreprocessType<ID_t, NNZ_t, VAL_t> {
    typedef typename std::make_signed<ID_t>::type s_ID_t;  
    public:
      RCMReorder();
    protected:
      static ID_t peripheral(NNZ_t* xadj, ID_t* adj, ID_t n, ID_t start, s_ID_t* distance, ID_t* Q);
      static ID_t* get_reorder_csr(std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> formats);
  };

  template <typename ID_t, typename NNZ_t, typename VAL_t, typename Reorder_T>
  class ReorderInstance : FormatMatcherMixin<ID_t, NNZ_t, VAL_t, Reorder_T, ReorderFunction<ID_t, NNZ_t, VAL_t>> {
    typedef FormatMatcherMixin<ID_t, NNZ_t, VAL_t, Reorder_T, ReorderFunction<ID_t, NNZ_t, VAL_t>> Base;
    using Base::Base; // Used to forward constructors from base
    public:
    ID_t* get_reorder(SparseFormat<ID_t, NNZ_t, VAL_t>* csr);
  };
} // namespace sparsebase


#endif
