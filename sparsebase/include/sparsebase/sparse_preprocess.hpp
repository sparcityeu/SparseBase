#ifndef _Reorder_HPP
#define _Reorder_HPP
#include "sparse_converter.hpp"
#include "sparse_format.hpp"
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sparsebase {
struct FormatVectorHash {
  std::size_t operator()(std::vector<Format> vf) const;
};
class PreprocessType {};
template <class Preprocess, typename Function, typename Key = std::vector<Format>,
          typename KeyHash = FormatVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<Format>>>
class MapToFunctionMixin : public Preprocess {
  using Preprocess::Preprocess;

public:
  std::unordered_map<Key, Function, KeyHash, KeyEqualTo> _map_to_function;
  bool register_function_no_override(const Key &key_of_function,
                                     const Function &func_ptr);
  void register_function(const Key &key_of_function, const Function &func_ptr);
  bool unregister_function(const Key &key_of_function);
};
template <class Parent, typename ID, typename NumNonZeros, typename Value>
class SparseConverterMixin : public Parent {
  using Parent::Parent;

protected:
  SparseConverter<ID, NumNonZeros, Value> sc_;

public:
  void set_converter(const SparseConverter<ID, NumNonZeros, Value> &new_sc);
  void reset_converter();
};

struct ReorderParams {};
template <typename ID, typename NumNonZeros, typename Value>
using ReorderFunction =
    ID *(*)(std::vector<SparseFormat<ID, NumNonZeros, Value> *>, ReorderParams *);

template <typename ID, typename NumNonZeros, typename Value>
class ReorderPreprocessType
    : public MapToFunctionMixin<
          SparseConverterMixin<PreprocessType, ID, NumNonZeros, Value>,
          ReorderFunction<ID, NumNonZeros, Value>> {
protected:
  std::unique_ptr<ReorderParams> params_;

public:
  virtual ~ReorderPreprocessType();
};

template <typename ID, typename NumNonZeros, typename Value,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key = std::vector<Format>,
          typename KeyHash = FormatVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<Format>>>
class FormatMatcherMixin : public PreprocessingImpl {
  typedef std::unordered_map<Key, PreprocessFunction, KeyHash,
                             KeyEqualTo>
      ConversionMap;

protected:
  using PreprocessingImpl::PreprocessingImpl;
  std::tuple<PreprocessFunction, ConversionSchema>
  get_function(Key key, ConversionMap map,
               SparseConverter<ID, NumNonZeros, Value> sc);
  template <typename F> std::vector<Format> pack_formats(F sf);
  template <typename F, typename... SF>
  std::vector<Format> pack_formats(F sf, SF... sfs);
  template <typename F> std::vector<F> pack_sfs(F sf);
  template <typename F, typename... SF>
  std::vector<F> pack_sfs(F sf, SF... sfs);
  template <typename F, typename... SF>
  std::tuple<PreprocessFunction,
             std::vector<SparseFormat<ID, NumNonZeros, Value> *>>
  execute(ConversionMap map, SparseConverter<ID, NumNonZeros, Value> sc, F sf,
          SF... sfs);
};
template <typename ID, typename NumNonZeros, typename Value>
class DegreeReorder : public ReorderPreprocessType<ID, NumNonZeros, Value> {
public:
  DegreeReorder(int hyperparameter);

protected:
  struct DegreeReorderParams : ReorderParams {
    int hyperparameter;
    DegreeReorderParams(int h) : hyperparameter(h) {}
  };
  static ID *
  calculate_Reorder_csr(std::vector<SparseFormat<ID, NumNonZeros, Value> *> formats,
                        ReorderParams *params);
};

template <typename ID, typename NumNonZeros, typename Value>
class GenericReorder : public ReorderPreprocessType<ID, NumNonZeros, Value> {
public:
  GenericReorder();
};
template <typename ID, typename NumNonZeros, typename Value>
class DegreeReorderInstance
    : public FormatMatcherMixin<ID, NumNonZeros, Value,
                                DegreeReorder<ID, NumNonZeros, Value>,
                                ReorderFunction<ID, NumNonZeros, Value>> {
  typedef FormatMatcherMixin<ID, NumNonZeros, Value,
                             DegreeReorder<ID, NumNonZeros, Value>,
                             ReorderFunction<ID, NumNonZeros, Value>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  ID *get_reorder(SparseFormat<ID, NumNonZeros, Value> *csr);
};

template <typename ID, typename NumNonZeros, typename Value>
class RCMReorder : public ReorderPreprocessType<ID, NumNonZeros, Value> {
  typedef typename std::make_signed<ID>::type SignedID;

public:
  RCMReorder(float a, float b);

protected:
  struct RCMReorderParams : ReorderParams {
    float alpha;
    float beta;
    RCMReorderParams(float a, float b) : alpha(a), beta(b) {}
  };
  static ID peripheral(NumNonZeros *xadj, ID *adj, ID n, ID start,
                         SignedID *distance, ID *Q);
  static ID *
  get_reorder_csr(std::vector<SparseFormat<ID, NumNonZeros, Value> *> formats,
                  ReorderParams *);
};

template <typename ID, typename NumNonZeros, typename Value>
class RCMReorderInstance
    : public FormatMatcherMixin<ID, NumNonZeros, Value,
                                RCMReorder<ID, NumNonZeros, Value>,
                                ReorderFunction<ID, NumNonZeros, Value>> {
  typedef FormatMatcherMixin<ID, NumNonZeros, Value, RCMReorder<ID, NumNonZeros, Value>,
                             ReorderFunction<ID, NumNonZeros, Value>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  ID *get_reorder(SparseFormat<ID, NumNonZeros, Value> *csr);
};

// template <typename ID, typename NumNonZeros, typename Value, typename ReorderImpl>
template <typename ID, typename NumNonZeros, typename Value,
          template <typename, typename, typename> class ReorderImpl>
class ReorderInstance
    : public FormatMatcherMixin<ID, NumNonZeros, Value,
                                ReorderImpl<ID, NumNonZeros, Value>,
                                ReorderFunction<ID, NumNonZeros, Value>> {
  typedef FormatMatcherMixin<ID, NumNonZeros, Value, ReorderImpl<ID, NumNonZeros, Value>,
                             ReorderFunction<ID, NumNonZeros, Value>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  ID *get_reorder(SparseFormat<ID, NumNonZeros, Value> *csr);
  ID *get_reorder(SparseFormat<ID, NumNonZeros, Value> *csr,
                    ReorderParams *params);
};

// transform
template <typename ID, typename NumNonZeros, typename Value>
using TransformFunction = SparseFormat<ID, NumNonZeros, Value>
    *(*)(std::vector<SparseFormat<ID, NumNonZeros, Value> *>, ID *order);

template <typename ID, typename NumNonZeros, typename Value>
class TransformPreprocessType
    : public MapToFunctionMixin<
          SparseConverterMixin<PreprocessType, ID, NumNonZeros, Value>,
          TransformFunction<ID, NumNonZeros, Value>> {
public:
  virtual ~TransformPreprocessType();
};

template <typename ID, typename NumNonZeros, typename Value>
class Transform : public TransformPreprocessType<ID, NumNonZeros, Value> {
public:
  Transform();

protected:
  static SparseFormat<ID, NumNonZeros, Value> *
  transform_csr(std::vector<SparseFormat<ID, NumNonZeros, Value> *> formats,
                ID *order);
};
// template <typename ID, typename NumNonZeros, typename Value, typename
// TransformImpl>
template <typename ID, typename NumNonZeros, typename Value,
          template <typename, typename, typename> class TransformImpl>
class TransformInstance
    : public FormatMatcherMixin<ID, NumNonZeros, Value,
                                TransformImpl<ID, NumNonZeros, Value>,
                                TransformFunction<ID, NumNonZeros, Value>> {
  typedef FormatMatcherMixin<ID, NumNonZeros, Value,
                             TransformImpl<ID, NumNonZeros, Value>,
                             TransformFunction<ID, NumNonZeros, Value>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  SparseFormat<ID, NumNonZeros, Value> *
  get_transformation(SparseFormat<ID, NumNonZeros, Value> *csr, ID *order);
};

} // namespace sparsebase

#endif
