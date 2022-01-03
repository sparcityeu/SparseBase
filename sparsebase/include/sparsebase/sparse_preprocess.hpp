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
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                     const Function &func_ptr);
  void RegisterFunction(const Key &key_of_function, const Function &func_ptr);
  bool UnregisterFunction(const Key &key_of_function);
};
template <class Parent, typename IDType, typename NNZType, typename ValueType>
class SparseConverterMixin : public Parent {
  using Parent::Parent;

protected:
  SparseConverter<IDType, NNZType, ValueType> sc_;

public:
  void SetConverter(const SparseConverter<IDType, NNZType, ValueType> &new_sc);
  void ResetConverter();
};

struct ReorderParams {};
template <typename IDType, typename NNZType, typename ValueType>
using ReorderFunction =
    IDType *(*)(std::vector<SparseFormat<IDType, NNZType, ValueType> *>, ReorderParams *);

template <typename IDType, typename NNZType, typename ValueType>
class ReorderPreprocessType
    : public MapToFunctionMixin<
          SparseConverterMixin<PreprocessType, IDType, NNZType, ValueType>,
          ReorderFunction<IDType, NNZType, ValueType>> {
protected:
  std::unique_ptr<ReorderParams> params_;

public:
  virtual ~ReorderPreprocessType();
};

template <typename IDType, typename NNZType, typename ValueType,
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
  GetFunction(Key key, ConversionMap map,
               SparseConverter<IDType, NNZType, ValueType> sc);
  template <typename F> std::vector<Format> PackFormats(F sf);
  template <typename F, typename... SF>
  std::vector<Format> PackFormats(F sf, SF... sfs);
  template <typename F> std::vector<F> PackSFS(F sf);
  template <typename F, typename... SF>
  std::vector<F> PackSFS(F sf, SF... sfs);
  template <typename F, typename... SF>
  std::tuple<PreprocessFunction,
             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
  Execute(ConversionMap map, SparseConverter<IDType, NNZType, ValueType> sc, F sf,
          SF... sfs);
};
template <typename IDType, typename NNZType, typename ValueType>
class DegreeReorder : public ReorderPreprocessType<IDType, NNZType, ValueType> {
public:
  DegreeReorder(int hyperparameter);

protected:
  struct DegreeReorderParams : ReorderParams {
    int hyperparameter;
    DegreeReorderParams(int h) : hyperparameter(h) {}
  };
  static IDType *
  CalculateReorderCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
                        ReorderParams *params);
};

template <typename IDType, typename NNZType, typename ValueType>
class GenericReorder : public ReorderPreprocessType<IDType, NNZType, ValueType> {
public:
  GenericReorder();
};
template <typename IDType, typename NNZType, typename ValueType>
class DegreeReorderInstance
    : public FormatMatcherMixin<IDType, NNZType, ValueType,
                                DegreeReorder<IDType, NNZType, ValueType>,
                                ReorderFunction<IDType, NNZType, ValueType>> {
  typedef FormatMatcherMixin<IDType, NNZType, ValueType,
                             DegreeReorder<IDType, NNZType, ValueType>,
                             ReorderFunction<IDType, NNZType, ValueType>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  IDType *GetReorder(SparseFormat<IDType, NNZType, ValueType> *csr);
};

template <typename IDType, typename NNZType, typename ValueType>
class RCMReorder : public ReorderPreprocessType<IDType, NNZType, ValueType> {
  typedef typename std::make_signed<IDType>::type SignedID;

public:
  RCMReorder(float a, float b);

protected:
  struct RCMReorderParams : ReorderParams {
    float alpha;
    float beta;
    RCMReorderParams(float a, float b) : alpha(a), beta(b) {}
  };
  static IDType peripheral(NNZType *xadj, IDType *adj, IDType n, IDType start,
                         SignedID *distance, IDType *Q);
  static IDType *
  GetReorderCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
                  ReorderParams *);
};

template <typename IDType, typename NNZType, typename ValueType>
class RCMReorderInstance
    : public FormatMatcherMixin<IDType, NNZType, ValueType,
                                RCMReorder<IDType, NNZType, ValueType>,
                                ReorderFunction<IDType, NNZType, ValueType>> {
  typedef FormatMatcherMixin<IDType, NNZType, ValueType, RCMReorder<IDType, NNZType, ValueType>,
                             ReorderFunction<IDType, NNZType, ValueType>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  IDType *GetReorder(SparseFormat<IDType, NNZType, ValueType> *csr);
};

// template <typename IDType, typename NNZType, typename ValueType, typename ReorderImpl>
template <typename IDType, typename NNZType, typename ValueType,
          template <typename, typename, typename> class ReorderImpl>
class ReorderInstance
    : public FormatMatcherMixin<IDType, NNZType, ValueType,
                                ReorderImpl<IDType, NNZType, ValueType>,
                                ReorderFunction<IDType, NNZType, ValueType>> {
  typedef FormatMatcherMixin<IDType, NNZType, ValueType, ReorderImpl<IDType, NNZType, ValueType>,
                             ReorderFunction<IDType, NNZType, ValueType>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  IDType *GetReorder(SparseFormat<IDType, NNZType, ValueType> *csr);
  IDType *GetReorder(SparseFormat<IDType, NNZType, ValueType> *csr,
                    ReorderParams *params);
};

// transform
template <typename IDType, typename NNZType, typename ValueType>
using TransformFunction = SparseFormat<IDType, NNZType, ValueType>
    *(*)(std::vector<SparseFormat<IDType, NNZType, ValueType> *>, IDType *order);

template <typename IDType, typename NNZType, typename ValueType>
class TransformPreprocessType
    : public MapToFunctionMixin<
          SparseConverterMixin<PreprocessType, IDType, NNZType, ValueType>,
          TransformFunction<IDType, NNZType, ValueType>> {
public:
  virtual ~TransformPreprocessType();
};

template <typename IDType, typename NNZType, typename ValueType>
class Transform : public TransformPreprocessType<IDType, NNZType, ValueType> {
public:
  Transform();

protected:
  static SparseFormat<IDType, NNZType, ValueType> *
  TransformCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
                IDType *order);
};
// template <typename IDType, typename NNZType, typename ValueType, typename
// TransformImpl>
template <typename IDType, typename NNZType, typename ValueType,
          template <typename, typename, typename> class TransformImpl>
class TransformInstance
    : public FormatMatcherMixin<IDType, NNZType, ValueType,
                                TransformImpl<IDType, NNZType, ValueType>,
                                TransformFunction<IDType, NNZType, ValueType>> {
  typedef FormatMatcherMixin<IDType, NNZType, ValueType,
                             TransformImpl<IDType, NNZType, ValueType>,
                             TransformFunction<IDType, NNZType, ValueType>>
      Base;
  using Base::Base; // Used to forward constructors from base
public:
  SparseFormat<IDType, NNZType, ValueType> *
  GetTransformation(SparseFormat<IDType, NNZType, ValueType> *csr, IDType *order);
};

} // namespace sparsebase

#endif
