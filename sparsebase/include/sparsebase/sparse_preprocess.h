#ifndef _Reorder_HPP
#define _Reorder_HPP
#include "sparse_converter.h"
#include "sparse_format.h"
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <typeindex>
#include <typeinfo>

namespace sparsebase {

namespace preprocess {

//struct FormatVectorHash {
//  std::size_t operator()(std::vector<Format> vf) const;
//};
struct FormatVectorHash {
  std::size_t operator()(std::vector<std::hash<std::type_index>> vf) const;
};
class PreprocessType {};

template <class Parent, typename IDType, typename NNZType, typename ValueType>
class SparseConverterMixin : public Parent {
  using Parent::Parent;

protected:
  utils::SparseConverter<IDType, NNZType, ValueType> sc_;

public:
  void SetConverter(const utils::SparseConverter<IDType, NNZType, ValueType> &new_sc);
  void ResetConverter();
};

template <typename IDType, typename NNZType, typename ValueType,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key = std::vector<std::type_index>,
          typename KeyHash = FormatVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class FunctionMatcherMixin : public PreprocessingImpl {
  typedef std::unordered_map<Key, PreprocessFunction, KeyHash,
                             KeyEqualTo>
      ConversionMap;
public:
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                     const PreprocessFunction &func_ptr);
  void RegisterFunction(const Key &key_of_function, const PreprocessFunction &func_ptr);
  bool UnregisterFunction(const Key &key_of_function);

protected:
  using PreprocessingImpl::PreprocessingImpl;
  std::unordered_map<Key, PreprocessFunction, KeyHash, KeyEqualTo> _map_to_function;
  std::tuple<PreprocessFunction, utils::ConversionSchema>
  GetFunction(Key key, ConversionMap map,
               utils::SparseConverter<IDType, NNZType, ValueType>& sc);
  template <typename F> std::vector<std::type_index> PackFormats(F sf);
  template <typename F, typename... SF>
  std::vector<std::type_index> PackFormats(F sf, SF... sfs);
  template <typename F> std::vector<F> PackSFS(F sf);
  template <typename F, typename... SF>
  std::vector<F> PackSFS(F sf, SF... sfs);
  template <typename F, typename... SF>
  std::tuple<PreprocessFunction,
             std::vector<Format<IDType, NNZType, ValueType> *>>
  Execute(ConversionMap map, utils::SparseConverter<IDType, NNZType, ValueType>& sc, F sf,
          SF... sfs);
};

struct ReorderParams {};
template <typename IDType, typename NNZType, typename ValueType>
using ReorderFunction =
    IDType *(*)(std::vector<Format<IDType, NNZType, ValueType> *>, ReorderParams *);

template <typename IDType, typename NNZType, typename ValueType>
class ReorderPreprocessType
    : public FunctionMatcherMixin<IDType, NNZType, ValueType, SparseConverterMixin<PreprocessType, IDType, NNZType, ValueType>, ReorderFunction<IDType, NNZType, ValueType>> {
protected:
  std::unique_ptr<ReorderParams> params_;

public:
  IDType *GetReorder(Format<IDType, NNZType, ValueType> *csr);
  IDType *GetReorder(Format<IDType, NNZType, ValueType> *csr,
                    ReorderParams *params);
  virtual ~ReorderPreprocessType();
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
  CalculateReorderCSR(std::vector<Format<IDType, NNZType, ValueType> *> formats,
                        ReorderParams *params);
};

template <typename IDType, typename NNZType, typename ValueType>
class GenericReorder : public ReorderPreprocessType<IDType, NNZType, ValueType> {
public:
  GenericReorder();
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
  GetReorderCSR(std::vector<Format<IDType, NNZType, ValueType> *> formats,
                  ReorderParams *);
};

// transform
template <typename IDType, typename NNZType, typename ValueType>
using TransformFunction = Format<IDType, NNZType, ValueType>
    *(*)(std::vector<Format<IDType, NNZType, ValueType> *>, IDType *order);

template <typename IDType, typename NNZType, typename ValueType>
class TransformPreprocessType
    : public FunctionMatcherMixin< IDType, NNZType, ValueType,
          SparseConverterMixin<PreprocessType, IDType, NNZType, ValueType>,
          TransformFunction<IDType, NNZType, ValueType>> {
public:
  Format<IDType, NNZType, ValueType> *
  GetTransformation(Format<IDType, NNZType, ValueType> *csr, IDType *order);
  virtual ~TransformPreprocessType();
};

template <typename IDType, typename NNZType, typename ValueType>
class Transform : public TransformPreprocessType<IDType, NNZType, ValueType> {
public:
  Transform();

protected:
  static Format<IDType, NNZType, ValueType> *
  TransformCSR(std::vector<Format<IDType, NNZType, ValueType> *> formats,
                IDType *order);
};

} // namespace preprocess

} // namespace sparsebase

#endif
