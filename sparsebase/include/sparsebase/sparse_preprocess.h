#ifndef _Reorder_HPP
#define _Reorder_HPP
#include "sparse_converter.h"
#include "sparse_format.h"
#include "sparse_object.h"
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sparsebase {

namespace preprocess {

struct FormatVectorHash {
  std::size_t operator()(std::vector<Format> vf) const;
};
struct PreprocessParams {};
class PreprocessType {
  protected:
  std::unique_ptr<PreprocessParams> params_;
};

template <class Parent, typename IDType, typename NNZType, typename ValueType>
class SparseConverterMixin : public Parent {
  using Parent::Parent;

protected:
  utils::SparseConverter<IDType, NNZType, ValueType> sc_;

public:
  void SetConverter(const utils::SparseConverter<IDType, NNZType, ValueType> &new_sc);
  void ResetConverter();
};

template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Key = std::vector<Format>,
          typename KeyHash = FormatVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<Format>>>
class FunctionMatcherMixin : public PreprocessingImpl {

  using PreprocessFunction =
      ReturnType (*)(std::vector<SparseFormat<IDType, NNZType, ValueType> *>, PreprocessParams *);
  
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
  ConversionMap _map_to_function;
  std::tuple<PreprocessFunction, utils::ConversionSchema>
  GetFunction(Key key, ConversionMap map,
               utils::SparseConverter<IDType, NNZType, ValueType>& sc);
  template <typename F> std::vector<Format> PackFormats(F sf);
  template <typename F, typename... SF>
  std::vector<Format> PackFormats(F sf, SF... sfs);
  template <typename F> std::vector<F> PackSFS(F sf);
  template <typename F, typename... SF>
  std::vector<F> PackSFS(F sf, SF... sfs);
  template <typename F, typename... SF>
  ReturnType Execute(PreprocessParams *params, utils::SparseConverter<IDType, NNZType, ValueType>& sc, F sf,
          SF... sfs);
};

//template <typename IDType, typename NNZType, typename ValueType>
//using ReorderFunction =
//    IDType *(*)(std::vector<SparseFormat<IDType, NNZType, ValueType> *>, ReorderParams *);

template <typename IDType, typename NNZType, typename ValueType>
class ReorderPreprocessType
    : public FunctionMatcherMixin<IDType, NNZType, ValueType, IDType*, SparseConverterMixin<PreprocessType,
        IDType, NNZType, ValueType>> {
protected:

public:
  IDType *GetReorder(SparseFormat<IDType, NNZType, ValueType> *csr);
  IDType *GetReorder(SparseFormat<IDType, NNZType, ValueType> *csr,
                    PreprocessParams *params);
  virtual ~ReorderPreprocessType();
};

template <typename IDType, typename NNZType, typename ValueType>
class DegreeReorder : public ReorderPreprocessType<IDType, NNZType, ValueType> {
public:
  DegreeReorder(int hyperparameter);
protected:
  struct DegreeReorderParams : PreprocessParams {
    int hyperparameter;
    DegreeReorderParams(int h) : hyperparameter(h) {}
  };
  static IDType *
  CalculateReorderCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
                        PreprocessParams *params);
          
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
  struct RCMReorderParams : PreprocessParams {
    float alpha;
    float beta;
    RCMReorderParams(float a, float b) : alpha(a), beta(b) {}
  };
  static IDType peripheral(NNZType *xadj, IDType *adj, IDType n, IDType start,
                         SignedID *distance, IDType *Q);
  static IDType *
  GetReorderCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
                  PreprocessParams *);
};

// transform
//template <typename IDType, typename NNZType, typename ValueType, typename ReturnType>
//using TransformFunction = ReturnType (*)(std::vector<SparseFormat<IDType, NNZType, ValueType> *>, IDType *order);

template <typename IDType, typename NNZType, typename ValueType>
class TransformPreprocessType
    : public FunctionMatcherMixin< IDType, NNZType, ValueType, SparseFormat<IDType, NNZType, ValueType>*,
          SparseConverterMixin<PreprocessType, IDType, NNZType, ValueType>> {
public:
  SparseFormat<IDType, NNZType, ValueType> *
  GetTransformation(SparseFormat<IDType, NNZType, ValueType> *csr);
  virtual ~TransformPreprocessType();
};

template <typename IDType, typename NNZType, typename ValueType>
class Transform : public TransformPreprocessType<IDType, NNZType, ValueType> {
public:
  Transform(IDType*);
  struct TransformParams : PreprocessParams {
    IDType* order;
    TransformParams(IDType* order):order(order){};
  };

protected:
  static SparseFormat<IDType, NNZType, ValueType> *
  TransformCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
                PreprocessParams*);
};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class DegreeDistribution : 
    public FunctionMatcherMixin<IDType, NNZType, ValueType, FeatureType*, SparseConverterMixin<PreprocessType,
        IDType, NNZType, ValueType>> {
    struct DegreeDistributionParams : PreprocessParams{};

public:
    DegreeDistribution();
    FeatureType * GetDistribution(SparseFormat<IDType, NNZType, ValueType> *format);
    FeatureType * GetDistribution(object::Graph<IDType, NNZType, ValueType> *object);
    //FeatureType * GetDistribution(SparseObject<IDType, NNZType, ValueType> *object);
    static FeatureType * GetDegreeDistributionCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats, PreprocessParams  * params);
    ~DegreeDistribution();

protected:
    //GetDegreeDistributionCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats, FeatureParams *);
    //static float * GetDegreeDistributionCSR(SparseObject<IDtype, NNZType, ValueType> * obj);
};
} // namespace preprocess

} // namespace sparsebase

#endif
