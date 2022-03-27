#ifndef _Reorder_HPP
#define _Reorder_HPP
#include "config.h"
#include "sparse_converter.h"
#include "sparse_format.h"
#include "sparse_object.h"
#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace sparsebase {

namespace preprocess {

struct TypeIndexVectorHash {
  std::size_t operator()(const std::vector<std::type_index> &vf) const;
};
struct PreprocessParams {};
class PreprocessType {
  protected:
  std::unique_ptr<PreprocessParams> params_;
};

template <class Parent>
class ConverterMixin : public Parent {
  using Parent::Parent;

protected:
  std::unique_ptr<utils::Converter> sc_;

public:
  void SetConverter(const utils::Converter &new_sc);
  void ResetConverter();
};

template <typename ReturnType,
          class PreprocessingImpl,
          typename Key = std::vector<std::type_index>,
          typename KeyHash = TypeIndexVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class FunctionMatcherMixin : public PreprocessingImpl {

  using PreprocessFunction =
      ReturnType (*)(std::vector<format::Format *>, PreprocessParams *);
  
  typedef std::unordered_map<Key, PreprocessFunction, KeyHash,
                             KeyEqualTo>
      ConversionMap;

public:
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                  const PreprocessFunction &func_ptr);
  void RegisterFunction(const Key &key_of_function,
                        const PreprocessFunction &func_ptr);
  bool UnregisterFunction(const Key &key_of_function);

protected:
  using PreprocessingImpl::PreprocessingImpl;
  ConversionMap _map_to_function;
  std::tuple<PreprocessFunction, utils::ConversionSchemaConditional>
  GetFunction(std::vector<format::Format *>packed_sfs, Key key, ConversionMap map, std::vector<context::Context*>,
              utils::Converter &sc);
  bool CheckIfKeyMatches(ConversionMap map, Key key, std::vector<format::Format*> packed_sfs, std::vector<context::Context*> contexts);
  template <typename F> std::vector<std::type_index> PackFormats(F sf);
  template <typename F, typename... SF>
  std::vector<std::type_index> PackFormats(F sf, SF... sfs);
  template <typename F> std::vector<F> PackSFS(F sf);
  template <typename F, typename... SF> std::vector<F> PackSFS(F sf, SF... sfs);
  template <typename F, typename... SF>
  ReturnType Execute(PreprocessParams *params, utils::Converter& sc, std::vector<context::Context*> contexts, F sf,
          SF... sfs);
};

template <typename IDType, typename NNZType, typename ValueType>
class ReorderPreprocessType
    : public FunctionMatcherMixin<IDType*, ConverterMixin<PreprocessType>> {
protected:

public:
  IDType *GetReorder(format::Format *csr, std::vector<context::Context*>);
  IDType *GetReorder(format::Format *csr, PreprocessParams  *params, std::vector<context::Context*>);
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
  static IDType *CalculateReorderCSR(std::vector<format::Format *> formats,
                                     PreprocessParams  *params);
};

template <typename IDType, typename NNZType, typename ValueType>
class GenericReorder
    : public ReorderPreprocessType<IDType, NNZType, ValueType> {
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
  GetReorderCSR(std::vector<format::Format *> formats,
                  PreprocessParams *);
};

// transform
//template <typename IDType, typename NNZType, typename ValueType, typename ReturnType>
//using TransformFunction = ReturnType (*)(std::vector<SparseFormat<IDType, NNZType, ValueType> *>, IDType *order);

template <typename IDType, typename NNZType, typename ValueType>
class TransformPreprocessType
    : public FunctionMatcherMixin<format::Format*,
          ConverterMixin<PreprocessType>> {
public:
  format::Format *
  GetTransformation(format::Format *csr, std::vector<context::Context*>);
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
  static format::Format *
  TransformCSR(std::vector<format::Format *> formats,
                PreprocessParams*);
};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class JaccardWeights : 
    public FunctionMatcherMixin<format::Format*, ConverterMixin<PreprocessType>> {
    struct JaccardParams : PreprocessParams{};

public:
    JaccardWeights ();
    format::Format * GetJaccardWeights(format::Format *format, std::vector<context::Context*>);
    #ifdef CUDA
    static format::Format* GetJaccardWeightCUDACSR(std::vector<format::Format *> formats, PreprocessParams  * params);
    #endif
    ~JaccardWeights ();

protected:
    //GetDegreeDistributionCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats, FeatureParams *);
    //static float * GetDegreeDistributionCSR(SparseObject<IDtype, NNZType, ValueType> * obj);
};
template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class DegreeDistribution : 
    public FunctionMatcherMixin<FeatureType*, ConverterMixin<PreprocessType>> {
    struct DegreeDistributionParams : PreprocessParams{};

public:
    DegreeDistribution();
    FeatureType * GetDistribution(format::Format *format, std::vector<context::Context*>);
    FeatureType * GetDistribution(object::Graph<IDType, NNZType, ValueType> *object, std::vector<context::Context*>);
    //FeatureType * GetDistribution(SparseObject<IDType, NNZType, ValueType> *object);
    static FeatureType * GetDegreeDistributionCSR(std::vector<format::Format *> formats, PreprocessParams  * params);
    ~DegreeDistribution();

protected:
    //GetDegreeDistributionCSR(std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats, FeatureParams *);
    //static float * GetDegreeDistributionCSR(SparseObject<IDtype, NNZType, ValueType> * obj);
};
} // namespace preprocess

} // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparse_preprocess.cc"
#ifdef CUDA
#include "cuda/preprocess.cu"
#endif
#endif

#endif
