#ifndef _Reorder_HPP
#define _Reorder_HPP
#include "config.h"
#include "sparse_converter.h"
#include "sparse_format.h"
#include "sparse_object.h"
//#include "sparse_feature.h"
#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <any>

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

class FType {
public:
  virtual std::unordered_map<std::type_index, std::any> Extract(format::Format *format) = 0;
  virtual std::type_index get_feature_id() = 0;
  virtual std::vector<std::type_index> get_sub_ids() = 0;
  virtual std::vector<FType*> get_subs() = 0;
  virtual PreprocessParams get_params() = 0;
  virtual PreprocessParams get_params(std::type_index) = 0;
  virtual void set_params(std::type_index, PreprocessParams) = 0;
  virtual ~FType() = default;
};

template <class Parent, typename IDType, typename NNZType, typename ValueType>
class ConverterMixin : public Parent {
  using Parent::Parent;

protected:
  utils::Converter<IDType, NNZType, ValueType> sc_;

public:
  void SetConverter(const utils::Converter<IDType, NNZType, ValueType> &new_sc);
  void ResetConverter();
};

template<typename ReturnType>
using PreprocessFunction =
ReturnType (*)(std::vector<format::Format *>, PreprocessParams *);

template <typename IDType, typename NNZType, typename ValueType, typename ReturnType,
          class PreprocessingImpl,
          typename Function = PreprocessFunction<ReturnType>,
          typename Key = std::vector<std::type_index>,
          typename KeyHash = TypeIndexVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class FunctionMatcherMixin : public PreprocessingImpl {


  typedef std::unordered_map<Key, Function, KeyHash,
                             KeyEqualTo>
      ConversionMap;

public:
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                  const Function &func_ptr);
  void RegisterFunction(const Key &key_of_function,
                        const Function &func_ptr);
  bool UnregisterFunction(const Key &key_of_function);

protected:
  using PreprocessingImpl::PreprocessingImpl;
  ConversionMap _map_to_function;
  std::tuple<Function, utils::ConversionSchema>
  GetFunction(Key key, ConversionMap map,
              utils::Converter<IDType, NNZType, ValueType> &sc);
  template <typename F> std::vector<std::type_index> PackFormats(F sf);
  template <typename F, typename... SF>
  std::vector<std::type_index> PackFormats(F sf, SF... sfs);
  template <typename F> std::vector<F> PackSFS(F sf);
  template <typename F, typename... SF> std::vector<F> PackSFS(F sf, SF... sfs);
  template <typename F, typename... SF>
  std::tuple<Function, std::vector<format::Format*>> Execute(utils::Converter<IDType, NNZType, ValueType>& sc, F sf,
          SF... sfs);
};

template <typename IDType, typename NNZType, typename ValueType>
class ReorderPreprocessType
    : public FunctionMatcherMixin<IDType, NNZType, ValueType, IDType*, ConverterMixin<PreprocessType,
        IDType, NNZType, ValueType>> {
protected:

public:
  IDType *GetReorder(format::Format *csr);
  IDType *GetReorder(format::Format *csr, PreprocessParams  *params);
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
    : public FunctionMatcherMixin< IDType, NNZType, ValueType, format::Format*,
          ConverterMixin<PreprocessType, IDType, NNZType, ValueType>> {
public:
  format::Format *
  GetTransformation(format::Format *csr);
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

template< typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeaturePreprocessType : 
    public FunctionMatcherMixin<IDType, NNZType, ValueType, FeatureType, ConverterMixin<FType,
        IDType, NNZType, ValueType>>{
public:
~FeaturePreprocessType();
};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class DegreeDistribution : public FeaturePreprocessType<IDType, NNZType, ValueType, FeatureType*> {
    struct DegreeDistributionParams : PreprocessParams{};

public:
    DegreeDistribution();
    DegreeDistribution(const DegreeDistribution &);
    DegreeDistribution(const DegreeDistributionParams);
    virtual std::unordered_map<std::type_index, std::any> Extract(format::Format * format);
    virtual PreprocessParams get_params();
    virtual PreprocessParams get_params(std::type_index);
    virtual void set_params(std::type_index, PreprocessParams);
    virtual std::type_index get_feature_id();
    virtual std::vector<std::type_index> get_sub_ids();
    virtual std::vector<FType*> get_subs();
    static std::type_index get_feature_id_static();

    FeatureType * GetDistribution(format::Format *format);
    FeatureType * GetDistribution(object::Graph<IDType, NNZType, ValueType> *object);
    static FeatureType * GetDegreeDistributionCSR(std::vector<format::Format *> formats, PreprocessParams  * params);
    ~DegreeDistribution();

protected:
    void Register();
    DegreeDistributionParams params_;
    std::unordered_map<std::type_index, PreprocessParams> pmap_;
};

template<typename IDType, typename NNZType, typename ValueType>
class Degrees : public FeaturePreprocessType<IDType, NNZType, ValueType, IDType*> {
  struct DegreesParams : PreprocessParams{};

public:
  Degrees();
  Degrees(const Degrees<IDType, NNZType, ValueType> & d);
  Degrees(const DegreesParams);
  virtual std::unordered_map<std::type_index, std::any> Extract(format::Format * format);
  virtual PreprocessParams get_params();
  virtual PreprocessParams get_params(std::type_index);
  virtual void set_params(std::type_index, PreprocessParams);
  virtual std::type_index get_feature_id();
  virtual std::vector<std::type_index> get_sub_ids();
  virtual std::vector<FType*> get_subs();
  static std::type_index get_feature_id_static();

  IDType * GetDegrees(format::Format *format);
  static IDType * GetDegreesCSR(std::vector<format::Format *> formats, PreprocessParams  * params);
  ~Degrees();

protected:
  void Register();
  DegreesParams params_;
  std::unordered_map<std::type_index, PreprocessParams> pmap_;
};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Degrees_DegreeDistribution : public FeaturePreprocessType<IDType, NNZType, ValueType, std::unordered_map<std::type_index, std::any>> {
  struct Params : PreprocessParams{};
public:
  Degrees_DegreeDistribution();
  virtual std::unordered_map<std::type_index, std::any> Extract(format::Format * format);
  virtual PreprocessParams get_params();
  virtual PreprocessParams get_params(std::type_index);
  virtual void set_params(std::type_index, PreprocessParams);
  virtual std::vector<std::type_index> get_sub_ids();
  virtual std::vector<FType*> get_subs();
  virtual std::type_index get_feature_id();
  static std::type_index get_feature_id_static();

  std::unordered_map<std::type_index, std::any> Get(format::Format *format);
  static std::unordered_map<std::type_index, std::any> GetCSR(std::vector<format::Format *> formats, PreprocessParams  * params);
  ~Degrees_DegreeDistribution();
protected:
  Params params_;
  std::unordered_map<std::type_index, PreprocessParams> pmap_;
};

} // namespace preprocess

} // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparse_preprocess.cc"
#endif

#endif
