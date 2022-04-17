#ifndef SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
#define SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/format/format.h"
#include "sparsebase/object/object.h"
#include "sparsebase/utils/converter/converter.h"

#include <any>
#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace sparsebase::preprocess {

struct TypeIndexVectorHash {
  std::size_t operator()(const std::vector<std::type_index> &vf) const;
};
struct PreprocessParams {};
class PreprocessType {
protected:
  std::unique_ptr<PreprocessParams> params_;
};

class ExtractableType {
public:
  virtual std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>) = 0;
  virtual std::type_index get_feature_id() = 0;
  virtual std::vector<std::type_index> get_sub_ids() = 0;
  virtual std::vector<ExtractableType *> get_subs() = 0;
  virtual std::shared_ptr<PreprocessParams> get_params() = 0;
  virtual std::shared_ptr<PreprocessParams> get_params(std::type_index) = 0;
  virtual void set_params(std::type_index,
                          std::shared_ptr<PreprocessParams>) = 0;
  virtual ~ExtractableType() = default;

protected:
  std::shared_ptr<PreprocessParams> params_;
  std::unordered_map<std::type_index, std::shared_ptr<PreprocessParams>> pmap_;
};

template <class Parent> class ConverterMixin : public Parent {
  using Parent::Parent;

protected:
  std::unique_ptr<utils::converter::Converter> sc_;

public:
  void SetConverter(const utils::converter::Converter &new_sc);
  void ResetConverter();
};

template <typename ReturnType>
using PreprocessFunction = ReturnType (*)(std::vector<format::Format *>,
                                          PreprocessParams *);

template <typename ReturnType,
          class PreprocessingImpl = ConverterMixin<PreprocessType>,
          typename Function = PreprocessFunction<ReturnType>,
          typename Key = std::vector<std::type_index>,
          typename KeyHash = TypeIndexVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class FunctionMatcherMixin : public PreprocessingImpl {

  typedef std::unordered_map<Key, Function, KeyHash, KeyEqualTo> ConversionMap;

public:
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                  const Function &func_ptr);
  void RegisterFunction(const Key &key_of_function, const Function &func_ptr);
  bool UnregisterFunction(const Key &key_of_function);

protected:
  using PreprocessingImpl::PreprocessingImpl;
  ConversionMap _map_to_function;
  std::tuple<Function, utils::converter::ConversionSchemaConditional>
  GetFunction(std::vector<format::Format *> packed_sfs, Key key,
              ConversionMap map, std::vector<context::Context *>,
              utils::converter::Converter &sc);
  bool CheckIfKeyMatches(ConversionMap map, Key key,
                         std::vector<format::Format *> packed_sfs,
                         std::vector<context::Context *> contexts);
  template <typename F> std::vector<std::type_index> PackFormats(F sf);
  template <typename F, typename... SF>
  std::vector<std::type_index> PackFormats(F sf, SF... sfs);
  template <typename F> std::vector<F> PackSFS(F sf);
  template <typename F, typename... SF> std::vector<F> PackSFS(F sf, SF... sfs);
  template <typename F, typename... SF>
  ReturnType Execute(PreprocessParams *params, utils::converter::Converter &sc,
                     std::vector<context::Context *> contexts, F sf, SF... sfs);
  template <typename F, typename... SF>
  std::tuple<std::vector<format::Format *>, ReturnType>
  CachedExecute(PreprocessParams *params, utils::converter::Converter &sc,
                std::vector<context::Context *> contexts, F sf, SF... sfs);
};

template <typename IDType, typename NNZType, typename ValueType>
class ReorderPreprocessType : public FunctionMatcherMixin<IDType *> {
protected:
public:
  IDType *GetReorder(format::Format *csr, std::vector<context::Context *>);
  IDType *GetReorder(format::Format *csr, PreprocessParams *params,
                     std::vector<context::Context *>);
  std::tuple<std::vector<format::Format *>, IDType *>
  GetReorderCached(format::Format *csr, std::vector<context::Context *>);
  std::tuple<std::vector<format::Format *>, IDType *>
  GetReorderCached(format::Format *csr, PreprocessParams *params,
                   std::vector<context::Context *>);
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
                                     PreprocessParams *params);
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
  static IDType *GetReorderCSR(std::vector<format::Format *> formats,
                               PreprocessParams *);
};

// transform
// template <typename IDType, typename NNZType, typename ValueType, typename
// ReturnType> using TransformFunction = ReturnType
// (*)(std::vector<SparseFormat<IDType, NNZType, ValueType> *>, IDType *order);

template <typename IDType, typename NNZType, typename ValueType>
class TransformPreprocessType : public FunctionMatcherMixin<format::Format *> {
public:
  format::Format *GetTransformation(format::Format *csr,
                                    std::vector<context::Context *>);
  std::tuple<std::vector<format::Format *>, format::Format *>
  GetTransformationCached(format::Format *csr, std::vector<context::Context *>);
  virtual ~TransformPreprocessType();
};

template <typename IDType, typename NNZType, typename ValueType>
class Transform : public TransformPreprocessType<IDType, NNZType, ValueType> {
public:
  Transform(IDType *);
  struct TransformParams : PreprocessParams {
    IDType *order;
    explicit TransformParams(IDType *order) : order(order){};
  };

protected:
  static format::Format *TransformCSR(std::vector<format::Format *> formats,
                                      PreprocessParams *);
};

template <typename FeatureType>
class FeaturePreprocessType
    : public FunctionMatcherMixin<FeatureType,
                                  ConverterMixin<ExtractableType>> {
public:
  std::shared_ptr<PreprocessParams> get_params() override;
  std::shared_ptr<PreprocessParams> get_params(std::type_index) override;
  void set_params(std::type_index, std::shared_ptr<PreprocessParams>) override;
  std::type_index get_feature_id() override;
  ~FeaturePreprocessType();

protected:
  std::shared_ptr<PreprocessParams> params_;
  std::unordered_map<std::type_index, std::shared_ptr<PreprocessParams>> pmap_;
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class JaccardWeights : public FunctionMatcherMixin<format::Format *> {
  struct JaccardParams : PreprocessParams {};

public:
  JaccardWeights();
  format::Format *GetJaccardWeights(format::Format *format,
                                    std::vector<context::Context *>);
#ifdef CUDA
  static format::Format *
  GetJaccardWeightCUDACSR(std::vector<format::Format *> formats,
                          PreprocessParams *params);
#endif
  ~JaccardWeights();

protected:
  // GetDegreeDistributionCSR(std::vector<SparseFormat<IDType, NNZType,
  // ValueType> *> formats, FeatureParams *); static float *
  // GetDegreeDistributionCSR(SparseObject<IDtype, NNZType, ValueType> * obj);
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class DegreeDistribution : public FeaturePreprocessType<FeatureType *> {
  struct DegreeDistributionParams : PreprocessParams {};

public:
  DegreeDistribution();
  DegreeDistribution(const DegreeDistribution &);
  DegreeDistribution(std::shared_ptr<DegreeDistributionParams>);
  virtual std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>);
  virtual std::vector<std::type_index> get_sub_ids();
  virtual std::vector<ExtractableType *> get_subs();
  static std::type_index get_feature_id_static();

  FeatureType *GetDistribution(format::Format *format,
                               std::vector<context::Context *>);
  FeatureType *
  GetDistribution(object::Graph<IDType, NNZType, ValueType> *object,
                  std::vector<context::Context *>);
  std::tuple<std::vector<format::Format *>, FeatureType *>
  GetDistributionCached(format::Format *format,
                        std::vector<context::Context *>);

  static FeatureType *
  GetDegreeDistributionCSR(std::vector<format::Format *> formats,
                           PreprocessParams *params);
  ~DegreeDistribution();

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class Degrees : public FeaturePreprocessType<IDType *> {
  struct DegreesParams : PreprocessParams {};

public:
  Degrees();
  Degrees(const Degrees<IDType, NNZType, ValueType> &d);
  Degrees(std::shared_ptr<DegreesParams>);
  std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<ExtractableType *> get_subs() override;
  static std::type_index get_feature_id_static();

  IDType *GetDegrees(format::Format *format, std::vector<context::Context *>);
  static IDType *GetDegreesCSR(std::vector<format::Format *> formats,
                               PreprocessParams *params);
  ~Degrees();

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class Degrees_DegreeDistribution
    : public FeaturePreprocessType<
          std::unordered_map<std::type_index, std::any>> {
  struct Params : PreprocessParams {};

public:
  Degrees_DegreeDistribution();
  std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<ExtractableType *> get_subs() override;
  static std::type_index get_feature_id_static();

  std::unordered_map<std::type_index, std::any>
  Get(format::Format *format, std::vector<context::Context *>);
  static std::unordered_map<std::type_index, std::any>
  GetCSR(std::vector<format::Format *> formats, PreprocessParams *params);
  ~Degrees_DegreeDistribution();

protected:
};

} // namespace sparsebase::preprocess
#ifdef _HEADER_ONLY
#include "sparsebase/preprocess/preprocess.cc"
#ifdef CUDA
#include "cuda/preprocess.cu"
#endif
#endif

#endif // SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
