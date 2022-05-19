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

//! Functor used for hashing vectors of type_index values.
struct TypeIndexVectorHash {
  std::size_t operator()(const std::vector<std::type_index> &vf) const;
};

//! An abstraction for parameter objects used for preprocessing
struct PreprocessParams {};

//! A generic type for all preprocessing types
class PreprocessType {
protected:
  //! Polymorphic pointer at a PreprocessParams object
  std::unique_ptr<PreprocessParams> params_;
};

//! Abstract class that can be utilized with fusued feature extraction
/*!
 * Classes implementing ExtractableType can be used with with a sparsebase::feature::Extractor for fused feature extraction.
 * Each ExtractableType object can be a fusion of multiple ExtractableType classes. An ExtractableType object will contain 
 * parameters for each of the ExtractableType it is fusud into as well as one for itself.
 */
class ExtractableType {
public:
  //! Extract features from the passed Format through passed Contexts
  /*!
   *
   * \param format object from which features are extracted.
   * \param contexts vector of contexts that can be used for extracting features.
   * \return An uordered map containing the extracted features as key-value pairs with the key being the std::type_index of the feature and the value an std::any to that feature.
   */
  virtual std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *> contexts) = 0;
  //! Returns the std::type_index of this class
  virtual std::type_index get_feature_id() = 0;
  //! Get the std::type_index of all the ExtractableType classes fused into this class 
  /*!
   *
   * \return a vector containing the std::type_index values of all the ExtractableType classes fusued into this class
   */
  virtual std::vector<std::type_index> get_sub_ids() = 0;
  //! Get instances of the ExtractableType classes that make up this class
  /*!
   * \return A vector of pointers to ExtractableType objects, each of which corresponds to one of the features that this class is extracting, and the classes will have their respective parameters passed over to them.
   */ 
  virtual std::vector<ExtractableType *> get_subs() = 0;
  //! Get a std::shared_ptr at the PreprocessParams of this object
  /*!
   *
   * \return An std::shared_ptr at the same PreprocessParams instance of this object (not a copy)
   */
  virtual std::shared_ptr<PreprocessParams> get_params() = 0;
  //! Get an std::shared_ptr at a PreprocessParams of one of the ExtractableType classes fused into this class
  /*!
   * Returns a std::shared_ptr at a PreprocessParams object belonging to one of the ExtractableType classes fused into this class
   * \param feature_extractor std::type_index identifying the ExtractableType within this class whose parameters are requested
   * \return an std::shared_ptr at the PreprocessParams corresponding feature_extractor
   */ 
  virtual std::shared_ptr<PreprocessParams> get_params(std::type_index feature_extractor) = 0;
  //! Set the parameters of one of ExtractableType classes fusued into this classes.
  /*!
   * \param feature_extractor std::type_index identifying the ExtractableType class fusued into this class whose parameters are to be set.
   * \param params an std::shared_ptr at the PreprocessParams belonging to the class feature_extractor
   */
  virtual void set_params(std::type_index feature_extractor,
                          std::shared_ptr<PreprocessParams> params) = 0;
  virtual ~ExtractableType() = default;

protected:
  //! a pointer at the PreprocessParams of this class
  std::shared_ptr<PreprocessParams> params_;
  //! A key-value map of PreprocessParams, one for each of the ExtractableType classes fused into this class
  std::unordered_map<std::type_index, std::shared_ptr<PreprocessParams>> pmap_;
};

//! A mixin class that attaches to its templated parameter a sparsebase::utils::converter::Converter
/*!
 *
 * @tparam Parent any class to which a converter should be added
 */
template <class Parent> class ConverterMixin : public Parent {
  using Parent::Parent;

protected:
  //! A unique pointer at an abstract sparsebase::utils::converter::Converter object
  std::unique_ptr<utils::converter::Converter> sc_ = nullptr;

public:
  //! Set the data member `sc_` to be a clone of `new_sc`
  /*!
   * @param new_sc a reference to a Converter object
   */
  void SetConverter(const utils::converter::Converter &new_sc);
  //! Resets the concrete converter pointed at by `sc_` to its initial state
  void ResetConverter();
  //! Returns a unique pointer at a copy of the current Converter pointed to by `new_sc`
  std::unique_ptr<utils::converter::Converter> GetConverter();
};

//! Template for implementation functions of all preprocesses
/*!
  \tparam ReturnType the return type of preprocessing functions
  \param formats a vector of pointers at format::Format objects
  \param params a polymorphic pointer at a PreprocessParams object
*/
template <typename ReturnType>
using PreprocessFunction = ReturnType (*)(std::vector<format::Format *> formats,
                                          PreprocessParams * params);

//! A mixin that attaches the functionality of matching keys to functions
/*!
  This mixin attaches the functionality of matching keys (which, by default, are vectors of type indices) to function pointer objects (by default, their signature is PreprocessFunction). 
  \tparam ReturnType the return type that will be returned by the preprocessing function implementations
  \tparam Function the function signatures that keys will map to. Default is sparsebase::preprocess::PreprocessFunction
  \tparam Key the type of the keys used to access function in the inner maps. Default is std::vector<std::type_index>>
  \tparam KeyHash the hash function used to has keys.
  \tparam KeyEqualTo the function used to evaluate equality of keys
*/
template <typename ReturnType,
          class PreprocessingImpl = ConverterMixin<PreprocessType>,
          typename Function = PreprocessFunction<ReturnType>,
          typename Key = std::vector<std::type_index>,
          typename KeyHash = TypeIndexVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class FunctionMatcherMixin : public PreprocessingImpl {

  //! Defines a map between `Key` objects and function pointer `Function` objects.
  typedef std::unordered_map<Key, Function, KeyHash, KeyEqualTo> ConversionMap;

public:
  //! Register a key to a function as long as that key isn't already registered 
  /*!
    \param key_of_function key used in the map
    \param func_ptr function pointer being registered
    \return True if the function was registered successfully and false otherwise
  */
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                  const Function &func_ptr);
  //! Register a key to a function and overrides previous registered function (if any) 
  /*!
    \param key_of_function key used in the map
    \param func_ptr function pointer being registered
  */
  void RegisterFunction(const Key &key_of_function, const Function &func_ptr);
  //! Unregister a key from the map if the key was registered to a function
  /*!
    \param key_of_function key to unregister
    \return true if the key was unregistered successfully, and false if it wasn't already registerd to something.
  */
  bool UnregisterFunction(const Key &key_of_function);

protected:
  using PreprocessingImpl::PreprocessingImpl;
  //! Map between `Key` objects and function pointer `Function` objects.
  ConversionMap _map_to_function;
  //! Determines the exact Function and format conversions needed to carry out preprocessing 
  /*!
   * \param packed_formats a vector of the input Format* needed for conversion.
   * \param key the Key representing the input formats.
   * \param map the map between Keys and Functions used to find the needed function
   * \param contexts Contexts available for execution of the preprocessing
   * \param converter Converter object to be used for determining available Format conversions
   * \return a tuple of a) the Function to use,  and b) a utils::converter::ConversionSchemaConditional indicating conversions to be done on input Format objects 
  */
  std::tuple<Function, utils::converter::ConversionSchemaConditional>
  GetFunction(std::vector<format::Format *> packed_formats, Key key,
              ConversionMap map, std::vector<context::Context *> contexts,
              utils::converter::Converter *converter);
  //! Check if a given Key has a function that can be used without any conversions.
  /*!
   * Given a conversion map, available execution contexts, input formats, and a key, determines whether the key has a corresponding function and that the available contexts allow that function to be executed.
   * \param map the map between Keys and Functions used to find the needed function
   * \param key the Key representing the input formats.
   * \param packed_formats a vector of the input Format* needed for conversion.
   * \param contexts Contexts available for execution of the preprocessing
   * \return true if the key has a matching function that can be used with the inputs without any conversions.
   */
  bool CheckIfKeyMatches(ConversionMap map, Key key,
                         std::vector<format::Format *> packed_formats,
                         std::vector<context::Context *> contexts);
  //! A variadic method to pack objects into a vector
  template <typename Object, typename... Objects> std::vector<Object> PackObjects(Object object, Objects... objects);
  //! Base case of a variadic method to pack objects into a vector
  template <typename Object> std::vector<Object> PackObjects(Object object);
  //! Executes preprocessing on input formats (given variadically)
  /*!
   * Determines the function needed to carry out preprocessing on input Format* objects (given variadically), as well as the Format conversions needed on the inputs, executes the preprocessing, and returns the results.  
   * Note: this function will delete any intermediery Format objects that were created due to a conversion.
   * \param PreprocessParams a polymorphic pointer at the object containing hyperparameters needed for preprocessing.
   * \param converter Converter object to be used for determining available Format conversions.
   * \param contexts Contexts available for execution of the preprocessing.
   * \param sf a single input Format* (this is templated to allow variadic definition).
   * \param sfs a variadic Format* (this is templated to allow variadic definition).
   * \return the output of the preprocessing (of type ReturnType). 
   */
  template <typename F, typename... SF>
  ReturnType Execute(PreprocessParams *params, utils::converter::Converter *converter,
                     std::vector<context::Context *> contexts, F sf, SF... sfs);
  //! Executes preprocessing on input formats (given variadically)
  /*!
   * Determines the function needed to carry out preprocessing on input Format* objects (given variadically), as well as the Format conversions needed on the inputs, executes the preprocessing, and returns:
   * - the preprocessing result.  
   * - pointers at any Format objects that were created due to a conversion.
   * Note: this function will delete any intermediery Format objects that were created due to a conversion.
   * \param PreprocessParams a polymorphic pointer at the object containing hyperparameters needed for preprocessing.
   * \param converter Converter object to be used for determining available Format conversions.
   * \param contexts Contexts available for execution of the preprocessing.
   * \param sf a single input Format* (this is templated to allow variadic definition).
   * \param sfs a variadic Format* (this is templated to allow variadic definition).
   * \return a tuple containing a) the output of the preprocessing (of type ReturnType), and b) a vector of Format*, where each pointer in the output points at the format that the corresponds Format object from the the input was converted to. If an input Format wasn't converted, the output pointer will point at nullptr.
   */
  template <typename F, typename... SF>
  std::tuple<std::vector<format::Format *>, ReturnType>
  CachedExecute(PreprocessParams *params, utils::converter::Converter *sc,
                std::vector<context::Context *> contexts, F sf, SF... sfs);
};

template <typename ReturnType>
class GenericPreprocessType : public FunctionMatcherMixin<ReturnType> {
protected:
public:
  int GetOutput(format::Format *csr, PreprocessParams *params,
                     std::vector<context::Context *>);
  std::tuple<std::vector<format::Format *>, int>
  GetOutputCached(format::Format *csr, PreprocessParams *params,
                   std::vector<context::Context *>);
  virtual ~GenericPreprocessType();
};

template <typename IDType>
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
class DegreeReorder : public ReorderPreprocessType<IDType> {
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
    : public ReorderPreprocessType<IDType> {
public:
  GenericReorder();
};

template <typename IDType, typename NNZType, typename ValueType>
class RCMReorder : public ReorderPreprocessType<IDType> {
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

//! A class that does feature extraction.
/*!
 * An ExtractableType class that has a Converter and the function matching capability. In other words, an Extractable to which implementation functions can be added and used.
 * \tparam FeatureType the return type of feature extraction
 */
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

//protected:
//  std::shared_ptr<PreprocessParams> params_;
//  std::unordered_map<std::type_index, std::shared_ptr<PreprocessParams>> pmap_;
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

public:
  struct DegreeDistributionParams : PreprocessParams {};
  DegreeDistribution();
  DegreeDistribution(const DegreeDistribution &);
  DegreeDistribution(std::shared_ptr<DegreeDistributionParams>);
  virtual std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>);
  virtual std::vector<std::type_index> get_sub_ids();
  virtual std::vector<ExtractableType *> get_subs();
  static std::type_index get_feature_id_static();

  //! Degree distribution generation executor function that carries out function matching
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting features.
   * \return an array of size format.get_dimensions()[0] where element i is the degree distribution of the ith vertex in `formats`
   */
  FeatureType *GetDistribution(format::Format *format,
                               std::vector<context::Context *> contexts);
  //! Degree distribution generation executor function that carries out function matching on a Graph
  /*!
   *
   * \param object a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting features.
   * \return an array of size format.get_dimensions()[0] where element i is the degree distribution of the ith vertex in `formats`
   */
  FeatureType *
  GetDistribution(object::Graph<IDType, NNZType, ValueType> *object,
                  std::vector<context::Context *> contexts);
  //! Degree distribution generation executer function that carries out function matching with cached outputs
  /*!
   * Generates the degree distribution of the passed format. If the input format was converted to other format types, the converting results are also returned with the output
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting features.
   * \return A tuple with the first element being a vector of Format*, where each pointer in the output points at the format that the corresponds Format object from the the input was converted to. If an input Format wasn't converted, the output pointer will point at nullptr. The second element is an array of size format.get_dimensions()[0] where element i is the degree distribution of the ith vertex in `formats`
   */
  std::tuple<std::vector<format::Format *>, FeatureType *>
  GetDistributionCached(format::Format *format,
                        std::vector<context::Context *> contexts);

  static FeatureType *
  //! Degree distribution generation implementation function for CSRs
  /*!
   *
   * \param format a single format pointer to any format
   * \return an array of size formats[0].get_dimensions()[0] where element i is the degree distribution of the ith vertex in `formats[0]`
   */
  GetDegreeDistributionCSR(std::vector<format::Format *> formats,
                           PreprocessParams *params);
  ~DegreeDistribution();

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class Degrees : public FeaturePreprocessType<IDType *> {

public:
  struct DegreesParams : PreprocessParams {};
  Degrees();
  Degrees(const Degrees<IDType, NNZType, ValueType> &d);
  Degrees(std::shared_ptr<DegreesParams>);
  std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<ExtractableType *> get_subs() override;
  static std::type_index get_feature_id_static();

  //! Degree generation executer function that carries out function matching
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting features.
   * \return an array of size format.get_dimensions()[0] where element i is the degree of the ith vertex in `format`
   */
  IDType *GetDegrees(format::Format *format, std::vector<context::Context *> contexts);
  //! Degree generation implementation function for CSRs
  /*!
   *
   * \param formats A vector containing a single format pointer that should point at a CSR object
   * \param params a PreprocessParams pointer, though it is not used in the function
   * \return an array of size formats[0].get_dimensions()[0] where element i is the degree of the ith vertex in `formats[0]`
   */
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

  //! Degree and degree distribution generation executor function that carries out function matching
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting features.
   * \return a map with two (type_index, any) pairs. One is a degrees array of type IDType*, and one is a degree distribution array of type FeatureType*. Both arrays have the respective metric of the ith vertex in the ith array element.
   */
  std::unordered_map<std::type_index, std::any>
  Get(format::Format *format, std::vector<context::Context *> contexts);

  //! Degree and degree distribution implementation function for CSRs
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting features.
   * \return a map with two (type_index, any) pairs. One is a degrees array of type IDType*, and one is a degree distribution array of type FeatureType*. Both arrays have the respective metric of the ith vertex in the ith array element.
   */
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
