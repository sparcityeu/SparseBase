/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
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
  typedef PreprocessParams ParamsType;
protected:
  //! Polymorphic pointer at a PreprocessParams object
  std::unique_ptr<PreprocessParams> params_;
};

//! Abstract class that can be utilized with fusued feature extraction
/*!
 * Classes implementing ExtractableType can be used with with a
 * sparsebase::feature::Extractor for fused feature extraction. Each
 * ExtractableType object can be a fusion of multiple ExtractableType classes.
 * An ExtractableType object will contain parameters for each of the
 * ExtractableType it is fusud into as well as one for itself.
 */
class ExtractableType {
public:
  //! Extract features from the passed Format through passed Contexts
  /*!
   *
   * \param format object from which features are extracted.
   * \param contexts vector of contexts that can be used for extracting
   * features. \return An uordered map containing the extracted features as
   * key-value pairs with the key being the std::type_index of the feature and
   * the value an std::any to that feature.
   */
  virtual std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *> contexts, bool convert_input) = 0;
  //! Returns the std::type_index of this class
  virtual std::type_index get_feature_id() = 0;
  //! Get the std::type_index of all the ExtractableType classes fused into this
  //! class
  /*!
   *
   * \return a vector containing the std::type_index values of all the
   * ExtractableType classes fusued into this class
   */
  virtual std::vector<std::type_index> get_sub_ids() = 0;
  //! Get instances of the ExtractableType classes that make up this class
  /*!
   * \return A vector of pointers to ExtractableType objects, each of which
   * corresponds to one of the features that this class is extracting, and the
   * classes will have their respective parameters passed over to them.
   */
  virtual std::vector<ExtractableType *> get_subs() = 0;
  //! Get a std::shared_ptr at the PreprocessParams of this object
  /*!
   *
   * \return An std::shared_ptr at the same PreprocessParams instance of this
   * object (not a copy)
   */
  virtual std::shared_ptr<PreprocessParams> get_params() = 0;
  //! Get an std::shared_ptr at a PreprocessParams of one of the ExtractableType
  //! classes fused into this class
  /*!
   * Returns a std::shared_ptr at a PreprocessParams object belonging to one of
   * the ExtractableType classes fused into this class \param feature_extractor
   * std::type_index identifying the ExtractableType within this class whose
   * parameters are requested \return an std::shared_ptr at the PreprocessParams
   * corresponding feature_extractor
   */
  virtual std::shared_ptr<PreprocessParams>
  get_params(std::type_index feature_extractor) = 0;
  //! Set the parameters of one of ExtractableType classes fusued into this
  //! classes.
  /*!
   * \param feature_extractor std::type_index identifying the ExtractableType
   * class fusued into this class whose parameters are to be set. \param params
   * an std::shared_ptr at the PreprocessParams belonging to the class
   * feature_extractor
   */
  virtual void set_params(std::type_index feature_extractor,
                          std::shared_ptr<PreprocessParams> params) = 0;
  virtual ~ExtractableType() = default;

protected:
  //! a pointer at the PreprocessParams of this class
  std::shared_ptr<PreprocessParams> params_;
  //! A key-value map of PreprocessParams, one for each of the ExtractableType
  //! classes fused into this class
  std::unordered_map<std::type_index, std::shared_ptr<PreprocessParams>> pmap_;
};

//! A mixin class that attaches to its templated parameter a
//! sparsebase::utils::converter::Converter
/*!
 *
 * @tparam Parent any class to which a converter should be added
 */
template <class Parent> class ConverterMixin : public Parent {
  using Parent::Parent;

protected:
  //! A unique pointer at an abstract sparsebase::utils::converter::Converter
  //! object
  std::unique_ptr<utils::converter::Converter> sc_ = nullptr;

public:
  //! Set the data member `sc_` to be a clone of `new_sc`
  /*!
   * @param new_sc a reference to a Converter object
   */
  void SetConverter(const utils::converter::Converter &new_sc);
  //! Resets the concrete converter pointed at by `sc_` to its initial state
  void ResetConverter();
  //! Returns a unique pointer at a copy of the current Converter pointed to by
  //! `new_sc`
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
                                          PreprocessParams *params);

//! A mixin that attaches the functionality of matching keys to functions
/*!
  This mixin attaches the functionality of matching keys (which, by default, are
  vectors of type indices) to function pointer objects (by default, their
  signature is PreprocessFunction). \tparam ReturnType the return type that will
  be returned by the preprocessing function implementations \tparam Function the
  function signatures that keys will map to. Default is
  sparsebase::preprocess::PreprocessFunction \tparam Key the type of the keys
  used to access function in the inner maps. Default is
  std::vector<std::type_index>> \tparam KeyHash the hash function used to has
  keys. \tparam KeyEqualTo the function used to evaluate equality of keys
*/
template <typename ReturnType,
          class PreprocessingImpl = ConverterMixin<PreprocessType>,
          typename Function = PreprocessFunction<ReturnType>,
          typename Key = std::vector<std::type_index>,
          typename KeyHash = TypeIndexVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class FunctionMatcherMixin : public PreprocessingImpl {

  //! Defines a map between `Key` objects and function pointer `Function`
  //! objects.
  typedef std::unordered_map<Key, Function, KeyHash, KeyEqualTo> ConversionMap;

public:
  std::vector<Key> GetAvailableFormats(){
    std::vector<Key> keys;
    for (auto element : map_to_function_){
      keys.push_back(element.first);
    }
    return keys;
  }
  //! Register a key to a function as long as that key isn't already registered
  /*!
    \param key_of_function key used in the map
    \param func_ptr function pointer being registered
    \return True if the function was registered successfully and false otherwise
  */
  bool RegisterFunctionNoOverride(const Key &key_of_function,
                                  const Function &func_ptr);
  //! Register a key to a function and overrides previous registered function
  //! (if any)
  /*!
    \param key_of_function key used in the map
    \param func_ptr function pointer being registered
  */
  void RegisterFunction(const Key &key_of_function, const Function &func_ptr);
  //! Unregister a key from the map if the key was registered to a function
  /*!
    \param key_of_function key to unregister
    \return true if the key was unregistered successfully, and false if it
    wasn't already registerd to something.
  */
  bool UnregisterFunction(const Key &key_of_function);

protected:
  using PreprocessingImpl::PreprocessingImpl;
  //! Map between `Key` objects and function pointer `Function` objects.
  ConversionMap map_to_function_;
  //! Determines the exact Function and format conversions needed to carry out
  //! preprocessing
  /*!
   * \param packed_formats a vector of the input Format* needed for conversion.
   * \param key the Key representing the input formats.
   * \param map the map between Keys and Functions used to find the needed
   * function \param contexts Contexts available for execution of the
   * preprocessing \param converter Converter object to be used for determining
   * available Format conversions \return a tuple of a) the Function to use, and
   * b) a utils::converter::ConversionSchemaConditional indicating conversions
   * to be done on input Format objects
   */
  std::tuple<Function, utils::converter::ConversionSchemaConditional>
  GetFunction(std::vector<format::Format *> packed_formats, Key key,
              ConversionMap map, std::vector<context::Context *> contexts,
              utils::converter::Converter *converter);
  //! Check if a given Key has a function that can be used without any
  //! conversions.
  /*!
   * Given a conversion map, available execution contexts, input formats, and a
   * key, determines whether the key has a corresponding function and that the
   * available contexts allow that function to be executed. \param map the map
   * between Keys and Functions used to find the needed function \param key the
   * Key representing the input formats. \param packed_formats a vector of the
   * input Format* needed for conversion. \param contexts Contexts available for
   * execution of the preprocessing \return true if the key has a matching
   * function that can be used with the inputs without any conversions.
   */
  bool CheckIfKeyMatches(ConversionMap map, Key key,
                         std::vector<format::Format *> packed_formats,
                         std::vector<context::Context *> contexts);
  //! A variadic method to pack objects into a vector
  template <typename Object, typename... Objects>
  std::vector<Object> PackObjects(Object object, Objects... objects);
  //! Base case of a variadic method to pack objects into a vector
  template <typename Object> std::vector<Object> PackObjects(Object object);
  //! Executes preprocessing on input formats (given variadically)
  /*!
   * Determines the function needed to carry out preprocessing on input Format*
   * objects (given variadically), as well as the Format conversions needed on
   * the inputs, executes the preprocessing, and returns the results. Note: this
   * function will delete any intermediery Format objects that were created due
   * to a conversion.
   * \param PreprocessParams a polymorphic pointer at the
   * object containing hyperparameters needed for preprocessing.
   * \param
   * converter Converter object to be used for determining available Format
   * conversions.
   * \param contexts Contexts available for execution of the
   * preprocessing.
   * \param convert_input whether or not to convert the input format if that is
   * needed
   * \param sf a single input Format* (this is templated to allow
   * variadic definition).
   * \param sfs a variadic Format* (this is templated to
   * allow variadic definition).
   * \return the output of the preprocessing (of
   * type ReturnType).
   */
  template <typename F, typename... SF>
  ReturnType Execute(PreprocessParams *params,
                     utils::converter::Converter *converter,
                     std::vector<context::Context *> contexts, bool convert_input, F sf, SF... sfs);
  //! Executes preprocessing on input formats (given variadically)
  /*!
   * Determines the function needed to carry out preprocessing on input Format*
   * objects (given variadically), as well as the Format conversions needed on
   * the inputs, executes the preprocessing, and returns:
   * - the preprocessing result.
   * - pointers at any Format objects that were created due to a conversion.
   * Note: this function will delete any intermediery Format objects that were
   * created due to a conversion.
   * \param PreprocessParams a polymorphic pointer
   * at the object containing hyperparameters needed for preprocessing.
   * \param
   * converter Converter object to be used for determining available Format
   * conversions.
   * \param contexts Contexts available for execution of the
   * preprocessing.
   * \param convert_input whether or not to convert the input format if that is
   * needed
   * \param sf a single input Format* (this is templated to allow
   * variadic definition).
   * \param sfs a variadic Format* (this is templated to
   * allow variadic definition).
   * \return a tuple containing a) the output of the
   * preprocessing (of type ReturnType), and b) a vector of Format*, where each
   * pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr.
   */
  template <typename F, typename... SF>
  std::tuple<std::vector<format::Format *>, ReturnType>
  CachedExecute(PreprocessParams *params, utils::converter::Converter *sc,
                std::vector<context::Context *> contexts, bool convert_input, F sf, SF... sfs);
};

template <typename ReturnType>
class GenericPreprocessType : public FunctionMatcherMixin<ReturnType> {
protected:
public:
  int GetOutput(format::Format *csr, PreprocessParams *params,
                std::vector<context::Context *>, bool convert_input);
  std::tuple<std::vector<format::Format *>, int>
  GetOutputCached(format::Format *csr, PreprocessParams *params,
                  std::vector<context::Context *>, bool convert_input);
  virtual ~GenericPreprocessType();
};

//! An abstract class representing reordering algorithms.
/*!
 * Class that generalizes reordering algorithms. It defines the API used for
 * reordering as well as the return type of reordering (IDType*).
 * @tparam IDType  the data type of row and column numbers (vertex IDs in the
 * case of graphs)
 */
template <typename IDType>
class ReorderPreprocessType : public FunctionMatcherMixin<IDType *> {
protected:
public:
  //! Generates a reordering inverse permutation of `format` using one of the
  //! contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @return an invere permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  IDType *GetReorder(format::Format *format,
                     std::vector<context::Context *> contexts, bool convert_input);
  //! Generates a reordering inverse permutation of `format` with the given
  //! PreprocessParams object and using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param params a polymorphic pointer at a `PreprocessParams` object that
   * will contain hyperparameters used for reordering.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @return an inverse permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  IDType *GetReorder(format::Format *format, PreprocessParams *params,
                     std::vector<context::Context *> contexts, bool convert_input);
  //! Generates a reordering using one of the contexts in `contexts`, and caches
  //! intermediate `Format` objects.
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * an inverse permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  std::tuple<std::vector<format::Format *>, IDType *>
  GetReorderCached(format::Format *csr, std::vector<context::Context *>, bool convert_input);
  //! Generates a reordering inverse permutation of `format` with the given
  //! PreprocessParams object and using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param params a polymorphic pointer at a `PreprocessParams` object that
   * will contain hyperparameters used for reordering.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * an inverse permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  std::tuple<std::vector<format::Format *>, IDType *>
  GetReorderCached(format::Format *csr, PreprocessParams *params,
                   std::vector<context::Context *>, bool convert_input);
  virtual ~ReorderPreprocessType();
};

//! Reordering preprocessing algorithm that reorders a format by representing it
//! as an adjacency matrix of a graph and ordering its vertices by degree
template <typename IDType, typename NNZType, typename ValueType>
class DegreeReorder : public ReorderPreprocessType<IDType> {
public:
  DegreeReorder(bool ascending);
  //! The hyperparameters used by the implementation functions
  struct DegreeReorderParams : PreprocessParams {
    bool ascending;
    DegreeReorderParams(bool ascending) : ascending(ascending) {}
  };
  typedef DegreeReorderParams ParamsType;
  DegreeReorder(DegreeReorderParams);

protected:
  //! An implementation function that will reorder a CSR format
  /*!
   *
   * @param formats a vector containing a single Format object of type CSR
   * @param params a polymorphic pointer at a `DegreeReorderParams` object
   * @return an inverse permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  static IDType *CalculateReorderCSR(std::vector<format::Format *> formats,
                                     PreprocessParams *params);
};

//! A generic reordering class that the user instantiate and then register their
//! own functions to.
template <typename IDType, typename NNZType, typename ValueType>
class GenericReorder : public ReorderPreprocessType<IDType> {
public:
  typedef PreprocessType ParamsType;
  GenericReorder();
};

//! Reordering using the Reverse Cuthill-McKee algorithm:
//! https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
template <typename IDType, typename NNZType, typename ValueType>
class RCMReorder : public ReorderPreprocessType<IDType> {
  typedef typename std::make_signed<IDType>::type SignedID;

public:
  struct RCMReorderParams : PreprocessParams {};
  typedef RCMReorderParams ParamsType;
  RCMReorder();
  RCMReorder(ParamsType p);

protected:
  static IDType peripheral(NNZType *xadj, IDType *adj, IDType n, IDType start,
                           SignedID *distance, IDType *Q);
  //! An implementation function that will reorder a CSR format
  /*!
   *
   * @param formats a vector containing a single Format object of type CSR
   * @param params a polymorphic pointer at a `RCMReorderParams` object
   * @return an inverse permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  static IDType *GetReorderCSR(std::vector<format::Format *> formats,
                               PreprocessParams *);
};

//! Transforms a format according to an inverse permutation of its rows/columns
template <typename InputFormatType, typename ReturnFormatType>
class TransformPreprocessType : public FunctionMatcherMixin<ReturnFormatType *> {
public:
  TransformPreprocessType(){
    static_assert(
        std::is_base_of<format::Format, InputFormatType>::value,
        "TransformationPreprocessType must take as input a Format object");
    static_assert(std::is_base_of<format::Format, ReturnFormatType>::value,
                  "TransformationPreprocessType must return a Format object");
  }
  //! Transforms `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @return a transformed Format object
   */
  ReturnFormatType *GetTransformation(format::Format *csr,
                                    std::vector<context::Context *>, bool convert_input);
  //! Transforms `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param params a polymorphic pointer at a params object
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @return a transformed Format object
   */
  ReturnFormatType *GetTransformation(format::Format *csr,
                                    PreprocessParams *params,
                                    std::vector<context::Context *>, bool convert_input);
  //! Transforms `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * a transformed Format object.
   */
  std::tuple<std::vector<format::Format *>, ReturnFormatType *>
  GetTransformationCached(format::Format *csr, std::vector<context::Context *>, bool convert_input);
  //! Transforms `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param params a polymorphic pointer at a params object
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * a transformed Format object.
   */
  std::tuple<std::vector<format::Format *>, ReturnFormatType *>
  GetTransformationCached(format::Format *csr, PreprocessParams *params,
                          std::vector<context::Context *>, bool convert_input);
  virtual ~TransformPreprocessType();
};

template <typename IDType, typename NNZType, typename ValueType>
class PermuteOrderTwo : public TransformPreprocessType<
                    format::FormatOrderTwo<IDType, NNZType, ValueType>,
                    format::FormatOrderTwo<IDType, NNZType, ValueType>> {
public:
  PermuteOrderTwo(IDType *, IDType *);
  struct PermuteOrderTwoParams : PreprocessParams {
    IDType *row_order;
    IDType *col_order;
    explicit PermuteOrderTwoParams(IDType *r_order, IDType* c_order) : row_order(r_order), col_order(c_order){};
  };
  explicit PermuteOrderTwo(PermuteOrderTwoParams);
  typedef PermuteOrderTwoParams ParamsType;

protected:
  //! An implementation function that will transform a CSR format into another
  //! CSR
  /*!
   *
   * @param formats a vector containing a single Format object of type CSR
   * @param params a polymorphic pointer at a `TransformParams` object
   * @return a transformed Format object of type CSR
   */
  static format::FormatOrderTwo<IDType, NNZType, ValueType> *PermuteOrderTwoCSR(std::vector<format::Format *> formats,
                                      PreprocessParams *);
};

template <typename IDType, typename ValueType>
class PermuteOrderOne
    : public TransformPreprocessType<format::FormatOrderOne<ValueType>,
                                     format::FormatOrderOne<ValueType>> {
public:
  PermuteOrderOne(IDType *);
  struct PermuteOrderOneParams : PreprocessParams {
    IDType *order;
    explicit PermuteOrderOneParams(IDType *order) : order(order){};
  };
  typedef PermuteOrderOneParams ParamsType;
  explicit PermuteOrderOne(ParamsType);

protected:
  //! An implementation function that will transform a CSR format into another
  //! CSR
  /*!
   *
   * @param formats a vector containing a single Format object of type Array
   * @param params a polymorphic pointer at a `TransformParams` object
   * @return a transformed Format object of type CSR
   */
  static format::FormatOrderOne<ValueType> *PermuteArray(std::vector<format::Format *> formats,
                                      PreprocessParams *);
};

//! A class that does feature extraction.
/*!
 * An ExtractableType class that has a Converter and the function matching
 * capability. In other words, an Extractable to which implementation functions
 * can be added and used. \tparam FeatureType the return type of feature
 * extraction
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
};

//! Calculate the Jaccard Weights of the edges in a graph representation of a
//! format object
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class JaccardWeights : public FunctionMatcherMixin<format::Format *> {
public:
  struct JaccardWeightsParams : PreprocessParams{};
  typedef JaccardWeightsParams ParamsType;
  JaccardWeights();
  JaccardWeights(ParamsType);
  //! Take a single Format object representating a graph and get the Jaccard
  //! Weights as a 1D format object
  /*!
   *
   * @param format input format object representing a graph
   * @return a 1D format (array) where element i in the array is the Jaccard
   * Weight of edge i in the graph (ith non-zero)
   */
  format::Format *GetJaccardWeights(format::Format *format,
                                    std::vector<context::Context *>, bool convert_input);
#ifdef USE_CUDA
  //! Take a CUDACSR representating a graph and get the Jaccard Weights as a
  //! CUDAArray
  /*!
   *
   * @param formats a vector of size 1 with formats[0] being CUDACSR
   * representing a graph
   * @param params a polymorphic pointer at a PreprocessParams (not used)
   * @return a 1D array (CUDAArray) where element i in the array is the Jaccard
   * Weight of edge i in the graph (ith non-zero)
   */
  static format::Format *
  GetJaccardWeightCUDACSR(std::vector<format::Format *> formats,
                          PreprocessParams *params);
#endif
  ~JaccardWeights();
};

//! Find the degree distribution of the graph representation of a format object
/*!
 *
 * @tparam FeatureType the type in which the distribution value are returned --
 * should be a floating type
 */
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class DegreeDistribution : public FeaturePreprocessType<FeatureType *> {

public:
  struct DegreeDistributionParams : PreprocessParams {};
  typedef DegreeDistributionParams ParamsType;
  DegreeDistribution();
  DegreeDistribution(DegreeDistributionParams);
  DegreeDistribution(const DegreeDistribution &);
  DegreeDistribution(std::shared_ptr<DegreeDistributionParams>);
  virtual std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>, bool convert_input);
  virtual std::vector<std::type_index> get_sub_ids();
  virtual std::vector<ExtractableType *> get_subs();
  static std::type_index get_feature_id_static();

  //! Degree distribution generation executor function that carries out function
  //! matching
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting
   * features. \return an array of size format.get_dimensions()[0] where element
   * i is the degree distribution of the ith vertex in `formats`
   */
  FeatureType *GetDistribution(format::Format *format,
                               std::vector<context::Context *> contexts, bool convert_input);
  //! Degree distribution generation executor function that carries out function
  //! matching on a Graph
  /*!
   *
   * \param object a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting
   * features. \return an array of size format.get_dimensions()[0] where element
   * i is the degree distribution of the ith vertex in `formats`
   */
  FeatureType *
  GetDistribution(object::Graph<IDType, NNZType, ValueType> *object,
                  std::vector<context::Context *> contexts, bool convert_input);
  //! Degree distribution generation executer function that carries out function
  //! matching with cached outputs
  /*!
   * Generates the degree distribution of the passed format. If the input format
   * was converted to other format types, the converting results are also
   * returned with the output \param format a single format pointer to any
   * format \param contexts vector of contexts that can be used for extracting
   * features. \return A tuple with the first element being a vector of Format*,
   * where each pointer in the output points at the format that the corresponds
   * Format object from the the input was converted to. If an input Format
   * wasn't converted, the output pointer will point at nullptr. The second
   * element is an array of size format.get_dimensions()[0] where element i is
   * the degree distribution of the ith vertex in `formats`
   */
  std::tuple<std::vector<format::Format *>, FeatureType *>
  GetDistributionCached(format::Format *format,
                        std::vector<context::Context *> contexts, bool convert_input);

  static FeatureType
      *
      //! Degree distribution generation implementation function for CSRs
      /*!
       *
       * \param format a single format pointer to any format
       * \return an array of size formats[0].get_dimensions()[0] where element i
       * is the degree distribution of the ith vertex in `formats[0]`
       */
      GetDegreeDistributionCSR(std::vector<format::Format *> formats,
                               PreprocessParams *params);
  ~DegreeDistribution();

protected:
  void Register();
};

//! Count the degrees of every vertex in the graph representation of a format
//! object
template <typename IDType, typename NNZType, typename ValueType>
class Degrees : public FeaturePreprocessType<IDType *> {

public:
  struct DegreesParams : PreprocessParams {};
  typedef DegreesParams ParamsType;
  Degrees();
  Degrees(DegreesParams);
  Degrees(const Degrees<IDType, NNZType, ValueType> &d);
  Degrees(std::shared_ptr<DegreesParams>);
  std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>, bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<ExtractableType *> get_subs() override;
  static std::type_index get_feature_id_static();

  //! Degree generation executer function that carries out function matching
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting
   * features. \return an array of size format.get_dimensions()[0] where element
   * i is the degree of the ith vertex in `format`
   */
  IDType *GetDegrees(format::Format *format,
                     std::vector<context::Context *> contexts, bool convert_input);
  //! Degree generation implementation function for CSRs
  /*!
   *
   * \param formats A vector containing a single format pointer that should
   * point at a CSR object \param params a PreprocessParams pointer, though it
   * is not used in the function \return an array of size
   * formats[0].get_dimensions()[0] where element i is the degree of the ith
   * vertex in `formats[0]`
   */
  static IDType *GetDegreesCSR(std::vector<format::Format *> formats,
                               PreprocessParams *params);
  ~Degrees();

protected:
  void Register();
};

//! Find the degree and degree distribution of each vertex in the graph
//! representation of a format object
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class Degrees_DegreeDistribution
    : public FeaturePreprocessType<
          std::unordered_map<std::type_index, std::any>> {
  struct Params : PreprocessParams {};
  typedef Params ParamsType;

public:
  Degrees_DegreeDistribution();
  Degrees_DegreeDistribution(Params);
  Degrees_DegreeDistribution(
      const Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>
          &d);
  Degrees_DegreeDistribution(std::shared_ptr<Params>);
  std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>, bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<ExtractableType *> get_subs() override;
  static std::type_index get_feature_id_static();

  //! Degree and degree distribution generation executor function that carries
  //! out function matching
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting
   * features. \return a map with two (type_index, any) pairs. One is a degrees
   * array of type IDType*, and one is a degree distribution array of type
   * FeatureType*. Both arrays have the respective metric of the ith vertex in
   * the ith array element.
   */
  std::unordered_map<std::type_index, std::any>
  Get(format::Format *format, std::vector<context::Context *> contexts, bool convert_input);

  //! Degree and degree distribution implementation function for CSRs
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting
   * features. \return a map with two (type_index, any) pairs. One is a degrees
   * array of type IDType*, and one is a degree distribution array of type
   * FeatureType*. Both arrays have the respective metric of the ith vertex in
   * the ith array element.
   */
  static std::unordered_map<std::type_index, std::any>
  GetCSR(std::vector<format::Format *> formats, PreprocessParams *params);
  ~Degrees_DegreeDistribution();

protected:
  void Register();
};

//! An abstract class representing partitioning algorithms.
/*!
 * Class that generalizes partitioning algorithms. It defines the API used for
 * partitioning as well as the return type of partitioning (IDType*).
 * @tparam IDType  the data type of row and column numbers (vertex IDs in the
 * case of graphs)
 */
template <typename IDType>
class PartitionPreprocessType : public FunctionMatcherMixin<IDType *> {
public:
  PartitionPreprocessType();

  //! Performs a partition operation using the default parameters
  /*!
   * @returns An IDType array where the i-th index contains the ID for the partitioning i belongs to
   */
  IDType* Partition(format::Format * format, std::vector<context::Context*> contexts, bool convert_input);

  //! Performs a partition operation using the parameters supplied by the user
  /*!
   * @returns An IDType array where the i-th index contains the ID for the partitioning i belongs to
   */
  IDType* Partition(format::Format *format, PreprocessParams *params,
                     std::vector<context::Context *> contexts, bool convert_input);
  virtual ~PartitionPreprocessType();
};


#ifdef USE_METIS


//! A wrapper for the METIS partitioner
/* !
 * Wraps the METIS partitioner available here: https://github.com/KarypisLab/METIS
 * The library must be compiled with the USE_METIS option turned on
 * and the pre-built METIS library should be available.
 * See the Optional Dependencies page (under Getting Started) in our documentation for more info.
 * Detailed explanations of the options can be found here: http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
 */
template <typename IDType, typename NNZType, typename ValueType>
class MetisPartition : public PartitionPreprocessType<IDType> {
private:
  static IDType* PartitionCSR(std::vector<format::Format*> formats, PreprocessParams* params);

public:
  MetisPartition();

  //! Objectives to be optimized by METIS
  typedef enum {
    METIS_OBJTYPE_CUT,
    METIS_OBJTYPE_VOL,
    METIS_OBJTYPE_NODE
  } mobjtype_et;

  //! Partitiong Methods
  typedef enum {
    METIS_PTYPE_RB,
    METIS_PTYPE_KWAY
  } mptype_et;

  //! Coarsening Schemes
  typedef enum {
    METIS_CTYPE_RM,
    METIS_CTYPE_SHEM
  } mctype_et;

  //! Determines the algorithm used for initial partitioning
  typedef enum {
    METIS_IPTYPE_GROW,
    METIS_IPTYPE_RANDOM,
    METIS_IPTYPE_EDGE,
    METIS_IPTYPE_NODE,
    METIS_IPTYPE_METISRB
  } miptype_et;


  //! Determines the algorithm used for refinement
  typedef enum {
    METIS_RTYPE_FM,
    METIS_RTYPE_GREEDY,
    METIS_RTYPE_SEP2SIDED,
    METIS_RTYPE_SEP1SIDED
  } mrtype_et;

  //! Parameters for metis partitioning
  /*!
   * This struct replaces the options array of METIS
   * The names of the options are identical to the array
   * and can be found here: http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
   */
  struct MetisParams : PreprocessParams{
    int64_t num_partitions = 2;
    int64_t ptype = METIS_PTYPE_KWAY;
    int64_t objtype = METIS_OBJTYPE_CUT;
    int64_t ctype = METIS_CTYPE_RM;
    int64_t iptype = METIS_IPTYPE_GROW;
    int64_t rtype = METIS_RTYPE_FM;
    int64_t ncuts = 1;
    int64_t nseps = 1;
    int64_t numbering = 0;
    int64_t niter = 10;
    int64_t seed = 42;
    int64_t minconn = 0;
    int64_t no2hop = 0;
    int64_t contig = 0;
    int64_t compress = 0;
    int64_t ccorder = 0;
    int64_t pfactor = 0;
    int64_t ufactor = 30;
  };
};
#endif


class GraphFeatureBase {
public:
  template <typename FeatureType, typename IDType, typename NNZType, typename ValueType>
  static FeatureType* GetDegreeDistribution(typename DegreeDistribution<IDType, NNZType, ValueType, FeatureType>::ParamsType params, format::FormatOrderTwo<IDType, NNZType, ValueType> *format, std::vector<context::Context*>contexts, bool convert_input){
    DegreeDistribution<IDType, NNZType, ValueType, FeatureType> deg_dist;
    return deg_dist.GetDistribution(format, contexts, convert_input);
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static NNZType* GetDegrees(typename Degrees<IDType, NNZType, ValueType>::ParamsType params, format::FormatOrderTwo<IDType, NNZType, ValueType> *format, std::vector<context::Context*>contexts, bool convert_input){
    Degrees<IDType, NNZType, ValueType> deg_dist;
    return deg_dist.GetDegrees(format, contexts, convert_input);
  }
};
//template <typename IDType, typename NNZType, typename ValueType>
class ReorderBase {
public:
  template <template <typename, typename, typename> typename Reordering, typename IDType, typename NNZType, typename ValueType>
  static IDType* Reorder(typename Reordering<IDType, NNZType, ValueType>::ParamsType params, format::FormatOrderTwo<IDType, NNZType, ValueType>* format, std::vector<context::Context*> contexts, bool convert_input){
    static_assert(std::is_base_of_v<ReorderPreprocessType<IDType>, Reordering<IDType, NNZType, ValueType>>, "You must pass a reordering function (with base ReorderPreprocessType) to ReorderBase::Reorder");
    static_assert(!std::is_same_v<GenericReorder<IDType, NNZType, ValueType>, Reordering<IDType, NNZType, ValueType>>, "You must pass a reordering function (with base ReorderPreprocessType) to ReorderBase::Reorder");
    Reordering<IDType, NNZType, ValueType> reordering(params);
    return reordering.GetReorder(format, contexts, convert_input);
  }

  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename IDType, typename NNZType, typename ValueType>
  static ReturnFormatType<IDType, NNZType, ValueType>* Permute2D(IDType* ordering, format::FormatOrderTwo<IDType, NNZType, ValueType>* format, std::vector<context::Context*> contexts, bool convert_input){
    PermuteOrderTwo<IDType, NNZType, ValueType> perm(ordering, ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<IDType, NNZType, ValueType>, format::FormatOrderTwo<IDType, NNZType, ValueType>>)
      return out_format;
    else
      return out_format->template Convert<ReturnFormatType>();
  }

  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename IDType, typename NNZType, typename ValueType>
  static ReturnFormatType<IDType, NNZType, ValueType>* Permute2DRowColumnWise(IDType* row_ordering, IDType* col_ordering, format::FormatOrderTwo<IDType, NNZType, ValueType>* format, std::vector<context::Context*> contexts, bool convert_input){
    PermuteOrderTwo<IDType, NNZType, ValueType> perm(row_ordering, col_ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<IDType, NNZType, ValueType>, format::FormatOrderTwo<IDType, NNZType, ValueType>>)
      return out_format;
    else
      return out_format->template Convert<ReturnFormatType>();
  }

  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename IDType, typename NNZType, typename ValueType>
  static ReturnFormatType<IDType, NNZType, ValueType>* Permute2DRowWise(IDType* ordering, format::FormatOrderTwo<IDType, NNZType, ValueType>* format, std::vector<context::Context*> contexts, bool convert_input){
    PermuteOrderTwo<IDType, NNZType, ValueType> perm(ordering, nullptr);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<IDType, NNZType, ValueType>, format::FormatOrderTwo<IDType, NNZType, ValueType>>)
      return out_format;
    else
      return out_format->template Convert<ReturnFormatType>();
  }

  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename IDType, typename NNZType, typename ValueType>
  static ReturnFormatType<IDType, NNZType, ValueType>* Permute2DColWise(IDType* ordering, format::FormatOrderTwo<IDType, NNZType, ValueType>* format, std::vector<context::Context*> contexts, bool convert_input){
    PermuteOrderTwo<IDType, NNZType, ValueType> perm(nullptr, ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<IDType, NNZType, ValueType>, format::FormatOrderTwo<IDType, NNZType, ValueType>>)
      return out_format;
    else
      return out_format->template Convert<ReturnFormatType>();
  }


  //! Permute a one-dimensional format using a permutation array.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param order Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \return The permuted format. Function returns a pointer at a generic FormatOrderOne object.
   * However, if the user passes a FormatOrderOne class as the templated parameter `ReturnFormatType`, e.g.
   * format::Array, then the returned format will be converted to that type.
   */
  template <template <typename> typename ReturnFormatType = format::FormatOrderOne, typename AutoIDType, typename AutoValueType>
  static ReturnFormatType<AutoValueType>* Permute1D(AutoIDType* ordering, format::FormatOrderOne<AutoValueType>* format, std::vector<context::Context*> context, bool convert_inputs){
    PermuteOrderOne<AutoIDType, AutoValueType> perm(ordering);
    auto out_format = perm.GetTransformation(format, context, convert_inputs);
    if constexpr (std::is_same_v<ReturnFormatType<AutoValueType>, format::FormatOrderOne<AutoValueType>>)
      return out_format;
    else
      return out_format->template Convert<ReturnFormatType>();
  }

  template <typename IDType, typename NumType>
  static IDType * InversePermutation(IDType*perm, NumType length){
    static_assert(std::is_integral_v<NumType>, "Length of the permutation array must be an integer");
    auto inv_perm = new IDType[length];
    for (IDType i = 0; i < length; i++){
      inv_perm[perm[i]] = i;
    }
    return inv_perm;
  }

};

template <template <typename, typename, typename> typename Reordering, typename IDType, typename NNZType, typename ValueType>
int tester(typename Reordering<IDType, NNZType, ValueType>::ParamsType params){
  std::cout << "It works!\n";
  Reordering<IDType, NNZType, ValueType> r;
  return 1;
}


} // namespace sparsebase::preprocess
#ifdef _HEADER_ONLY
#include "sparsebase/preprocess/preprocess.cc"
#ifdef USE_CUDA
#include "cuda/preprocess.cu"
#endif
#endif

#endif // SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
