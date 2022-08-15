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
  Extract(format::Format *format, std::vector<context::Context *> contexts) = 0;
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
   * to a conversion. \param PreprocessParams a polymorphic pointer at the
   * object containing hyperparameters needed for preprocessing. \param
   * converter Converter object to be used for determining available Format
   * conversions. \param contexts Contexts available for execution of the
   * preprocessing. \param sf a single input Format* (this is templated to allow
   * variadic definition). \param sfs a variadic Format* (this is templated to
   * allow variadic definition). \return the output of the preprocessing (of
   * type ReturnType).
   */
  template <typename F, typename... SF>
  ReturnType Execute(PreprocessParams *params,
                     utils::converter::Converter *converter,
                     std::vector<context::Context *> contexts, F sf, SF... sfs);
  //! Executes preprocessing on input formats (given variadically)
  /*!
   * Determines the function needed to carry out preprocessing on input Format*
   * objects (given variadically), as well as the Format conversions needed on
   * the inputs, executes the preprocessing, and returns:
   * - the preprocessing result.
   * - pointers at any Format objects that were created due to a conversion.
   * Note: this function will delete any intermediery Format objects that were
   * created due to a conversion. \param PreprocessParams a polymorphic pointer
   * at the object containing hyperparameters needed for preprocessing. \param
   * converter Converter object to be used for determining available Format
   * conversions. \param contexts Contexts available for execution of the
   * preprocessing. \param sf a single input Format* (this is templated to allow
   * variadic definition). \param sfs a variadic Format* (this is templated to
   * allow variadic definition). \return a tuple containing a) the output of the
   * preprocessing (of type ReturnType), and b) a vector of Format*, where each
   * pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr.
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
                     std::vector<context::Context *> contexts);
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
                     std::vector<context::Context *> contexts);
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
  GetReorderCached(format::Format *csr, std::vector<context::Context *>);
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
                   std::vector<context::Context *>);
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
  GenericReorder();
};

//! Reordering using the Reverse Cuthill-McKee algorithm:
//! https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
template <typename IDType, typename NNZType, typename ValueType>
class RCMReorder : public ReorderPreprocessType<IDType> {
  typedef typename std::make_signed<IDType>::type SignedID;

public:
  RCMReorder();

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
template <typename IDType, typename NNZType, typename ValueType>
class TransformPreprocessType : public FunctionMatcherMixin<format::Format *> {
public:
  //! Transforms `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @return a transformed Format object
   */
  format::Format *GetTransformation(format::Format *csr,
                                    std::vector<context::Context *>);
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
  format::Format *GetTransformation(format::Format *csr,
                                    PreprocessParams *params,
                                    std::vector<context::Context *>);
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
  std::tuple<std::vector<format::Format *>, format::Format *>
  GetTransformationCached(format::Format *csr, std::vector<context::Context *>);
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
  std::tuple<std::vector<format::Format *>, format::Format *>
  GetTransformationCached(format::Format *csr, PreprocessParams *params,
                          std::vector<context::Context *>);
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
  //! An implementation function that will transform a CSR format into another
  //! CSR
  /*!
   *
   * @param formats a vector containing a single Format object of type CSR
   * @param params a polymorphic pointer at a `TransformParams` object
   * @return a transformed Format object of type CSR
   */
  static format::Format *TransformCSR(std::vector<format::Format *> formats,
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
  JaccardWeights();
  //! Take a single Format object representating a graph and get the Jaccard
  //! Weights as a 1D format object
  /*!
   *
   * @param format input format object representing a graph
   * @return a 1D format (array) where element i in the array is the Jaccard
   * Weight of edge i in the graph (ith non-zero)
   */
  format::Format *GetJaccardWeights(format::Format *format,
                                    std::vector<context::Context *>);
#ifdef CUDA
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
  DegreeDistribution();
  DegreeDistribution(const DegreeDistribution &);
  DegreeDistribution(std::shared_ptr<DegreeDistributionParams>);
  virtual std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>);
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
                               std::vector<context::Context *> contexts);
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
                  std::vector<context::Context *> contexts);
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
                        std::vector<context::Context *> contexts);

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
   * \param contexts vector of contexts that can be used for extracting
   * features. \return an array of size format.get_dimensions()[0] where element
   * i is the degree of the ith vertex in `format`
   */
  IDType *GetDegrees(format::Format *format,
                     std::vector<context::Context *> contexts);
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

public:
  Degrees_DegreeDistribution();
  Degrees_DegreeDistribution(
      const Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>
          &d);
  Degrees_DegreeDistribution(std::shared_ptr<Params>);
  std::unordered_map<std::type_index, std::any>
  Extract(format::Format *format, std::vector<context::Context *>) override;
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
  Get(format::Format *format, std::vector<context::Context *> contexts);

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

enum BitMapSize{
  BitSize16 = 16,
  BitSize32 = 32/*,
  BitSize64 = 64*/ //at the moment, using 64 bits is not working as intended
};

template <typename IDType, typename NNZType, typename ValueType>
class GrayReorder : public ReorderPreprocessType<IDType> {
    struct GrayReorderParams : PreprocessParams {
      int resolution;
      int nnz_threshold;
      int sparse_density_group_size;
    };

    typedef std::pair<IDType,unsigned long> row_grey_pair;

    public:
      GrayReorder(BitMapSize resolution, int nnz_threshold, int sparse_density_group_size) {
        auto params_struct = new GrayReorderParams;
        params_struct->resolution = resolution;
        params_struct->nnz_threshold = nnz_threshold;
        params_struct->sparse_density_group_size = sparse_density_group_size;
        // this->params_ = std::unique_ptr<GrayReorderParams>(new GrayReorderParams{resolution, nnz_threshold, sparse_density_group_size});
        this->params_ = std::unique_ptr<GrayReorderParams>(params_struct);

        this->SetConverter(
        utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});

        this->RegisterFunction(
            {format::CSR<IDType, NNZType, ValueType>::get_format_id_static()}, GrayReorderingCSR);
      }

  protected:

	static bool desc_comparator ( const row_grey_pair& l, const row_grey_pair& r)
	{ return l.second > r.second; }

	static bool asc_comparator ( const row_grey_pair& l, const row_grey_pair& r)
	{ return l.second < r.second; }

  //not sure if all IDTypes work for this
	static unsigned long grey_bin_to_dec(unsigned long n){
		unsigned long inv = 0;

		for(;n;n=n>>1)
			inv ^= n;

		return inv;
	}

  static void print_dec_in_bin(unsigned long  n, int size){
    // array to store binary number
    int binaryNum[size];

    // counter for binary array
    int i = 0;
    while (n > 0) {

      // storing remainder in binary array
      binaryNum[i] = n % 2;
      n = n / 2;
      i++;
    }

    // printing binary array in reverse order
    for (int j = i - 1; j >= 0; j--)
      std::cout << binaryNum[j];

    std::cout << "\n";  
  }


	//not sure if all IDTypes work for this
	static unsigned long bin_to_grey(unsigned long n){
		/* Right Shift the number by 1
		taking xor with original number */
		return n ^ (n >> 1);
	}

	// bool is_banded(std::vector<format::Format *> input_sf, int band_size = -1, std::vector<IDType> order) {
	static bool is_banded(int nnz, int n_cols, IDType *row_ptr, IDType *cols, std::vector<IDType> order, int band_size = -1) {

		if (band_size == -1)
		band_size = n_cols / 64;
		int band_count = 0;
		bool banded = false;

		for (int r = 0; r < order.size(); r++) {
		for (int i = row_ptr[order[r]]; i < row_ptr[order[r]+1]; i++) {
			int col = cols[i];
			if (abs(col - r) <= band_size)
			band_count++;
		}
		}

		if (double(band_count) / nnz >= 0.3) {
		banded = true;
		}
		std::cout << "NNZ % in band: " << double(band_count) / nnz << std::endl;
		return banded;
	}

	static IDType* GrayReorderingCSR(std::vector<format::Format *> input_sf, PreprocessParams *poly_params) {
    auto csr = input_sf[0]->As<format::CSR<IDType, NNZType, ValueType>>();
		context::CPUContext *cpu_context = static_cast<context::CPUContext *>(csr->get_context());

		IDType n_rows = csr->get_dimensions()[0];
		/*This array stores the permutation vector such as order[0] = 243 means that row 243 is the first row of the reordered matrix*/
		IDType * order = new IDType[n_rows](); 

		GrayReorderParams *params = static_cast<GrayReorderParams *>(poly_params);
		int group_size = params->sparse_density_group_size;
		int bit_resolution = params->resolution;

		int raise_to = 0;
		int adder = 0;
		int start_split_reorder,end_split_reorder;

		int last_row_nnz_count = 0;
		int threshold = 0; //threshold used to set bit in bitmap to 1
		bool decresc_grey_order = false;

		int group_count = 0;

		//Initializing row order
		std::vector<IDType> v_order;
		std::vector<IDType> sparse_v_order;
		std::vector<IDType> dense_v_order;

		//Splitting original matrix's rows in two submatrices
		IDType sparse_dense_split=0;
		for(IDType i = 0; i < n_rows; i++){
			if((csr->get_row_ptr()[i+1]-csr->get_row_ptr()[i]) <= params->nnz_threshold){
				sparse_v_order.push_back(i);
				sparse_dense_split++;
			}
			else{
				dense_v_order.push_back(i);
			}
		}


		v_order.reserve( sparse_v_order.size() + dense_v_order.size() ); // preallocate memory

		bool is_sparse_banded = is_banded(csr->get_num_nnz(), csr->get_dimensions()[1], csr->get_row_ptr(), csr->get_col(), sparse_v_order);
		if(is_sparse_banded) std::cout << "Sparse Sub-Matrix highly banded - Performing just density reordering" << std::endl;

		bool is_dense_banded = is_banded(csr->get_num_nnz(), csr->get_dimensions()[1], csr->get_row_ptr(), csr->get_col(), dense_v_order);
		if(is_dense_banded) std::cout << "Dense Sub-Matrix highly banded - Maintaining structure" << std::endl;

		std::sort(sparse_v_order.begin(),sparse_v_order.end(), [&](int i,int j) -> bool {return (csr->get_row_ptr()[i+1]-csr->get_row_ptr()[i])<(csr->get_row_ptr()[j+1]-csr->get_row_ptr()[j]);} ); //reorder sparse matrix into nnz amount

		//bit resolution determines the width of the bitmap of each row
		if(n_rows < bit_resolution){
			bit_resolution = n_rows;
		}

		int row_split = n_rows/bit_resolution;

		auto nnz_per_row_split = new IDType[bit_resolution];
		auto nnz_per_row_split_bin = new IDType[bit_resolution];

		unsigned long  decimal_bit_map = 0;
		unsigned long  dec_begin = 0;
		int dec_begin_ind = 0;

		std::vector<row_grey_pair> reorder_section; //vector that contains a section to be reordered

		if(!is_sparse_banded){ //if banded just row ordering by nnz count is enough, else do bitmap reordering in groups

		for(int i = 0; i < sparse_v_order.size(); i++) { //sparse sub matrix if not highly banded
			if(i == 0){
				last_row_nnz_count = csr->get_row_ptr()[sparse_v_order[i]+1] - csr->get_row_ptr()[sparse_v_order[i]]; //get nnz count in first row
				start_split_reorder = 0;
			} //check if nnz amount changes from last row

			if((csr->get_row_ptr()[sparse_v_order[i]+1]-csr->get_row_ptr()[sparse_v_order[i]]) == 0){ //for cases where rows are empty
				start_split_reorder = i+1;
				last_row_nnz_count = csr->get_row_ptr()[sparse_v_order[i+1]+1] - csr->get_row_ptr()[sparse_v_order[i+1]];
				continue;
			}

			//reset bitmap for this row
			for(int i = 0; i < bit_resolution; i++) nnz_per_row_split[i] = 0; 
			for(int i = 0; i < bit_resolution; i++) nnz_per_row_split_bin[i] = 0; 
			
			//get number of nnz in each bitmap section
			for (int k = csr->get_row_ptr()[sparse_v_order[i]]; k < csr->get_row_ptr()[sparse_v_order[i]+1]; k++){
				nnz_per_row_split[csr->get_col()[k]/row_split]++;
			}
			
			//get bitmap of the row in decimal value (first rows are less significant bits)
			decimal_bit_map = 0;
			for (int j=0; j < bit_resolution; j++){
				adder=0;
				if(nnz_per_row_split[j] > threshold){
					nnz_per_row_split_bin[j] = 1;
					raise_to = j;
					adder = pow(2,raise_to);

					decimal_bit_map = decimal_bit_map + adder;
				}
			}

			//if number of nnz changed from last row, increment group count, which might trigger a reorder of the group
			if((i != 0) && (last_row_nnz_count != (csr->get_row_ptr()[sparse_v_order[i]+1] - csr->get_row_ptr()[sparse_v_order[i]]))){
				group_count = group_count+1;
				std::cout << "Rows[" << start_split_reorder << " -> " << i-1 << "] NNZ Count: " << last_row_nnz_count <<"\n";
				//update nnz count for current row
				last_row_nnz_count = csr->get_row_ptr()[sparse_v_order[i]+1] - csr->get_row_ptr()[sparse_v_order[i]];

				//if group size achieved, start reordering section until this row
				if(group_count == group_size){
					end_split_reorder = i;
					std::cout << "Reorder Group[" << start_split_reorder << " -> " << end_split_reorder-1 << "]\n";
					//start next split the split for processing
					
					//process and reorder the reordered_matrix array till this point (ascending or descending alternately)
					if(!decresc_grey_order){
						sort(reorder_section.begin(),reorder_section.end(), asc_comparator); 
						decresc_grey_order = !decresc_grey_order;
					}
					else{
						sort(reorder_section.begin(),reorder_section.end(), desc_comparator); 
						decresc_grey_order = !decresc_grey_order;
					}   
					
          dec_begin = reorder_section[0].second;
          dec_begin_ind = start_split_reorder;

					//apply reordered
					for(int a = start_split_reorder; a < end_split_reorder; a++){
            if((dec_begin != reorder_section[a-start_split_reorder].second)&&(a < 100000)){

              std::cout << "Rows[" << dec_begin_ind << " -> " << a << "] Grey Order: " << dec_begin << "// Binary: \n";
              // print_dec_in_bin(bin_to_grey(dec_begin));

              dec_begin = reorder_section[a-start_split_reorder].second;
              dec_begin_ind = a;
            }

						sparse_v_order[a] = reorder_section[a-start_split_reorder].first;
					}
					
					start_split_reorder = i;

					reorder_section.clear();
					group_count = 0;
				}
			}

// if(decimal_bit_map != 0){
//   for(int i = 0; i < bit_resolution; i++){
//     std::cout << "[" << nnz_per_row_split_bin[(bit_resolution-1)-i] << "]";
//   }
//     std::cout << "\nRow "<< i << "[" << v_order[i] << "] grey value: " << decimal_bit_map << " translates to: "<< grey_bin_to_dec(decimal_bit_map) <<"\n";
// }

//

			reorder_section.push_back(row_grey_pair(sparse_v_order[i],grey_bin_to_dec(decimal_bit_map)));

			//when reaching end of sparse submatrix, reorder section
			if(i == sparse_v_order.size()-1){
				end_split_reorder = sparse_v_order.size();
				std::cout << "Rows[" << start_split_reorder << " -> " << end_split_reorder-1 << "] NNZ Count: " << last_row_nnz_count <<"\n";
				if(!decresc_grey_order){
					sort(reorder_section.begin(),reorder_section.end(), asc_comparator); 
					decresc_grey_order = !decresc_grey_order;
				}
				else{
					sort(reorder_section.begin(),reorder_section.end(), desc_comparator); 
					decresc_grey_order = !decresc_grey_order;
				}   
				for(int a = start_split_reorder; a < end_split_reorder; a++){
					sparse_v_order[a] = reorder_section[a-start_split_reorder].first;
				}

			}
		}

		reorder_section.clear();          
		}

		if(!is_dense_banded){

			std::cout << "Rows [" << sparse_dense_split << "-" << n_rows << "] Starting Dense Sorting through NNZ and Grey code..\n";

			for(int i = 0; i < dense_v_order.size(); i++) {
					//if first row, establish the nnz amount, and starting index
				for(int i = 0; i < bit_resolution; i++) nnz_per_row_split[i] = 0;

				for (int k = csr->get_row_ptr()[dense_v_order[i]]; k < csr->get_row_ptr()[dense_v_order[i]+1]; k++){
					nnz_per_row_split[csr->get_col()[k]/row_split]++;
				}
				threshold = (csr->get_row_ptr()[dense_v_order[i]+1]-csr->get_row_ptr()[dense_v_order[i]])/bit_resolution; //floor
				decimal_bit_map = 0;
				for (int j=0; j < bit_resolution; j++){
					adder=0;
					if(nnz_per_row_split[j] > threshold){

						raise_to = j; // row 0 = lowest significant bit
						adder = pow(2,raise_to);

						decimal_bit_map = decimal_bit_map + adder;
					}
				}
				reorder_section.push_back(row_grey_pair(dense_v_order[i],grey_bin_to_dec(decimal_bit_map)));
			}
			std::cout << "Reordering Rows based on grey values...\n";
			std::sort(reorder_section.begin(),reorder_section.end(), asc_comparator); 

			for(int a = 0; a < dense_v_order.size(); a++){
				dense_v_order[a] = reorder_section[a].first;
			}

			reorder_section.clear();
		}

		v_order.insert( v_order.end(), sparse_v_order.begin(), sparse_v_order.end() );
		v_order.insert( v_order.end(), dense_v_order.begin(), dense_v_order.end() );


		/*This order array stores the inverse permutation vector such as order[0] = 243 means that row 0 is placed at the row 243 of the reordered matrix*/
		// std::vector<IDType> v_order_inv(n_rows);
		for(int i = 0; i < n_rows; i++){
		order[v_order[i]] = i;
		}
		// std::copy(v_order_inv.begin(), v_order_inv.end(), order); 

		return order;
	}

};


} // namespace sparsebase::preprocess
#ifdef _HEADER_ONLY
#include "sparsebase/preprocess/preprocess.cc"
#ifdef CUDA
#include "cuda/preprocess.cu"
#endif
#endif

#endif // SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
