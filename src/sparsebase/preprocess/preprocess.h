/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://github.com/SU-HPC/sparsebase/blob/main/LICENSE
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

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class NumSlices : public FeaturePreprocessType<FeatureType> {

public:
  struct NumSlicesParams : PreprocessParams {};

  NumSlices();

  NumSlices(const NumSlices &d);

  NumSlices(
      const std::shared_ptr<NumSlicesParams> p);

  std::unordered_map<std::type_index, std::any>
  Extract(
      format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();
  ~NumSlices() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetNumSlicesCached(format::Format *format,
                     std::vector<context::Context *> contexts);

  FeatureType
  GetNumSlices(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetNumSlices(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetNumSlicesCOO(std::vector<format::Format *> formats,
                  PreprocessParams *params);

  static FeatureType
  GetNumSlicesHigherOrderCOO(std::vector<format::Format *> formats,
                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class NumFibers : public FeaturePreprocessType<FeatureType> {

public:
  struct NumFibersParams : PreprocessParams {};

  NumFibers();

  NumFibers(const NumFibers &d);
  NumFibers(const std::shared_ptr<NumFibersParams> p);

  std::unordered_map<std::type_index, std::any>
  Extract(
      format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~NumFibers() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetNumFibersCached(format::Format *format,
                     std::vector<context::Context *> contexts);
  FeatureType
  GetNumFibers(
      format::Format *format, std::vector<context::Context *> contexts);


  FeatureType
  GetNumFibers(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetNumFibersCOO(std::vector<format::Format *> formats,
                  PreprocessParams *params);

  static FeatureType
  GetNumFibersHigherOrderCOO(std::vector<format::Format *> formats,
                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class NnzPerFiber : public FeaturePreprocessType<NNZType *> {

public:
  struct NnzPerFiberParams : PreprocessParams {};

  NnzPerFiber();

  NnzPerFiber(const NnzPerFiber &d);

  NnzPerFiber(
      const std::shared_ptr<NnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any>
  Extract(
      format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();
  ~NnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, NNZType *>
  GetNnzPerFiberCached(format::Format *format,
                       std::vector<context::Context *> contexts);

  NNZType *
  GetNnzPerFiber(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType *
  GetNnzPerFiber(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType *
  GetNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                               PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class NnzPerSlice : public FeaturePreprocessType<NNZType *> {

public:
  struct NnzPerSliceParams : PreprocessParams {};

  NnzPerSlice();

  NnzPerSlice(const NnzPerSlice &d);

  NnzPerSlice(
      const std::shared_ptr<NnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any>
  Extract(
      format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();
  ~NnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType *>
  GetNnzPerSliceCached(format::Format *format,
                       std::vector<context::Context *> contexts);

  NNZType *
  GetNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType *
  GetNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType *
  GetNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                               PreprocessParams *params);

protected:
  void Register();
};


template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class FibersPerSlice : public FeaturePreprocessType<NNZType *> {

public:
  struct FibersPerSliceParams : PreprocessParams {};

  FibersPerSlice();

  FibersPerSlice(const FibersPerSlice &d);

  FibersPerSlice(
      const std::shared_ptr<FibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any>
  Extract(
      format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();
  ~FibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType *>
  GetFibersPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);

  NNZType *
  GetFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType *
  GetFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType *
  GetFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class NumNnzFibers : public FeaturePreprocessType<FeatureType> {

public:
  struct NumNnzFibersParams : PreprocessParams {};

  NumNnzFibers();

  NumNnzFibers(const NumNnzFibers &d);

  NumNnzFibers(const std::shared_ptr<NumNnzFibersParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~NumNnzFibers() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetNumNnzFibersCached(format::Format *format,
                        std::vector<context::Context *> contexts);
  FeatureType
  GetNumNnzFibers(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetNumNnzFibers(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetNumNnzFibersCOO(std::vector<format::Format *> formats,
                     PreprocessParams *params);

  static FeatureType
  GetNumNnzFibersHigherOrderCOO(std::vector<format::Format *> formats,
                                PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class NumNnzSlices : public FeaturePreprocessType<FeatureType> {

public:
  struct NumNnzSlicesParams : PreprocessParams {};

  NumNnzSlices();

  NumNnzSlices(const NumNnzSlices &d);

  NumNnzSlices(const std::shared_ptr<NumNnzSlicesParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~NumNnzSlices() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetNumNnzSlicesCached(format::Format *format,
                        std::vector<context::Context *> contexts);
  FeatureType
  GetNumNnzSlices(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetNumNnzSlices(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetNumNnzSlicesCOO(std::vector<format::Format *> formats,
                     PreprocessParams *params);

  static FeatureType
  GetNumNnzSlicesHigherOrderCOO(std::vector<format::Format *> formats,
                                PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class MaxNnzPerSlice : public FeaturePreprocessType<NNZType> {

public:
  struct MaxNnzPerSliceParams : PreprocessParams {};

  MaxNnzPerSlice();

  MaxNnzPerSlice(const MaxNnzPerSlice &d);

  MaxNnzPerSlice(const std::shared_ptr<MaxNnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~MaxNnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetMaxNnzPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  NNZType
  GetMaxNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetMaxNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetMaxNnzPerSliceCOO(std::vector<format::Format *> formats,
                       PreprocessParams *params);

  static NNZType
  GetMaxNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);
protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class MinNnzPerSlice : public FeaturePreprocessType<NNZType> {

public:
  struct MinNnzPerSliceParams : PreprocessParams {};

  MinNnzPerSlice();

  MinNnzPerSlice(const MinNnzPerSlice &d);

  MinNnzPerSlice(const std::shared_ptr<MinNnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~MinNnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetMinNnzPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  NNZType
  GetMinNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetMinNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetMinNnzPerSliceCOO(std::vector<format::Format *> formats,
                       PreprocessParams *params);

  static NNZType
  GetMinNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class AvgNnzPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct AvgNnzPerSliceParams : PreprocessParams {};

  AvgNnzPerSlice();

  AvgNnzPerSlice(const AvgNnzPerSlice &d);

  AvgNnzPerSlice(const std::shared_ptr<AvgNnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~AvgNnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetAvgNnzPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetAvgNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetAvgNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetAvgNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class DevNnzPerSlice : public FeaturePreprocessType<NNZType> {

public:
  struct DevNnzPerSliceParams : PreprocessParams {};

  DevNnzPerSlice();

  DevNnzPerSlice(const DevNnzPerSlice &d);

  DevNnzPerSlice(const std::shared_ptr<DevNnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~DevNnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetDevNnzPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  NNZType
  GetDevNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetDevNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetDevNnzPerSliceCOO(std::vector<format::Format *> formats,
                       PreprocessParams *params);

  static NNZType
  GetDevNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};


template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class StdNnzPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct StdNnzPerSliceParams : PreprocessParams {};

  StdNnzPerSlice();

  StdNnzPerSlice(const StdNnzPerSlice &d);

  StdNnzPerSlice(const std::shared_ptr<StdNnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~StdNnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetStdNnzPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetStdNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetStdNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetStdNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class CovNnzPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct CovNnzPerSliceParams : PreprocessParams {};

  CovNnzPerSlice();

  CovNnzPerSlice(const CovNnzPerSlice &d);

  CovNnzPerSlice(const std::shared_ptr<CovNnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~CovNnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetCovNnzPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetCovNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetCovNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetCovNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};


template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class AdjNnzPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct AdjNnzPerSliceParams : PreprocessParams {};

  AdjNnzPerSlice();

  AdjNnzPerSlice(const AdjNnzPerSlice &d);

  AdjNnzPerSlice(const std::shared_ptr<AdjNnzPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~AdjNnzPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetAdjNnzPerSliceCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetAdjNnzPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetAdjNnzPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetAdjNnzPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};


template <typename IDType, typename NNZType, typename ValueType>
class MaxNnzPerFiber : public FeaturePreprocessType<NNZType> {

public:
  struct MaxNnzPerFiberParams : PreprocessParams {};

  MaxNnzPerFiber();

  MaxNnzPerFiber(const MaxNnzPerFiber &d);

  MaxNnzPerFiber(const std::shared_ptr<MaxNnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~MaxNnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetMaxNnzPerFiberCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  NNZType
  GetMaxNnzPerFiber(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetMaxNnzPerFiber(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetMaxNnzPerFiberCOO(std::vector<format::Format *> formats,
                       PreprocessParams *params);

  static NNZType
  GetMaxNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);
protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class MinNnzPerFiber : public FeaturePreprocessType<NNZType> {

public:
  struct MinNnzPerFiberParams : PreprocessParams {};

  MinNnzPerFiber();

  MinNnzPerFiber(const MinNnzPerFiber &d);

  MinNnzPerFiber(const std::shared_ptr<MinNnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~MinNnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetMinNnzPerFiberCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  NNZType
  GetMinNnzPerFiber(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetMinNnzPerFiber(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetMinNnzPerFiberCOO(std::vector<format::Format *> formats,
                       PreprocessParams *params);

  static NNZType
  GetMinNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class AvgNnzPerFiber : public FeaturePreprocessType<FeatureType> {

public:
  struct AvgNnzPerFiberParams : PreprocessParams {};

  AvgNnzPerFiber();

  AvgNnzPerFiber(const AvgNnzPerFiber &d);

  AvgNnzPerFiber(const std::shared_ptr<AvgNnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~AvgNnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetAvgNnzPerFiberCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetAvgNnzPerFiber(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetAvgNnzPerFiber(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetAvgNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class DevNnzPerFiber : public FeaturePreprocessType<NNZType> {

public:
  struct DevNnzPerFiberParams : PreprocessParams {};

  DevNnzPerFiber();

  DevNnzPerFiber(const DevNnzPerFiber &d);

  DevNnzPerFiber(const std::shared_ptr<DevNnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~DevNnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetDevNnzPerFiberCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  NNZType
  GetDevNnzPerFiber(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetDevNnzPerFiber(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetDevNnzPerFiberCOO(std::vector<format::Format *> formats,
                       PreprocessParams *params);

  static NNZType
  GetDevNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};


template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class StdNnzPerFiber : public FeaturePreprocessType<FeatureType> {

public:
  struct StdNnzPerFiberParams : PreprocessParams {};

  StdNnzPerFiber();

  StdNnzPerFiber(const StdNnzPerFiber &d);

  StdNnzPerFiber(const std::shared_ptr<StdNnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~StdNnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetStdNnzPerFiberCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetStdNnzPerFiber(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetStdNnzPerFiber(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetStdNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class CovNnzPerFiber : public FeaturePreprocessType<FeatureType> {

public:
  struct CovNnzPerFiberParams : PreprocessParams {};

  CovNnzPerFiber();

  CovNnzPerFiber(const CovNnzPerFiber &d);

  CovNnzPerFiber(const std::shared_ptr<CovNnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~CovNnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetCovNnzPerFiberCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetCovNnzPerFiber(format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetCovNnzPerFiber(object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetCovNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};


template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class AdjNnzPerFiber : public FeaturePreprocessType<FeatureType> {

public:
  struct AdjNnzPerFiberParams : PreprocessParams {};

  AdjNnzPerFiber();

  AdjNnzPerFiber(const AdjNnzPerFiber &d);

  AdjNnzPerFiber(const std::shared_ptr<AdjNnzPerFiberParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~AdjNnzPerFiber() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetAdjNnzPerFiberCached(format::Format *format,
                          std::vector<context::Context *> contexts);
  FeatureType
  GetAdjNnzPerFiber(format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetAdjNnzPerFiber(object::Graph<IDType, NNZType, ValueType> *obj,
                    std::vector<context::Context *> contexts);

  static FeatureType
  GetAdjNnzPerFiberHigherOrderCOO(std::vector<format::Format *> formats,
                                  PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class MaxFibersPerSlice : public FeaturePreprocessType<NNZType> {

public:
  struct MaxFibersPerSliceParams : PreprocessParams {};

  MaxFibersPerSlice();

  MaxFibersPerSlice(const MaxFibersPerSlice &d);

  MaxFibersPerSlice(const std::shared_ptr<MaxFibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~MaxFibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetMaxFibersPerSliceCached(format::Format *format,
                             std::vector<context::Context *> contexts);
  NNZType
  GetMaxFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetMaxFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetMaxFibersPerSliceCOO(std::vector<format::Format *> formats,
                          PreprocessParams *params);

  static NNZType
  GetMaxFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                     PreprocessParams *params);
protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class MinFibersPerSlice : public FeaturePreprocessType<NNZType> {

public:
  struct MinFibersPerSliceParams : PreprocessParams {};

  MinFibersPerSlice();

  MinFibersPerSlice(const MinFibersPerSlice &d);

  MinFibersPerSlice(const std::shared_ptr<MinFibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~MinFibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetMinFibersPerSliceCached(format::Format *format,
                             std::vector<context::Context *> contexts);
  NNZType
  GetMinFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetMinFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetMinFibersPerSliceCOO(std::vector<format::Format *> formats,
                          PreprocessParams *params);

  static NNZType
  GetMinFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                     PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class AvgFibersPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct AvgFibersPerSliceParams : PreprocessParams {};

  AvgFibersPerSlice();

  AvgFibersPerSlice(const AvgFibersPerSlice &d);

  AvgFibersPerSlice(const std::shared_ptr<AvgFibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~AvgFibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetAvgFibersPerSliceCached(format::Format *format,
                             std::vector<context::Context *> contexts);
  FeatureType
  GetAvgFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetAvgFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetAvgFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                     PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType>
class DevFibersPerSlice : public FeaturePreprocessType<NNZType> {

public:
  struct DevFibersPerSliceParams : PreprocessParams {};

  DevFibersPerSlice();

  DevFibersPerSlice(const DevFibersPerSlice &d);

  DevFibersPerSlice(const std::shared_ptr<DevFibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~DevFibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, NNZType>
  GetDevFibersPerSliceCached(format::Format *format,
                             std::vector<context::Context *> contexts);
  NNZType
  GetDevFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  NNZType
  GetDevFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static NNZType
  GetDevFibersPerSliceCOO(std::vector<format::Format *> formats,
                          PreprocessParams *params);

  static NNZType
  GetDevFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                     PreprocessParams *params);

protected:
  void Register();
};


template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class StdFibersPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct StdFibersPerSliceParams : PreprocessParams {};

  StdFibersPerSlice();

  StdFibersPerSlice(const StdFibersPerSlice &d);

  StdFibersPerSlice(const std::shared_ptr<StdFibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~StdFibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetStdFibersPerSliceCached(format::Format *format,
                             std::vector<context::Context *> contexts);
  FeatureType
  GetStdFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetStdFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetStdFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                     PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class CovFibersPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct CovFibersPerSliceParams : PreprocessParams {};

  CovFibersPerSlice();

  CovFibersPerSlice(const CovFibersPerSlice &d);

  CovFibersPerSlice(const std::shared_ptr<CovFibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~CovFibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetCovFibersPerSliceCached(format::Format *format,
                             std::vector<context::Context *> contexts);
  FeatureType
  GetCovFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetCovFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetCovFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                     PreprocessParams *params);

protected:
  void Register();
};

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class AdjFibersPerSlice : public FeaturePreprocessType<FeatureType> {

public:
  struct AdjFibersPerSliceParams : PreprocessParams {};

  AdjFibersPerSlice();

  AdjFibersPerSlice(const AdjFibersPerSlice &d);

  AdjFibersPerSlice(const std::shared_ptr<AdjFibersPerSliceParams> p);

  std::unordered_map<std::type_index, std::any> Extract(format::Format *format, std::vector<context::Context *> c);

  std::vector<std::type_index>
  get_sub_ids();

  std::vector<ExtractableType *>
  get_subs();

  std::type_index get_feature_id_static();

  ~AdjFibersPerSlice() = default;

  std::tuple<std::vector<format::Format *>, FeatureType>
  GetAdjFibersPerSliceCached(format::Format *format,
                             std::vector<context::Context *> contexts);
  FeatureType
  GetAdjFibersPerSlice(
      format::Format *format, std::vector<context::Context *> contexts);

  FeatureType
  GetAdjFibersPerSlice(
      object::Graph<IDType, NNZType, ValueType> *obj,
      std::vector<context::Context *> contexts);

  static FeatureType
  GetAdjFibersPerSliceHigherOrderCOO(std::vector<format::Format *> formats,
                                     PreprocessParams *params);

protected:
  void Register();
};


} // namespace sparsebase::preprocess
#ifdef _HEADER_ONLY
#include "sparsebase/preprocess/preprocess.cc"
#ifdef CUDA
#include "cuda/preprocess.cu"
#endif
#endif

#endif // SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
