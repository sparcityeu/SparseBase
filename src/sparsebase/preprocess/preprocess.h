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
#include <cmath>

namespace sparsebase::preprocess {

//! Functor used for hashing vectors of type_index values.
struct TypeIndexVectorHash {
  std::size_t operator()(const std::vector<std::type_index> &vf) const;
};

//! An abstraction for parameter objects used for preprocessing
struct PreprocessParams {};

//! A generic type for all preprocessing types
class PreprocessType {
  //! The parameter class used to pass parameters to this preprocessing
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
   * function. \param contexts Contexts available for execution of the
   * preprocessing. \param converter Converter object to be used for determining
   * available Format conversions. \return a tuple of a) the Function to use, and
   * b) a utils::converter::ConversionSchemaConditional indicating conversions
   * to be done on input Format objects.
   */
  std::tuple<Function, utils::converter::ConversionSchema>
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
   * \param convert_input whether or not to convert the input formats if that is
   * needed.
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
   * \param convert_input whether or not to convert the input formats if that is
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
  std::tuple<std::vector<std::vector<format::Format *>>, ReturnType>
  CachedExecute(PreprocessParams *params, utils::converter::Converter *sc,
                std::vector<context::Context *> contexts, bool convert_input, F sf, SF... sfs);
};

template <typename ReturnType>
class GenericPreprocessType : public FunctionMatcherMixin<ReturnType> {
protected:
public:
  int GetOutput(format::Format *csr, PreprocessParams *params,
                std::vector<context::Context *>, bool convert_input);
  std::tuple<std::vector<std::vector<format::Format *>>, int>
  GetOutputCached(format::Format *csr, PreprocessParams *params,
                  std::vector<context::Context *> contexts, bool convert_input);
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the inverse permutation array `inv_perm` of the input format; an array of size
   * `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of row/column `i`.
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the inverse permutation array `inv_perm` of the input format; an array of size
   * `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of row/column `i`.
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * the inverse permutation array `inv_perm` of the input format; an array of size
   * `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of row/column `i`.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  GetReorderCached(format::Format *csr,
                   std::vector<context::Context *> contexts,
                   bool convert_input);
  //! Generates a reordering inverse permutation of `format` with the given
  //! PreprocessParams object and using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param params a polymorphic pointer at a `PreprocessParams` object that
   * will contain hyperparameters used for reordering.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * the inverse permutation array `inv_perm` of the input format; an array of size
   * `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of row/column `i`.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  GetReorderCached(format::Format *csr, PreprocessParams *params,
                   std::vector<context::Context *> contexts,
                   bool convert_input);
  virtual ~ReorderPreprocessType();
};

//! Parameters used in DegreeReordering, namely whether or not degrees are ordered in ascending order.
struct DegreeReorderParams : PreprocessParams {
  bool ascending;
  DegreeReorderParams(bool ascending) : ascending(ascending) {}
};
//! Reordering preprocessing algorithm that reorders a format by representing it
//! as an adjacency matrix of a graph and ordering its vertices by degree
template <typename IDType, typename NNZType, typename ValueType>
class DegreeReorder : public ReorderPreprocessType<IDType> {
public:
  DegreeReorder(bool ascending);
  //! The hyperparameters used by the implementation functions of DegreeReorder
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
//! An empty struct used for the parameters of RCMReorder
struct RCMReorderParams : PreprocessParams {};

//! Reordering using the Reverse Cuthill-McKee algorithm:
//! https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
template <typename IDType, typename NNZType, typename ValueType>
class RCMReorder : public ReorderPreprocessType<IDType> {
  typedef typename std::make_signed<IDType>::type SignedID;

public:
  //! An empty struct used for the parameters of RCMReorder
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * a transformed Format object.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
  GetTransformationCached(format::Format *csr,
                          std::vector<context::Context *> contexts,
                          bool convert_input);
  //! Transforms `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param params a polymorphic pointer at a params object
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * a transformed Format object.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
  GetTransformationCached(format::Format *csr, PreprocessParams *params,
                          std::vector<context::Context *> contexts,
                          bool convert_input);
  virtual ~TransformPreprocessType();
};

//! The hyperparameters of the PermuteOrderTwo transformation.
/*!
 * The permutation vectors used for permuting the rows and the columns of a 2D format.
 * @tparam IDType the data type of row and column numbers (vertex IDs in the
 */
template <typename IDType>
struct PermuteOrderTwoParams : PreprocessParams {
  //! Permutation vector for reordering the rows.
  IDType *row_order;
  //! Permutation vector for reordering the columns.
  IDType *col_order;
  explicit PermuteOrderTwoParams(IDType *r_order, IDType* c_order) : row_order(r_order), col_order(c_order){};
};
template <typename IDType, typename NNZType, typename ValueType>
class PermuteOrderTwo : public TransformPreprocessType<
                    format::FormatOrderTwo<IDType, NNZType, ValueType>,
                    format::FormatOrderTwo<IDType, NNZType, ValueType>> {
public:
  PermuteOrderTwo(IDType *, IDType *);
  explicit PermuteOrderTwo(PermuteOrderTwoParams<IDType>);
  //! Struct used to store permutation vectors used by each instance of PermuteOrderTwo
  typedef PermuteOrderTwoParams<IDType> ParamsType;

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


//! The hyperparameters of the PermuteOrderTwo transformation.
/*!
 * The permutation vectors used for permuting the rows and the columns of a 2D format.
 * @tparam IDType the data type of row and column numbers (vertex IDs in the
 */
template <typename IDType>
struct PermuteOrderOneParams : PreprocessParams {
  //! Permutation vector
  IDType *order;
  explicit PermuteOrderOneParams(IDType *order) : order(order){};
};
template <typename IDType, typename ValueType>
class PermuteOrderOne
    : public TransformPreprocessType<format::FormatOrderOne<ValueType>,
                                     format::FormatOrderOne<ValueType>> {
public:
  PermuteOrderOne(IDType *);
  //! Struct used to store permutation vectors used by each instance of PermuteOrderTwo
  typedef PermuteOrderOneParams<IDType> ParamsType;
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

//! An empty struct used for the parameters of JaccardWeights
struct JaccardWeightsParams : PreprocessParams{};
//! Calculate the Jaccard Weights of the edges in a graph representation of a
//! format object
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class JaccardWeights : public FunctionMatcherMixin<format::Format *> {
public:
  //! An empty struct used for the parameters of JaccardWeights
  typedef JaccardWeightsParams ParamsType;
  JaccardWeights();
  JaccardWeights(ParamsType);
  //! Take a single Format object representating a graph and get the Jaccard
  //! Weights as a 1D format object
  /*!
   *
   * @param format input format object representing a graph
   * @param convert_input whether or not to convert the input format if that is
   * needed.
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

//! An empty struct used for the parameters of DegreeDistribution
struct DegreeDistributionParams : PreprocessParams {};
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
  //! An empty struct used for the parameters of DegreeDistribution
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
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
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * features. \return an array of size format.get_dimensions()[0] where element
   * i is the degree distribution of the ith vertex in `formats`
   */
  FeatureType *
  GetDistribution(object::Graph<IDType, NNZType, ValueType> *object,
                  std::vector<context::Context *> contexts, bool convert_input);
  //! Degree distribution generation executor function that carries out function
  //! matching with cached outputs
  /*!
   * Generates the degree distribution of the passed format. If the input format
   * was converted to other format types, the converting results are also
   * returned with the output \param format a single format pointer to any
   * format \param contexts vector of contexts that can be used for extracting
   * features.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \return A tuple with the first element being a vector of Format*
   * where each pointer in the output points at the format that the corresponds
   * Format object from the the input was converted to. If an input Format
   * wasn't converted, the output pointer will point at nullptr. The second
   * element is an array of size format.get_dimensions()[0] where element i is
   * the degree distribution of the ith vertex in `formats`
   */
  std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
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

//! An empty struct used for the parameters of Degrees
struct DegreesParams : PreprocessParams {};
//! Count the degrees of every vertex in the graph representation of a format
//! object
template <typename IDType, typename NNZType, typename ValueType>
class Degrees : public FeaturePreprocessType<IDType *> {

public:
  //! An empty struct used for the parameters of Degrees
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

  //! Degree generation executor function that carries out function matching
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting
   * features.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \return an array of size format.get_dimensions()[0] where element
   * i is the degree of the ith vertex in `format`
   */
  IDType *GetDegrees(format::Format *format,
                     std::vector<context::Context *> contexts, bool convert_input);
  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  //! Degree generation executor function that carries out function matching with cached output
  /*!
   *
   * \param format a single format pointer to any format
   * \param contexts vector of contexts that can be used for extracting
   * features.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \return an array of size format.get_dimensions()[0] where element
   * i is the degree of the ith vertex in `format`
   */
  GetDegreesCached(format::Format *format,
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

//! An empty struct used for the parameters of Degrees_DegreeDistribution
struct Params : PreprocessParams {};
//! Find the degree and degree distribution of each vertex in the graph
//! representation of a format object
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class Degrees_DegreeDistribution
    : public FeaturePreprocessType<
          std::unordered_map<std::type_index, std::any>> {
  //! An empty struct used for the parameters of Degrees_DegreeDistribution
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
   * features.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \return a map with two (type_index, any) pairs. One is a degrees
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
   * \param params a PreprocessParams pointer, though it
   * is not used in the function
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
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * partitioning.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @returns An IDType array where the i-th index contains the ID for the partitioning i belongs to.
   */
  IDType* Partition(format::Format * format, std::vector<context::Context*> contexts, bool convert_input);

  //! Performs a partition operation using the parameters supplied by the user
  /*!
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * partitioning.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @returns An IDType array where the i-th index contains the ID for the partitioning i belongs to
   */
  IDType* Partition(format::Format *format, PreprocessParams *params,
                     std::vector<context::Context *> contexts, bool convert_input);
  virtual ~PartitionPreprocessType();
};


#ifdef USE_METIS

namespace metis {
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

}
//! Parameters for metis partitioning
/*!
 * This struct replaces the options array of METIS
 * The names of the options are identical to the array
 * and can be found here: http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
 */
struct MetisParams : PreprocessParams{
  int64_t num_partitions = 2;
  int64_t ptype = metis::METIS_PTYPE_KWAY;
  int64_t objtype = metis::METIS_OBJTYPE_CUT;
  int64_t ctype = metis::METIS_CTYPE_RM;
  int64_t iptype = metis::METIS_IPTYPE_GROW;
  int64_t rtype = metis::METIS_RTYPE_FM;
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

};

#endif


class GraphFeatureBase {
public:

  //! Calculates the degree distribution of every vertex represented by the FormerOrderTwo object `format`.
  /*!
   * @tparam FeatureType data type used for storing degree distribution values.
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return an array of type `FeatureType*` size format->get_dimensions()[0] with the degree distribution of each vertex.
   */
  template <typename FeatureType, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static FeatureType *GetDegreeDistribution(
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input) {
    DegreeDistribution<AutoIDType, AutoNNZType, AutoValueType, FeatureType> deg_dist;
    return deg_dist.GetDistribution(format, contexts, convert_input);
  }

  //! Calculates the degree distribution of every vertex represented by the FormerOrderTwo object `format` with cached output.
  /*!
   * @tparam FeatureType data type used for storing degree distribution values.
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @return An std::pair with the second element being an array of type `FeatureType*` size format->get_dimensions()[0]
   * with the degree distribution of each vertex, and the first being a vector of all the formats
   * generated by converting the input (if such conversions were needed to execute the permutation).
   */
  template <typename FeatureType, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>, FeatureType*> GetDegreeDistributionCached(format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format, std::vector<context::Context*>contexts){
    DegreeDistribution<AutoIDType, AutoNNZType, AutoValueType, FeatureType> deg_dist;
    auto output = deg_dist.GetDistributionCached(format, contexts, true);
    std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>(intermediate_format);
    });
    return std::make_pair(converted_formats, std::get<1>(output));
  }
  //! Calculates the degree count of every vertex represented by the FormerOrderTwo object `format`.
  /*!
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return an array of size format->get_dimensions()[0] with the degree of each vertex.
   */
  template <typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static AutoNNZType *
  GetDegrees(format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
             std::vector<context::Context *> contexts, bool convert_input) {
    Degrees<AutoIDType, AutoNNZType, AutoValueType> deg_dist;
    return deg_dist.GetDegrees(format, contexts, convert_input);
  }
  //! Calculates the degree count of every vertex represented by the FormerOrderTwo object `format` with cached output.
  /*!
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return An std::pair with the second element being an array of type `FeatureType*` size format->get_dimensions()[0]
   * with the degree of each vertex, and the first being a vector of all the formats
   * generated by converting the input (if such conversions were needed to execute the permutation).
   */
  template <typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>, AutoNNZType*> GetDegreesCached(format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format, std::vector<context::Context*>contexts){
    Degrees<AutoIDType, AutoNNZType, AutoValueType> deg_dist;
    auto output = deg_dist.GetDegreesCached(format, contexts, true);
    std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>(intermediate_format);
    });
    return std::make_pair(converted_formats, std::get<1>(output));
  }
};

//! A class containing the interface for reordering and permuting data.
/*!
 * The class contains all the functionalities needed for reordering. That includes a function generate reordering
 * permutations from data, functions to permute data using a permutation vector, and a function to inverse the
 * permutation of data. In the upcoming release, ReorderBase will include functions to extract feeatures from
 * permutation vectors and permuted data.
 */
class ReorderBase {
public:
  //! Generates a permutation array from a FormatOrderTwo object using the Reordering class `Reordering`.
  /*!
   *
   * @tparam Reordering a reordering class defining a reordering algorithm. For a full list of available reordering algorithms, please check [here](../pages/getting_started/available.html).
   * @param params a struct containing the parameters specific for the reordering algorithm `Reordering`. Please check the documentation of each reordering for the specifications of its parameters.
   * @param format FormatOrderTwo object to be used to generate permutation array.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the permutation array.
   */
  template <template <typename, typename, typename> typename Reordering, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static AutoIDType* Reorder(typename Reordering<AutoIDType, AutoNNZType, AutoValueType>::ParamsType params, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_input){
    static_assert(std::is_base_of_v<ReorderPreprocessType<AutoIDType>, Reordering<AutoIDType, AutoNNZType, AutoValueType>>, "You must pass a reordering function (with base ReorderPreprocessType) to ReorderBase::Reorder");
    static_assert(!std::is_same_v<GenericReorder<AutoIDType, AutoNNZType, AutoValueType>, Reordering<AutoIDType, AutoNNZType, AutoValueType>>, "You must pass a reordering function (with base ReorderPreprocessType) to ReorderBase::Reorder");
    Reordering<AutoIDType, AutoNNZType, AutoValueType> reordering(params);
    return reordering.GetReorder(format, contexts, convert_input);
  }
  // TODO: add page for reordering
  //! Generates a permutation array from a FormatOrderTwo object using the Reordering class `Reordering` with cached output.
  /*!
   *
   * @tparam Reordering a reordering class defining a reordering algorithm. For a full list of available reordering algorithms, please check: xxx
   * @param params a struct containing the parameters specific for the reordering algorithm `Reordering`. Please check the documentation of each reordering for the specifications of its parameters.
   * @param format FormatOrderTwo object to be used to generate permutation array.
   * @param contexts vector of contexts that can be used for permutation.
   * @return An std::pair with the second element being the permutation array, and the first being a vector of all the formats
   * generated by converting the input (if such conversions were needed to execute the permutation).
   */
  template <template <typename, typename, typename> typename Reordering, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>, AutoIDType*> ReorderCached(typename Reordering<AutoIDType, AutoNNZType, AutoValueType>::ParamsType params, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts){
    static_assert(std::is_base_of_v<ReorderPreprocessType<AutoIDType>, Reordering<AutoIDType, AutoNNZType, AutoValueType>>, "You must pass a reordering function (with base ReorderPreprocessType) to ReorderBase::Reorder");
    static_assert(!std::is_same_v<GenericReorder<AutoIDType, AutoNNZType, AutoValueType>, Reordering<AutoIDType, AutoNNZType, AutoValueType>>, "You must pass a reordering function (with base ReorderPreprocessType) to ReorderBase::Reorder");
    Reordering<AutoIDType, AutoNNZType, AutoValueType> reordering(params);
    auto output = reordering.GetReorderCached(format, contexts, true);
    std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>(intermediate_format);
    });
    return std::make_pair(converted_formats, std::get<1>(output));
  }

  //! Permute a two-dimensional format row- and column-wise using a single permutation array for both axes.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param order Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return The permuted format. By default, the function returns a pointer at a generic FormatOrderTwo object.
   * However, if the user passes a concrete FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be converted to that type. If not, the returned
   * object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>* Permute2D(AutoIDType* ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_input, bool convert_output=false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering, ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return out_format;
    else{
      if (convert_output)
        return out_format->template Convert<ReturnFormatType>();
      else
        return out_format->template As<ReturnFormatType>();
    }
  }

  //! Permute a two-dimensional format row- and column-wise using a single permutation array for both axes with cached output.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param ordering Permutation array to use when permuting rows and columns.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return An std::pair with the second element being the permuted format, and the first being a vector of all the formats generated
   * by converting the input (if such conversions were needed to execute the permutation). By default, the permuted object is returned as a pointer
   * at a generic FormatOrderTwo object. However, if the user passes a concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the returned format will be converted to
   * that type. If not, the returned object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>, ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>*> Permute2DCached(AutoIDType* ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_output = false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering, ordering);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>(intermediate_format);
    });
    if constexpr (std::is_same_v<ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return std::make_pair(converted_formats, std::get<1>(output));
    else {
      if (convert_output)
        return std::make_pair(
            converted_formats,
            std::get<1>(output)->template Convert<ReturnFormatType>());
      else
        return std::make_pair(
            converted_formats,
            std::get<1>(output)->template As<ReturnFormatType>());
    }
  }

  //! Permute a two-dimensional format row- and column-wise using a permutation array for each axis with cached output.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param row_ordering Permutation array to use when permuting rows.
   * \param col_ordering Permutation array to use when permuting col.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return An std::pair with the second element being the permuted format, and the first being a vector of all the formats generated
   * by converting the input (if such conversions were needed to execute the permutation). By default, the permuted object is returned as a pointer
   * at a generic FormatOrderTwo object. However, if the user passes a concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the returned format will be converted to
   * that type. If not, the returned object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>, ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>*> Permute2DRowColumnWiseCached(AutoIDType* row_ordering, AutoIDType* col_ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_output = false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(row_ordering, col_ordering);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>(intermediate_format);
    });
    if constexpr (std::is_same_v<ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return std::make_pair(converted_formats, std::get<1>(output));
    else {
      if (convert_output)
        return std::make_pair(
            converted_formats,
            std::get<1>(output)->template Convert<ReturnFormatType>());
      else
        return std::make_pair(
            converted_formats,
            std::get<1>(output)->template As<ReturnFormatType>());
    }
  }

  //! Permute a two-dimensional format row- and column-wise using a permutation array for each axis.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param order Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return The permuted format. By default, the function returns a pointer at a generic FormatOrderTwo object.
   * However, if the user passes a concrete FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be converted to that type. If not, the returned
   * object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>* Permute2DRowColumnWise(AutoIDType* row_ordering, AutoIDType* col_ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_input, bool convert_output = false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(row_ordering, col_ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return out_format;
    else{
      if (convert_output)
        return out_format->template Convert<ReturnFormatType>();
      else
        return out_format->template As<ReturnFormatType>();
    }
  }

  //! Permute a two-dimensional format row-wise using a permutation array.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param order Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return The permuted format. By default, the function returns a pointer at a generic FormatOrderTwo object.
   * However, if the user passes a concrete FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be converted to that type. If not, the returned
   * object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>* Permute2DRowWise(AutoIDType* ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_input, bool convert_output = false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering, nullptr);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return out_format;
    else{
      if (convert_output)
        return out_format->template Convert<ReturnFormatType>();
      else
        return out_format->template As<ReturnFormatType>();
    }
  }

  //! Permute a two-dimensional format row-wise using a permutation array with cached output.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param ordering Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return An std::pair with the second element being the permuted format, and the first being a vector of all the formats generated
   * by converting the input (if such conversions were needed to execute the permutation). By default, the permuted object is returned as a pointer
   * at a generic FormatOrderTwo object. However, if the user passes a concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the returned format will be converted to
   * that type. If not, the returned object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename RelativeReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>, RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>*> Permute2DRowWiseCached(AutoIDType* ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_output = false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering, nullptr);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>(intermediate_format);
    });
    if constexpr (std::is_same_v<RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return std::make_pair(converted_formats, std::get<1>(output));
    else{
      if (convert_output)
        return std::make_pair(converted_formats, std::get<1>(output)->template Convert<RelativeReturnFormatType>());
      else
        return std::make_pair(converted_formats,
            std::get<1>(output)->template As<RelativeReturnFormatType>());
    }
  }

  //! Permute a two-dimensional format column-wise using a permutation array.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param order Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return The permuted format. By default, the function returns a pointer at a generic FormatOrderTwo object.
   * However, if the user passes a concrete FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be converted to that type. If not, the returned
   * object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>* Permute2DColWise(AutoIDType* ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_input, bool convert_output = false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(nullptr, ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    if constexpr (std::is_same_v<ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return out_format;
    else{
      if (convert_output)
        return out_format->template Convert<ReturnFormatType>();
      else
        return out_format->template As<ReturnFormatType>();
    }
  }

  //! Permute a two-dimensional format column-wise using a permutation array with cached output.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the return pointer type. Default is FormatOrderTwo.
   * \param ordering Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return An std::pair with the second element being the permuted format, and the first being a vector of all the formats generated
   * by converting the input (if such conversions were needed to execute the permutation). By default, the permuted object is returned as a pointer
   * at a generic FormatOrderTwo object. However, if the user passes a concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the returned format will be converted to
   * that type. If not, the returned object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename> typename ReturnFormatType = format::FormatOrderTwo, typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>, ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>*> Permute2DColWiseCached(AutoIDType* ordering, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>* format, std::vector<context::Context*> contexts, bool convert_output = false){
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(nullptr, ordering);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>*>(intermediate_format);
    });
    if constexpr (std::is_same_v<ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>, format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType>>)
      return std::make_pair(converted_formats, std::get<1>(output));
    else{
      if (convert_output)
        return std::make_pair(converted_formats, std::get<1>(output)->template Convert<ReturnFormatType>());
      else
        return std::make_pair(converted_formats,
            std::get<1>(output)->template As<ReturnFormatType>());
    }
  }

  //! Permute a one-dimensional format using a permutation array.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderOne. Defines the return pointer type. Default is FormatOrderOne.
   * \param order Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_input whether or not to convert the input format if that is
   * needed.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return The permuted format. By default, the function returns a pointer at a generic FormatOrderOne object.
   * However, if the user passes a concrete FormatOrderOne class as the templated parameter `ReturnFormatType`, e.g.
   * format::Array, then if `convert_output` is true, the returned format will be converted to that type. If not, the returned
   * object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename> typename ReturnFormatType = format::FormatOrderOne, typename AutoIDType, typename AutoValueType>
  static ReturnFormatType<AutoValueType>* Permute1D(AutoIDType* ordering, format::FormatOrderOne<AutoValueType>* format, std::vector<context::Context*> context, bool convert_inputs, bool convert_output = false){
    PermuteOrderOne<AutoIDType, AutoValueType> perm(ordering);
    auto out_format = perm.GetTransformation(format, context, convert_inputs);
    if constexpr (std::is_same_v<ReturnFormatType<AutoValueType>, format::FormatOrderOne<AutoValueType>>)
      return out_format;
    else{
      if (convert_output)
        return out_format->template Convert<ReturnFormatType>();
      else
        return out_format->template As<ReturnFormatType>();
    }
  }

  //! Permute a one-dimensional format using a permutation array with cached output.
  /*!
   *
   * \tparam ReturnFormatType a child class of type FormatOrderOne. Defines the return pointer type. Default is FormatOrderOne.
   * \param order Permutation array.
   * \param format object to be permuted.
   * \param contexts vector of contexts that can be used for permutation.
   * \param convert_output if true, the returned object will be converted to `ReturnFormatType`. Otherwise, the returned object
   * will be cast to `ReturnFormatType`, and if the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * \return An std::pair with the second element being the permuted format, and the first being a vector of all the formats generated
   * by converting the input (if such conversions were needed to execute the permutation). By default, the permuted object is returned as a pointer
   * at a generic FormatOrderOne object. However, if the user passes a FormatOrderOne class as the templated parameter
   * `ReturnFormatType`, e.g. format::Array, then if `convert_output` is true, the returned format will be converted to
   * that type. If not, the returned object will only be cast to that type (if casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename> typename ReturnFormatType = format::FormatOrderOne, typename AutoIDType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderOne<AutoValueType>*>, ReturnFormatType<AutoValueType>*> Permute1DCached(AutoIDType* ordering, format::FormatOrderOne<AutoValueType>* format, std::vector<context::Context*> context, bool convert_output = false){
    PermuteOrderOne<AutoIDType, AutoValueType> perm(ordering);
    auto output = perm.GetTransformationCached(format, context, true);
    std::vector<format::FormatOrderOne<AutoValueType>*> converted_formats;
    std::transform(std::get<0>(output)[0].begin(),std::get<0>(output)[0].end(), std::back_inserter(converted_formats), [](format::Format* intermediate_format){
      return static_cast<format::FormatOrderOne<AutoValueType>*>(intermediate_format);
    });
    if constexpr (std::is_same_v<ReturnFormatType<AutoValueType>, format::FormatOrderOne<AutoValueType>>)
      return std::make_pair(converted_formats, std::get<1>(output));
    else{
      if (convert_output)
        return std::make_pair(converted_formats, std::get<1>(output)->template Convert<ReturnFormatType>());
      else
        return std::make_pair(converted_formats,
            std::get<1>(output)->template As<ReturnFormatType>());
    }
  }

  //! Takes a permutation array and its length and inverses it.
  /*!
   * Takes a permutation array and its length and inverses it. If a format `A` was permuted with `perm`
   * into object `B`, then permuting `B` with the inverse permutation returns its order to `A`.
   * @param perm a permutation array of length `length`
   * @param length the length of the permutation array
   * @return a permutation array of length `length` that is the inverse of `perm`, i.e. can be used
   * to reverse a permutation done by `perm`.
   */
  template <typename AutoIDType, typename AutoNumType>
  static AutoIDType * InversePermutation(AutoIDType*perm, AutoNumType length){
    static_assert(std::is_integral_v<AutoNumType>, "Length of the permutation array must be an integer");
    auto inv_perm = new AutoIDType[length];
    for (AutoIDType i = 0; i < length; i++){
      inv_perm[perm[i]] = i;
    }
    return inv_perm;
  }

};

enum BitMapSize{
  BitSize16 = 16,
  BitSize32 = 32/*,
  BitSize64 = 64*/ //at the moment, using 64 bits is not working as intended
};
//! Params struct for GrayReorder
struct GrayReorderParams : PreprocessParams {
  BitMapSize resolution;
  int nnz_threshold;
  int sparse_density_group_size;
  explicit GrayReorderParams(){}
  GrayReorderParams(BitMapSize r, int nnz_thresh, int group_size): resolution(r), nnz_threshold(nnz_threshold), sparse_density_group_size(group_size){}
};

template <typename IDType, typename NNZType, typename ValueType>
class GrayReorder : public ReorderPreprocessType<IDType> {
  typedef std::pair<IDType, unsigned long> row_grey_pair;

public:
  //! Parameter type for GrayReorder
  typedef GrayReorderParams ParamsType;
  GrayReorder(BitMapSize resolution, int nnz_threshold,
              int sparse_density_group_size);
  explicit GrayReorder(GrayReorderParams);

protected:
  static bool desc_comparator(const row_grey_pair &l, const row_grey_pair &r);

  static bool asc_comparator(const row_grey_pair &l, const row_grey_pair &r);

  // not sure if all IDTypes work for this
  static unsigned long grey_bin_to_dec(unsigned long n);

  static void print_dec_in_bin(unsigned long n, int size);

  // not sure if all IDTypes work for this
  static unsigned long bin_to_grey(unsigned long n);

  // bool is_banded(std::vector<format::Format *> input_sf, int band_size = -1,
  // std::vector<IDType> order) {
  static bool is_banded(int nnz, int n_cols, NNZType *row_ptr, IDType *cols,
                        std::vector<IDType> order, int band_size = -1);

  static IDType *GrayReorderingCSR(std::vector<format::Format *> input_sf,
                                   PreprocessParams *poly_params);
};
template <typename ReturnType, class PreprocessingImpl, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
template <typename F, typename... SF>
std::tuple<std::vector<std::vector<format::Format *>>, ReturnType>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::CachedExecute(PreprocessParams *params,
                               utils::converter::Converter *sc,
                               std::vector<context::Context *>
                               contexts, bool convert_input,
                               F format, SF... formats) {
  ConversionMap map = this->map_to_function_;
  // pack the Formats into a vector
  std::vector<format::Format *> packed_formats =
      PackObjects(format, formats...);
  // pack the types of Formats into a vector
  std::vector<std::type_index> packed_format_types;
  for (auto f : packed_formats)
    packed_format_types.push_back(f->get_format_id());
  // get conversion schema
  std::tuple<Function, utils::converter::ConversionSchema> ret =
      GetFunction(packed_formats, packed_format_types, map, contexts, sc);
  Function func = std::get<0>(ret);
  utils::converter::ConversionSchema cs = std::get<1>(ret);
  // carry out conversion
  // ready_formats contains the format to use in preprocessing
  if (!convert_input) {
    for (const auto &conversion_chain : cs) {
      if (conversion_chain)
        throw utils::DirectExecutionNotAvailableException(
            packed_format_types, this->GetAvailableFormats());
    }
  }
  std::vector<std::vector<format::Format *>> all_formats =
      sparsebase::utils::converter::Converter::ApplyConversionSchema(
          cs, packed_formats);
  // The formats that will be used in the preprocessing implementation function calls
  std::vector<format::Format *> final_formats;
  std::transform(all_formats.begin(), all_formats.end(),
                 std::back_inserter(final_formats),
                 [](std::vector<format::Format *> conversion_chain) {
                   return conversion_chain.back();
                 });
  // Formats that are used to get to the final formats
  std::vector<std::vector<format::Format *>> intermediate_formats;
  std::transform(all_formats.begin(), all_formats.end(),
                 std::back_inserter(intermediate_formats),
                 [](std::vector<format::Format *> conversion_chain) {
                   if (conversion_chain.size() > 1)
                     return std::vector<format::Format *>(conversion_chain.begin() + 1,
                                                  conversion_chain.end());
                   return std::vector<format::Format *>();
                 });
  // carry out the correct call
  return std::make_tuple(intermediate_formats, func(final_formats, params));
}

template <typename ReturnType, class PreprocessingImpl, typename Function,
    typename Key, typename KeyHash, typename KeyEqualTo>
template <typename F, typename... SF>
ReturnType
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Function, Key, KeyHash,
    KeyEqualTo>::Execute(PreprocessParams *params,
                         utils::converter::Converter *sc,
                         std::vector<context::Context *>
                         contexts, bool convert_input,
                         F sf, SF... sfs) {
  auto cached_output = CachedExecute(params, sc, contexts, convert_input, sf, sfs...);
  auto converted_format_chains = std::get<0>(cached_output);
  auto return_object = std::get<1>(cached_output);
  for (const auto& converted_format_chain : converted_format_chains) {
    for (const auto& converted_format : converted_format_chain)
      delete converted_format;
  }
  return return_object;
}

template <typename ReturnType, class PreprocessingImpl, typename Key,
    typename KeyHash, typename KeyEqualTo, typename Function>
template <typename Object>
std::vector<Object>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Key, KeyHash, KeyEqualTo,
    Function>::PackObjects(Object object) {
  return {object};
}
template <typename ReturnType, class PreprocessingImpl, typename Key,
    typename KeyHash, typename KeyEqualTo, typename Function>
template <typename Object, typename... Objects>
std::vector<Object>
FunctionMatcherMixin<ReturnType, PreprocessingImpl, Key, KeyHash, KeyEqualTo,
    Function>::PackObjects(Object object, Objects... objects) {
  std::vector<Object> v = {object};
  std::vector<Object> remainder = PackObjects(objects...);
  for (auto i : remainder) {
    v.push_back(i);
  }
  return v;
}

} // namespace sparsebase::preprocess
#ifdef _HEADER_ONLY
#include "sparsebase/preprocess/preprocess.cc"
#ifdef USE_CUDA
#include "cuda/preprocess.cu"
#endif
#endif

#endif // SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
