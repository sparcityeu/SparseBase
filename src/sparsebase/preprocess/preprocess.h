/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
#define SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
#include <any>
#include <cmath>
#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

#include "sparsebase/object/object.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/utils/function_matcher_mixin.h"

namespace sparsebase::preprocess {

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
  virtual std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *> contexts,
      bool convert_input) = 0;
  //! Returns the std::type_index of this class
  virtual std::type_index get_id() = 0;
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
  //! Get a std::shared_ptr at the Parameters of this object
  /*!
   *
   * \return An std::shared_ptr at the same Parameters instance of this
   * object (not a copy)
   */
  virtual std::shared_ptr<utils::Parameters> get_params() = 0;
  //! Get an std::shared_ptr at a Parameters of one of the ExtractableType
  //! classes fused into this class
  /*!
   * Returns a std::shared_ptr at a Parameters object belonging to one of
   * the ExtractableType classes fused into this class \param feature_extractor
   * std::type_index identifying the ExtractableType within this class whose
   * parameters are requested \return an std::shared_ptr at the Parameters
   * corresponding feature_extractor
   */
  virtual std::shared_ptr<utils::Parameters> get_params(
      std::type_index feature_extractor) = 0;
  //! Set the parameters of one of ExtractableType classes fusued into this
  //! classes.
  /*!
   * \param feature_extractor std::type_index identifying the ExtractableType
   * class fusued into this class whose parameters are to be set. \param params
   * an std::shared_ptr at the Parameters belonging to the class
   * feature_extractor
   */
  virtual void set_params(std::type_index feature_extractor,
                          std::shared_ptr<utils::Parameters> params) = 0;
  virtual ~ExtractableType() = default;

 protected:
  //! a pointer at the Parameters of this class
  std::shared_ptr<utils::Parameters> params_;
  //! A key-value map of Parameters, one for each of the ExtractableType
  //! classes fused into this class
  std::unordered_map<std::type_index, std::shared_ptr<utils::Parameters>> pmap_;
};


template <typename ReturnType>
class GenericPreprocessType : public utils::FunctionMatcherMixin<ReturnType> {
 protected:
 public:
  int GetOutput(format::Format *csr, utils::Parameters *params,
                std::vector<context::Context *>, bool convert_input);
  std::tuple<std::vector<std::vector<format::Format *>>, int> GetOutputCached(
      format::Format *csr, utils::Parameters *params,
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
class ReorderPreprocessType : public utils::FunctionMatcherMixin<IDType *> {
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
   * @return the inverse permutation array `inv_perm` of the input format; an
   * array of size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID
   * of row/column `i`.
   */
  IDType *GetReorder(format::Format *format,
                     std::vector<context::Context *> contexts,
                     bool convert_input);
  //! Generates a reordering inverse permutation of `format` with the given
  //! Parameters object and using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param params a polymorphic pointer at a `Parameters` object that
   * will contain hyperparameters used for reordering.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the inverse permutation array `inv_perm` of the input format; an
   * array of size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID
   * of row/column `i`.
   */
  IDType *GetReorder(format::Format *format, utils::Parameters *params,
                     std::vector<context::Context *> contexts,
                     bool convert_input);
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
   * the inverse permutation array `inv_perm` of the input format; an array of
   * size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of
   * row/column `i`.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  GetReorderCached(format::Format *csr,
                   std::vector<context::Context *> contexts,
                   bool convert_input);
  //! Generates a reordering inverse permutation of `format` with the given
  //! Parameters object and using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param params a polymorphic pointer at a `Parameters` object that
   * will contain hyperparameters used for reordering.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * the inverse permutation array `inv_perm` of the input format; an array of
   * size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of
   * row/column `i`.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  GetReorderCached(format::Format *csr, utils::Parameters *params,
                   std::vector<context::Context *> contexts,
                   bool convert_input);
  virtual ~ReorderPreprocessType();
};

//! Parameters used in DegreeReorder, namely whether or not degrees are ordered
//! in ascending order.
struct DegreeReorderParams : utils::Parameters {
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
                                     utils::Parameters *params);
};

//! A generic reordering class that the user instantiate and then register their
//! own functions to.
template <typename IDType, typename NNZType, typename ValueType>
class GenericReorder : public ReorderPreprocessType<IDType> {
 public:
  typedef utils::Parameters ParamsType;
  GenericReorder();
};
//! An empty struct used for the parameters of RCMReorder
struct RCMReorderParams : utils::Parameters {};

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
                               utils::Parameters *);
};

//! Transforms a format according to an inverse permutation of its rows/columns
template <typename InputFormatType, typename ReturnFormatType>
class TransformPreprocessType
    : public utils::FunctionMatcherMixin<ReturnFormatType *> {
 public:
  TransformPreprocessType() {
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
                                      std::vector<context::Context *>,
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
   * @return a transformed Format object
   */
  ReturnFormatType *GetTransformation(format::Format *csr, utils::Parameters *params,
                                      std::vector<context::Context *>,
                                      bool convert_input);
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
  GetTransformationCached(format::Format *csr, utils::Parameters *params,
                          std::vector<context::Context *> contexts,
                          bool convert_input);
  virtual ~TransformPreprocessType();
};

//! The hyperparameters of the PermuteOrderTwo transformation.
/*!
 * The permutation vectors used for permuting the rows and the columns of a 2D
 * format.
 * @tparam IDType the data type of row and column numbers (vertex IDs in the
 */
template <typename IDType>
struct PermuteOrderTwoParams : utils::Parameters {
  //! Permutation vector for reordering the rows.
  IDType *row_order;
  //! Permutation vector for reordering the columns.
  IDType *col_order;
  explicit PermuteOrderTwoParams(IDType *r_order, IDType *c_order)
      : row_order(r_order), col_order(c_order){};
};
template <typename IDType, typename NNZType, typename ValueType>
class PermuteOrderTwo
    : public TransformPreprocessType<
          format::FormatOrderTwo<IDType, NNZType, ValueType>,
          format::FormatOrderTwo<IDType, NNZType, ValueType>> {
 public:
  PermuteOrderTwo(IDType *, IDType *);
  explicit PermuteOrderTwo(PermuteOrderTwoParams<IDType>);
  //! Struct used to store permutation vectors used by each instance of
  //! PermuteOrderTwo
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
  static format::FormatOrderTwo<IDType, NNZType, ValueType> *PermuteOrderTwoCSR(
      std::vector<format::Format *> formats, utils::Parameters *);
};

//! The hyperparameters of the PermuteOrderTwo transformation.
/*!
 * The permutation vectors used for permuting the rows and the columns of a 2D
 * format.
 * @tparam IDType the data type of row and column numbers (vertex IDs in the
 */
template <typename IDType>
struct PermuteOrderOneParams : utils::Parameters {
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
  //! Struct used to store permutation vectors used by each instance of
  //! PermuteOrderTwo
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
  static format::FormatOrderOne<ValueType> *PermuteArray(
      std::vector<format::Format *> formats, utils::Parameters *);
};

//! A class that does feature extraction.
/*!
 * An ExtractableType class that has a function matching
 * capability. In other words, an Extractable to which implementation functions
 * can be added and used. @tparam FeatureType the return type of feature
 * extraction
 */
template <typename FeatureType>
class FeaturePreprocessType
    : public utils::FunctionMatcherMixin<FeatureType,
                                  ExtractableType> {
 public:
  std::shared_ptr<utils::Parameters> get_params() override;
  std::shared_ptr<utils::Parameters> get_params(std::type_index) override;
  void set_params(std::type_index, std::shared_ptr<utils::Parameters>) override;
  std::type_index get_id() override;
  ~FeaturePreprocessType();
};

//! An empty struct used for the parameters of JaccardWeights
struct JaccardWeightsParams : utils::Parameters {};
//! Calculate the Jaccard Weights of the edges in a graph representation of a
//! format object
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class JaccardWeights : public utils::FunctionMatcherMixin<format::Format *> {
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
                                    std::vector<context::Context *>,
                                    bool convert_input);
#ifdef USE_CUDA
  //! Take a CUDACSR representating a graph and get the Jaccard Weights as a
  //! CUDAArray
  /*!
   *
   * @param formats a vector of size 1 with formats[0] being CUDACSR
   * representing a graph
   * @param params a polymorphic pointer at a Parameters (not used)
   * @return a 1D array (CUDAArray) where element i in the array is the Jaccard
   * Weight of edge i in the graph (ith non-zero)
   */
  static format::Format *GetJaccardWeightCUDACSR(
      std::vector<format::Format *> formats, utils::Parameters *params);
#endif
  ~JaccardWeights();
};

//! An empty struct used for the parameters of DegreeDistribution
struct DegreeDistributionParams : utils::Parameters {};
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
  virtual std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input);
  virtual std::vector<std::type_index> get_sub_ids();
  virtual std::vector<ExtractableType *> get_subs();
  static std::type_index get_id_static();

  //! Degree distribution generation executor function that carries out function
  //! matching
  /*!
   *
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * features. @return an array of size format.get_dimensions()[0] where element
   * i is the degree distribution of the ith vertex in `formats`
   */
  FeatureType *GetDistribution(format::Format *format,
                               std::vector<context::Context *> contexts,
                               bool convert_input);
  //! Degree distribution generation executor function that carries out function
  //! matching on a Graph
  /*!
   *
   * @param object a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * features. @return an array of size format.get_dimensions()[0] where element
   * i is the degree distribution of the ith vertex in `formats`
   */
  FeatureType *GetDistribution(
      object::Graph<IDType, NNZType, ValueType> *object,
      std::vector<context::Context *> contexts, bool convert_input);
  //! Degree distribution generation executor function that carries out function
  //! matching with cached outputs
  /*!
   * Generates the degree distribution of the passed format. If the input format
   * was converted to other format types, the converting results are also
   * returned with the output @param format a single format pointer to any
   * format @param contexts vector of contexts that can be used for extracting
   * features.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*
   * where each pointer in the output points at the format that the corresponds
   * Format object from the the input was converted to. If an input Format
   * wasn't converted, the output pointer will point at nullptr. The second
   * element is an array of size format.get_dimensions()[0] where element i is
   * the degree distribution of the ith vertex in `formats`
   */
  std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
  GetDistributionCached(format::Format *format,
                        std::vector<context::Context *> contexts,
                        bool convert_input);

  static FeatureType
      *
      //! Degree distribution generation implementation function for CSRs
      /*!
       *
       * @param format a single format pointer to any format
       * @return an array of size formats[0].get_dimensions()[0] where element i
       * is the degree distribution of the ith vertex in `formats[0]`
       */
      GetDegreeDistributionCSR(std::vector<format::Format *> formats,
                               utils::Parameters *params);
  ~DegreeDistribution();

 protected:
  void Register();
};

//! An empty struct used for the parameters of Degrees
struct DegreesParams : utils::Parameters {};
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
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<ExtractableType *> get_subs() override;
  static std::type_index get_id_static();

  //! Degree generation executor function that carries out function matching
  /*!
   *
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * features.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return an array of size format.get_dimensions()[0] where element
   * i is the degree of the ith vertex in `format`
   */
  IDType *GetDegrees(format::Format *format,
                     std::vector<context::Context *> contexts,
                     bool convert_input);
  std::
      tuple<std::vector<std::vector<format::Format *>>, IDType *>
      //! Degree generation executor function that carries out function matching
      //! with cached output
      /*!
       *
       * @param format a single format pointer to any format
       * @param contexts vector of contexts that can be used for extracting
       * features.
       * @param convert_input whether or not to convert the input format if that
       * is needed. @return an array of size format.get_dimensions()[0] where
       * element i is the degree of the ith vertex in `format`
       */
      GetDegreesCached(format::Format *format,
                       std::vector<context::Context *> contexts,
                       bool convert_input);
  //! Degree generation implementation function for CSRs
  /*!
   *
   * @param formats A vector containing a single format pointer that should
   * point at a CSR object @param params a Parameters pointer, though it
   * is not used in the function @return an array of size
   * formats[0].get_dimensions()[0] where element i is the degree of the ith
   * vertex in `formats[0]`
   */
  static IDType *GetDegreesCSR(std::vector<format::Format *> formats,
                               utils::Parameters *params);
  ~Degrees();

 protected:
  void Register();
};

//! An empty struct used for the parameters of Degrees_DegreeDistribution
struct Params : utils::Parameters {};
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
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<ExtractableType *> get_subs() override;
  static std::type_index get_id_static();

  //! Degree and degree distribution generation executor function that carries
  //! out function matching
  /*!
   *
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * features.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return a map with two (type_index, any) pairs. One is a degrees
   * array of type IDType*, and one is a degree distribution array of type
   * FeatureType*. Both arrays have the respective metric of the ith vertex in
   * the ith array element.
   */
  std::unordered_map<std::type_index, std::any> Get(
      format::Format *format, std::vector<context::Context *> contexts,
      bool convert_input);

  //! Degree and degree distribution implementation function for CSRs
  /*!
   *
   * @param format a single format pointer to any format
   * @param params a utils::Parameters pointer, though it
   * is not used in the function
   * features. @return a map with two (type_index, any) pairs. One is a degrees
   * array of type IDType*, and one is a degree distribution array of type
   * FeatureType*. Both arrays have the respective metric of the ith vertex in
   * the ith array element.
   */
  static std::unordered_map<std::type_index, std::any> GetCSR(
      std::vector<format::Format *> formats, utils::Parameters *params);
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
class PartitionPreprocessType : public utils::FunctionMatcherMixin<IDType *> {
 public:
  PartitionPreprocessType();

  //! Performs a partition operation using the default parameters
  /*!
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * partitioning.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @returns An IDType array where the i-th index contains the ID for the
   * partitioning i belongs to.
   */
  IDType *Partition(format::Format *format,
                    std::vector<context::Context *> contexts,
                    bool convert_input);

  //! Performs a partition operation using the parameters supplied by the user
  /*!
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * partitioning.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @returns An IDType array where the i-th index contains the ID for the
   * partitioning i belongs to
   */
  IDType *Partition(format::Format *format, utils::Parameters *params,
                    std::vector<context::Context *> contexts,
                    bool convert_input);
  virtual ~PartitionPreprocessType();
};

#ifdef USE_RABBIT_ORDER

#define BOOST_ATOMIC_DETAIL_NO_CXX11_IS_TRIVIALLY_COPYABLE
#define BOOST_ATOMIC_DETAIL_NO_HAS_UNIQUE_OBJECT_REPRESENTATIONS
#define BOOST_ATOMIC_NO_CLEAR_PADDING

struct RabbitReorderParams : utils::Parameters {};

template <typename IDType, typename NNZType, typename ValueType>
class RabbitReorder : public ReorderPreprocessType<IDType> {
 public:
  typedef RabbitReorderParams ParamsType;
  RabbitReorder();
  explicit RabbitReorder(RabbitReorderParams);

 protected:
  static IDType *CalculateReorderCSR(std::vector<format::Format *> formats,
                                     utils::Parameters *params);
};

#endif

#ifdef USE_METIS

namespace metis {
//! Objectives to be optimized by METIS
typedef enum {
  METIS_OBJTYPE_CUT,
  METIS_OBJTYPE_VOL,
  METIS_OBJTYPE_NODE
} mobjtype_et;

//! Partitiong Methods
typedef enum { METIS_PTYPE_RB, METIS_PTYPE_KWAY } mptype_et;

//! Coarsening Schemes
typedef enum { METIS_CTYPE_RM, METIS_CTYPE_SHEM } mctype_et;

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
};  // namespace metis

//! Parameters for metis partitioning
/*!
 * This struct replaces the options array of METIS
 * The names of the options are identical to the array
 * and can be found here:
 * http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
 */
struct MetisPartitionParams : utils::Parameters {
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
 * Wraps the METIS partitioner available here:
 * https://github.com/KarypisLab/METIS The library must be compiled with the
 * USE_METIS option turned on and the pre-built METIS library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info. Detailed explanations of the options can be
 * found here: http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
 */
template <typename IDType, typename NNZType, typename ValueType>
class MetisPartition : public PartitionPreprocessType<IDType> {
 private:
  static IDType *PartitionCSR(std::vector<format::Format *> formats,
                              utils::Parameters *params);

 public:
  typedef MetisPartitionParams ParamsType;
  MetisPartition();
  MetisPartition(ParamsType params);
};

struct MetisReorderParams : utils::Parameters {
  int64_t ctype = metis::METIS_CTYPE_RM;
  int64_t rtype = metis::METIS_RTYPE_SEP2SIDED;
  int64_t nseps = 1;
  int64_t numbering = 0;
  int64_t niter = 10;
  int64_t seed = 42;
  int64_t no2hop = 0;
  int64_t compress = 0;
  int64_t ccorder = 0;
  int64_t pfactor = 0;
  int64_t ufactor = 30;
};

template <typename IDType, typename NNZType, typename ValueType>
class MetisReorder : public ReorderPreprocessType<IDType> {
 public:
  typedef MetisReorderParams ParamsType;
  MetisReorder();
  MetisReorder(ParamsType params);
  static IDType *GetReorderCSR(std::vector<format::Format *> formats,
                               utils::Parameters *);
};

#endif

#ifdef USE_PULP

//! Parameters for the PulpPartition class
struct PulpPartitionParams : utils::Parameters {
  double vert_balance = 1.1;
  double edge_balance = 1.5;
  bool do_lp_init = false;
  bool do_bfs_init = true;
  bool do_repart = false;
  bool do_edge_balance = false;
  bool do_maxcut_balance = false;
  bool verbose_output = false;
  int seed = 42;
  int num_partitions = 2;
};

//! A wrapper for the PULP graph partitioner
/* !
 * Wraps the PULP partitioner available here:
 * https://github.com/HPCGraphAnalysis/PuLP. The library must be compiled with the
 * USE_PULP option turned on and the pre-built PULP library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info.
 */
template <typename IDType, typename NNZType, typename ValueType>
class PulpPartition : public PartitionPreprocessType<IDType> {
 private:
  static IDType *PartitionCSR(std::vector<format::Format *> formats,
                              utils::Parameters *params);

 public:
  typedef PulpPartitionParams ParamsType;
  PulpPartition();
  PulpPartition(ParamsType params);
};
#endif

#ifdef USE_PATOH

namespace patoh {
enum Objective {
  CON = 1,
  CUT = 2
};

enum ParameterInit {
  DEFAULT = 0,
  SPEED = 1,
  QUALITY = 2
};

}

//! Parameters for the PulpPartition class
struct PatohPartitionParams : utils::Parameters {
  patoh::Objective objective = patoh::CON;
  patoh::ParameterInit param_init = patoh::DEFAULT;
  int num_partitions = 2;
  int final_imbalance = -1;
  int seed = 42;
};

//! A wrapper for the Patoh graph partitioner
/* !
 * Wraps the Patoh partitioner available here:
 * https://faculty.cc.gatech.edu/~umit/software.html.
 * The library must be compiled with the
 * USE_PATOH option turned on and the pre-built PATOH library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info.
 */
template <typename IDType, typename NNZType, typename ValueType>
class PatohPartition : public PartitionPreprocessType<IDType> {
 private:
  static IDType *PartitionCSR(std::vector<format::Format *> formats,
                              utils::Parameters *params);

 public:
  typedef PatohPartitionParams ParamsType;
  PatohPartition();
  PatohPartition(ParamsType params);
};
#endif

class GraphFeatureBase {
 public:
  //! Calculates the degree distribution of every vertex represented by the
  //! FormerOrderTwo object `format`.
  /*!
   * @tparam FeatureType data type used for storing degree distribution values.
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return an array of type `FeatureType*` size format->get_dimensions()[0]
   * with the degree distribution of each vertex.
   */
  template <typename FeatureType, typename AutoIDType, typename AutoNNZType,
            typename AutoValueType>
  static FeatureType *GetDegreeDistribution(
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input) {
    DegreeDistribution<AutoIDType, AutoNNZType, AutoValueType, FeatureType>
        deg_dist;
    return deg_dist.GetDistribution(format, contexts, convert_input);
  }

  //! Calculates the degree distribution of every vertex represented by the
  //! FormerOrderTwo object `format` with cached output.
  /*!
   * @tparam FeatureType data type used for storing degree distribution values.
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @return An std::pair with the second element being an array of type
   * `FeatureType*` size format->get_dimensions()[0] with the degree
   * distribution of each vertex, and the first being a vector of all the
   * formats generated by converting the input (if such conversions were needed
   * to execute the permutation).
   */
  template <typename FeatureType, typename AutoIDType, typename AutoNNZType,
            typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                                      AutoValueType> *>,
                   FeatureType *>
  GetDegreeDistributionCached(
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts) {
    DegreeDistribution<AutoIDType, AutoNNZType, AutoValueType, FeatureType>
        deg_dist;
    auto output = deg_dist.GetDistributionCached(format, contexts, true);
    std::vector<
        format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
        converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    return std::make_pair(converted_formats, std::get<1>(output));
  }
  //! Calculates the degree count of every vertex represented by the
  //! FormerOrderTwo object `format`.
  /*!
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return an array of size format->get_dimensions()[0] with the degree of
   * each vertex.
   */
  template <typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static AutoNNZType *GetDegrees(
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input) {
    Degrees<AutoIDType, AutoNNZType, AutoValueType> deg_dist;
    return deg_dist.GetDegrees(format, contexts, convert_input);
  }
  //! Calculates the degree count of every vertex represented by the
  //! FormerOrderTwo object `format` with cached output.
  /*!
   * @param format FormatOrderTwo object representing a graph.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return An std::pair with the second element being an array of type
   * `FeatureType*` size format->get_dimensions()[0] with the degree of each
   * vertex, and the first being a vector of all the formats generated by
   * converting the input (if such conversions were needed to execute the
   * permutation).
   */
  template <typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                                      AutoValueType> *>,
                   AutoNNZType *>
  GetDegreesCached(
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts) {
    Degrees<AutoIDType, AutoNNZType, AutoValueType> deg_dist;
    auto output = deg_dist.GetDegreesCached(format, contexts, true);
    std::vector<
        format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
        converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    return std::make_pair(converted_formats, std::get<1>(output));
  }
};

#ifdef USE_AMD_ORDER
#ifdef __cplusplus
extern "C" {
#endif
#include <amd.h>
#ifdef __cplusplus
}
#endif

//! Parameters for AMDReordering
/*!
 * For the exact definitions, please consult the documentation given with the 
 * code, which is available here:
 * https://dl.acm.org/doi/abs/10.1145/1024074.1024081
 */
struct AMDReorderParams : utils::Parameters {
  double dense = AMD_DEFAULT_DENSE;
  double aggressive = AMD_DEFAULT_AGGRESSIVE;
};

//! A wrapper for the AMD reordering algorithm
/*!
 * Wraps the AMD reordering algorithm library available here as supplemental material:
 * https://dl.acm.org/doi/abs/10.1145/1024074.1024081
 * The library must be compiled with the
 * USE_AMD_ORDER option turned on and the pre-built AMD library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info.
 */
template <typename IDType, typename NNZType, typename ValueType>
class AMDReorder : public ReorderPreprocessType<IDType> {
public:
  typedef AMDReorderParams ParamsType;
  AMDReorder(ParamsType);
  AMDReorder();
protected:
  static IDType* AMDReorderCSR(std::vector<format::Format*>, utils::Parameters*);
};
#endif

//! Parameters for Reorder Heatmap generator
struct ReorderHeatmapParams : utils::Parameters {
  //! Number of parts to split vertices over
  int num_parts = 3;
  ReorderHeatmapParams(int b) : num_parts(b){}
  ReorderHeatmapParams(){}
};

//! Calculates density of non-zeros of a 2D format on a num_parts * num_parts grid
/*!
 * Splits the input 2D matrix into a grid of size num_parts * num_parts containing an
 * equal number of rows and columns, and calculates the density of non-zeros in each
 * cell in the grid relative to the total number of non-zeros in the matrix, given that the 
 * matrix was reordered according to a permutation matrix. 
 * Returns the densities as a dense array (FormatOrderOne) of size num_parts * num_parts where
 * the density at cell [i][j] in the 2D grid is located at index [i*num_parts+j] in the 
 * grid. The density values sum up to 1.
 * @tparam FloatType type used to represent the densities of non-zeros.
 */
template <typename IDType, typename NNZType, typename ValueType, typename FloatType>
class ReorderHeatmap : public utils::FunctionMatcherMixin<format::FormatOrderOne<FloatType>*>{
public:
  ReorderHeatmap();
  ReorderHeatmap(ReorderHeatmapParams params);
  format::FormatOrderOne<FloatType>* Get(format::FormatOrderTwo<IDType, NNZType, ValueType> *format, format::FormatOrderOne<IDType>* permutation_r, format::FormatOrderOne<IDType>* permutation_c, std::vector<context::Context*> contexts, bool convert_input);
protected:
  static format::FormatOrderOne<FloatType>* ReorderHeatmapCSRArrayArray(std::vector<format::Format*> formats, utils::Parameters * poly_params);
};

//! A class containing the interface for reordering and permuting data.
/*!
 * The class contains all the functionalities needed for reordering. That
 * includes a function generate reordering permutations from data, functions to
 * permute data using a permutation vector, and a function to inverse the
 * permutation of data. In the upcoming release, ReorderBase will include
 * functions to extract feeatures from permutation vectors and permuted data.
 */
class ReorderBase {
 public:
  //! Generates a permutation array from a FormatOrderTwo object using the
  //! Reordering class `Reordering`.
  /*!
   *
   * @tparam Reordering a reordering class defining a reordering algorithm. For
   * a full list of available reordering algorithms, please check
   * [here](../pages/getting_started/available.html).
   * @param params a struct containing the parameters specific for the
   * reordering algorithm `Reordering`. Please check the documentation of each
   * reordering for the specifications of its parameters.
   * @param format FormatOrderTwo object to be used to generate permutation
   * array.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the permutation array.
   */
  template <template <typename, typename, typename> typename Reordering,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static AutoIDType *Reorder(
      typename Reordering<AutoIDType, AutoNNZType, AutoValueType>::ParamsType
          params,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input) {
    static_assert(
        std::is_base_of_v<ReorderPreprocessType<AutoIDType>,
                          Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
        "You must pass a reordering function (with base ReorderPreprocessType) "
        "to ReorderBase::Reorder");
    static_assert(
        !std::is_same_v<GenericReorder<AutoIDType, AutoNNZType, AutoValueType>,
                        Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
        "You must pass a reordering function (with base ReorderPreprocessType) "
        "to ReorderBase::Reorder");
    Reordering<AutoIDType, AutoNNZType, AutoValueType> reordering(params);
    return reordering.GetReorder(format, contexts, convert_input);
  }
  // TODO: add page for reordering
  //! Generates a permutation array from a FormatOrderTwo object using the
  //! Reordering class `Reordering` with cached output.
  /*!
   *
   * @tparam Reordering a reordering class defining a reordering algorithm. For
   * a full list of available reordering algorithms, please check: xxx
   * @param params a struct containing the parameters specific for the
   * reordering algorithm `Reordering`. Please check the documentation of each
   * reordering for the specifications of its parameters.
   * @param format FormatOrderTwo object to be used to generate permutation
   * array.
   * @param contexts vector of contexts that can be used for permutation.
   * @return An std::pair with the second element being the permutation array,
   * and the first being a vector of all the formats generated by converting the
   * input (if such conversions were needed to execute the permutation).
   */
  template <template <typename, typename, typename> typename Reordering,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                                      AutoValueType> *>,
                   AutoIDType *>
  ReorderCached(
      typename Reordering<AutoIDType, AutoNNZType, AutoValueType>::ParamsType
          params,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts) {
    static_assert(
        std::is_base_of_v<ReorderPreprocessType<AutoIDType>,
                          Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
        "You must pass a reordering function (with base ReorderPreprocessType) "
        "to ReorderBase::Reorder");
    static_assert(
        !std::is_same_v<GenericReorder<AutoIDType, AutoNNZType, AutoValueType>,
                        Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
        "You must pass a reordering function (with base ReorderPreprocessType) "
        "to ReorderBase::Reorder");
    Reordering<AutoIDType, AutoNNZType, AutoValueType> reordering(params);
    auto output = reordering.GetReorderCached(format, contexts, true);
    std::vector<
        format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
        converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    return std::make_pair(converted_formats, std::get<1>(output));
  }

  //! Permute a two-dimensional format row- and column-wise using a single
  //! permutation array for both axes.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
            typename ReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *Permute2D(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                      ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format row- and column-wise using a single
  //! permutation array for both axes with cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param ordering Permutation
   * array to use when permuting rows and columns. @param format object to be
   * permuted. @param contexts vector of contexts that can be used for
   * permutation. @param convert_output if true, the returned object will be
   * converted to `ReturnFormatType`. Otherwise, the returned object will be
   * cast to `ReturnFormatType`, and if the cast fails, an exception of type
   * `sparsebase::utils::TypeException`. @return An std::pair with the second
   * element being the permuted format, and the first being a vector of all the
   * formats generated by converting the input (if such conversions were needed
   * to execute the permutation). By default, the permuted object is returned as
   * a pointer at a generic FormatOrderTwo object. However, if the user passes a
   * concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the
   * returned format will be converted to that type. If not, the returned object
   * will only be cast to that type (if casting fails, an exception of type
   * utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
            typename ReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                                      AutoValueType> *>,
                   ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DCached(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 ordering);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<
        format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
        converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                      ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<ReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a two-dimensional format row- and column-wise using a permutation
  //! array for each axis with cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param row_ordering
   * Permutation array to use when permuting rows. @param col_ordering
   * Permutation array to use when permuting col. @param format object to be
   * permuted. @param contexts vector of contexts that can be used for
   * permutation. @param convert_output if true, the returned object will be
   * converted to `ReturnFormatType`. Otherwise, the returned object will be
   * cast to `ReturnFormatType`, and if the cast fails, an exception of type
   * `sparsebase::utils::TypeException`. @return An std::pair with the second
   * element being the permuted format, and the first being a vector of all the
   * formats generated by converting the input (if such conversions were needed
   * to execute the permutation). By default, the permuted object is returned as
   * a pointer at a generic FormatOrderTwo object. However, if the user passes a
   * concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the
   * returned format will be converted to that type. If not, the returned object
   * will only be cast to that type (if casting fails, an exception of type
   * utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
            typename ReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                                      AutoValueType> *>,
                   ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DRowColumnWiseCached(
      AutoIDType *row_ordering, AutoIDType *col_ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(row_ordering,
                                                                 col_ordering);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<
        format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
        converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                      ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<ReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a two-dimensional format row- and column-wise using a permutation
  //! array for each axis.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
            typename ReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *
  Permute2DRowColumnWise(
      AutoIDType *row_ordering, AutoIDType *col_ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(row_ordering,
                                                                 col_ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                      ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format row-wise using a permutation array.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
            typename ReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *
  Permute2DRowWise(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 nullptr);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                      ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format row-wise using a permutation array with
  //! cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param ordering Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_output if true,
   * the returned object will be converted to `ReturnFormatType`. Otherwise, the
   * returned object will be cast to `ReturnFormatType`, and if the cast fails,
   * an exception of type `sparsebase::utils::TypeException`. @return An
   * std::pair with the second element being the permuted format, and the first
   * being a vector of all the formats generated by converting the input (if
   * such conversions were needed to execute the permutation). By default, the
   * permuted object is returned as a pointer at a generic FormatOrderTwo
   * object. However, if the user passes a concrete FormatOrderTwo class as the
   * templated parameter `ReturnFormatType`, e.g. format::CSR, then if
   * `convert_output` is true, the returned format will be converted to that
   * type. If not, the returned object will only be cast to that type (if
   * casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
            typename RelativeReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<
      std::vector<
          format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>,
      RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DRowWiseCached(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 nullptr);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<
        format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
        converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                      RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<RelativeReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<RelativeReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a two-dimensional format column-wise using a permutation array.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
            typename ReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *
  Permute2DColWise(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(nullptr,
                                                                 ordering);
    auto out_format = perm.GetTransformation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                      ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format column-wise using a permutation array
  //! with cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param ordering Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_output if true,
   * the returned object will be converted to `ReturnFormatType`. Otherwise, the
   * returned object will be cast to `ReturnFormatType`, and if the cast fails,
   * an exception of type `sparsebase::utils::TypeException`. @return An
   * std::pair with the second element being the permuted format, and the first
   * being a vector of all the formats generated by converting the input (if
   * such conversions were needed to execute the permutation). By default, the
   * permuted object is returned as a pointer at a generic FormatOrderTwo
   * object. However, if the user passes a concrete FormatOrderTwo class as the
   * templated parameter `ReturnFormatType`, e.g. format::CSR, then if
   * `convert_output` is true, the returned format will be converted to that
   * type. If not, the returned object will only be cast to that type (if
   * casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
            typename ReturnFormatType = format::FormatOrderTwo,
            typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                                      AutoValueType> *>,
                   ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DColWiseCached(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_output = false) {
    PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(nullptr,
                                                                 ordering);
    auto output = perm.GetTransformationCached(format, contexts, true);
    std::vector<
        format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
        converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                      ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                      format::FormatOrderTwo<AutoIDType, AutoNNZType,
                                             AutoValueType>>)
      output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<ReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a one-dimensional format using a permutation array.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderOne. Defines the
   * return pointer type. Default is FormatOrderOne. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderOne object. However, if the user passes a concrete
   * FormatOrderOne class as the templated parameter `ReturnFormatType`, e.g.
   * format::Array, then if `convert_output` is true, the returned format will
   * be converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename>
            typename ReturnFormatType = format::FormatOrderOne,
            typename AutoIDType, typename AutoValueType>
  static ReturnFormatType<AutoValueType> *Permute1D(
      AutoIDType *ordering, format::FormatOrderOne<AutoValueType> *format,
      std::vector<context::Context *> context, bool convert_inputs,
      bool convert_output = false) {
    PermuteOrderOne<AutoIDType, AutoValueType> perm(ordering);
    auto out_format = perm.GetTransformation(format, context, convert_inputs);
    ReturnFormatType<AutoValueType> * output;
    if constexpr (std::is_same_v<ReturnFormatType<AutoValueType>,
                                 format::FormatOrderOne<AutoValueType>>)
      output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a one-dimensional format using a permutation array with cached
  //! output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderOne. Defines the
   * return pointer type. Default is FormatOrderOne. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_output if true,
   * the returned object will be converted to `ReturnFormatType`. Otherwise, the
   * returned object will be cast to `ReturnFormatType`, and if the cast fails,
   * an exception of type `sparsebase::utils::TypeException`. @return An
   * std::pair with the second element being the permuted format, and the first
   * being a vector of all the formats generated by converting the input (if
   * such conversions were needed to execute the permutation). By default, the
   * permuted object is returned as a pointer at a generic FormatOrderOne
   * object. However, if the user passes a FormatOrderOne class as the templated
   * parameter `ReturnFormatType`, e.g. format::Array, then if `convert_output`
   * is true, the returned format will be converted to that type. If not, the
   * returned object will only be cast to that type (if casting fails, an
   * exception of type utils::TypeException will be thrown).
   */
  template <template <typename>
            typename ReturnFormatType = format::FormatOrderOne,
            typename AutoIDType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderOne<AutoValueType> *>,
                   ReturnFormatType<AutoValueType> *>
  Permute1DCached(AutoIDType *ordering,
                  format::FormatOrderOne<AutoValueType> *format,
                  std::vector<context::Context *> context,
                  bool convert_output = false) {
    PermuteOrderOne<AutoIDType, AutoValueType> perm(ordering);
    auto output = perm.GetTransformationCached(format, context, true);
    std::vector<format::FormatOrderOne<AutoValueType> *> converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<format::FormatOrderOne<AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoValueType> * output_format;
    if constexpr (std::is_same_v<ReturnFormatType<AutoValueType>,
                                 format::FormatOrderOne<AutoValueType>>)
      output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format = std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format = std::get<1>(output)->template As<ReturnFormatType>();
            
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Takes a permutation array and its length and inverses it.
  /*!
   * Takes a permutation array and its length and inverses it. If a format `A`
   * was permuted with `perm` into object `B`, then permuting `B` with the
   * inverse permutation returns its order to `A`.
   * @param perm a permutation array of length `length`
   * @param length the length of the permutation array
   * @return a permutation array of length `length` that is the inverse of
   * `perm`, i.e. can be used to reverse a permutation done by `perm`.
   */
  template <typename AutoIDType, typename AutoNumType>
  static AutoIDType *InversePermutation(AutoIDType *perm, AutoNumType length) {
    static_assert(std::is_integral_v<AutoNumType>,
                  "Length of the permutation array must be an integer");
    auto inv_perm = new AutoIDType[length];
    for (AutoIDType i = 0; i < length; i++) {
      inv_perm[perm[i]] = i;
    }
    return inv_perm;
  }
  //! Calculates density of non-zeros of a 2D format on a num_parts * num_parts grid
  /*!
 * Splits the input 2D matrix into a grid of size num_parts * num_parts containing an
 * equal number of rows and columns, and calculates the density of non-zeros in each
 * cell in the grid relative to the total number of non-zeros in the matrix, given that the 
 * matrix was reordered according to a permutation matrix. 
 * Returns the densities as a dense array (FormatOrderOne) of size num_parts * num_parts where
 * the density at cell [i][j] in the 2D grid is located at index [i*num_parts+j] in the 
 * grid. The density values sum up to 1.
  * @tparam FloatType type used to represent the densities of non-zeros.
  * @param format the 2D matrix to calculate densities for.
  * @param permutation the permutation array containing the reordering of rows and columns.
  * @param num_parts number of parts to split rows/columns over
  * @param contexts vector of contexts that can be used for permutation. 
  * @param convert_input whether or not to convert the input format if that is needed.
  * @return a format::Array containing the densities of the cells in the num_parts * num_parts 2D grid.
  */
  template <typename FloatType,  typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static sparsebase::format::Array<FloatType>* Heatmap(format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> * format, format::FormatOrderOne<AutoIDType>* permutation_r, format::FormatOrderOne<AutoIDType>* permutation_c, int num_parts,
                  std::vector<context::Context *> contexts, bool convert_input) {
    ReorderHeatmap<AutoIDType, AutoNNZType, AutoValueType, FloatType> heatmapper(num_parts); 
    format::FormatOrderOne<FloatType>* arr = heatmapper.Get(format, permutation_r, permutation_c, contexts, convert_input);
    return arr->template Convert<sparsebase::format::Array>();
  }
};

enum BitMapSize{
  BitSize16 = 16,
  BitSize32 = 32/*,
  BitSize64 = 64*/ //at the moment, using 64 bits is not working as intended
};
//! Params struct for GrayReorder
struct GrayReorderParams : utils::Parameters {
  BitMapSize resolution;
  int nnz_threshold;
  int sparse_density_group_size;
  explicit GrayReorderParams() {}
  GrayReorderParams(BitMapSize r, int nnz_thresh, int group_size)
      : resolution(r),
        nnz_threshold(nnz_thresh),
        sparse_density_group_size(group_size) {}
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
                                   utils::Parameters *poly_params);
};



}  // namespace sparsebase::preprocess
#ifdef _HEADER_ONLY
#include "sparsebase/preprocess/preprocess.cc"
#endif

#ifdef USE_CUDA
#include "cuda/preprocess.cuh"
#ifdef _HEADER_ONLY
#include "cuda/preprocess.cu"
#endif
#endif

#endif  // SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
