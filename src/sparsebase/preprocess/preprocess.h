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
#include "sparsebase/utils/extractable.h"
#include "sparsebase/feature/feature_preprocess_type.h"

#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif

namespace sparsebase::preprocess {

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
class DegreeDistribution : public feature::FeaturePreprocessType<FeatureType *> {
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
  virtual std::vector<utils::Extractable *> get_subs();
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
class Degrees : public feature::FeaturePreprocessType<IDType *> {
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
  std::vector<utils::Extractable *> get_subs() override;
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
    : public feature::FeaturePreprocessType<
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
  std::vector<utils::Extractable *> get_subs() override;
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
