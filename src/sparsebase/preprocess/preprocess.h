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

#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif

namespace sparsebase::preprocess {

//! A class that does feature extraction.
/*!
 * An Extractable class that has a function matching
 * capability. In other words, an Extractable to which implementation functions
 * can be added and used. @tparam FeatureType the return type of feature
 * extraction
 */
template <typename FeatureType>
class FeaturePreprocessType
    : public utils::FunctionMatcherMixin<FeatureType,
                                  utils::Extractable> {
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


#ifdef USE_METIS


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
