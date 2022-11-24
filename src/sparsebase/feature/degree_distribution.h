#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/object/object.h"

#ifndef SPARSEBASE_PROJECT_DEGREE_DISTRIBUTION_H
#define SPARSEBASE_PROJECT_DEGREE_DISTRIBUTION_H
namespace sparsebase::feature {

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
}
#ifdef _HEADER_ONLY
#include "sparsebase/feature/degree_distribution.cc"
#endif
#endif  // SPARSEBASE_PROJECT_DEGREE_DISTRIBUTION_H
