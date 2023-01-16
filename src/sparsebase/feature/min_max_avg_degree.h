#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_MIN_MAX_AVG_DEGREE_H
#define SPARSEBASE_PROJECT_MIN_MAX_AVG_DEGREE_H
namespace sparsebase::feature {
//! An empty struct used for the parameters of MinMaxAvgDegree
struct Params : utils::Parameters {};
//! Find the min, max and avg degree in the graph representation of a format
//! object
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class MinMaxAvgDegree
    : public feature::FeaturePreprocessType<
          std::unordered_map<std::type_index, std::any>> {
  //! An empty struct used for the parameters of MinMaxAvgDegree
  typedef Params ParamsType;

 public:
  MinMaxAvgDegree();
  MinMaxAvgDegree(Params);
  MinMaxAvgDegree(
      const MinMaxAvgDegree<IDType, NNZType, ValueType, FeatureType>
          &d);
  MinMaxAvgDegree(std::shared_ptr<Params>);
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<utils::Extractable *> get_subs() override;
  static std::type_index get_id_static();

  //! Min, max, avg degree generation executor function that carries
  //! out function matching
  /*!
   *
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * features.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return a map with two (type_index, any) pairs. Two of them are
   * the min and max degree of type NNZType*, and average degree of type
   * FeatureType*.
   */
  std::unordered_map<std::type_index, std::any> Get(
      format::Format *format, std::vector<context::Context *> contexts,
      bool convert_input);

  //! Min, max, avg degree implementation function for CSRs
  /*!
   *
   * @param format a single format pointer to any format
   * @param params a utils::Parameters pointer, though it
   * is not used in the function features.
   * @return a map with two (type_index, any) pairs. Two of them are
   * the min and max degree of type NNZType*, and average degree of type
   * FeatureType*.
   */
  static std::unordered_map<std::type_index, std::any> GetCSR(
      std::vector<format::Format *> formats, utils::Parameters *params);
  ~MinMaxAvgDegree();

 protected:
  void Register();
};

}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/min_max_avg_degree.cc"
#endif

#endif  // SPARSEBASE_PROJECT_MIN_MAX_AVG_DEGREE_H
