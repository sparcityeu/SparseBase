#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_MIN_DEGREE_COLUMN_H
#define SPARSEBASE_PROJECT_MIN_DEGREE_COLUMN_H

namespace sparsebase::feature {
//! Find the min degree in the graph representation of a format object
template <typename IDType, typename NNZType, typename ValueType>
class MinDegreeColumn : public feature::FeaturePreprocessType<NNZType *> {
 public:
  //! An empty struct used for the parameters of Min Degree
  typedef utils::Parameters ParamsType;
  MinDegreeColumn();
  MinDegreeColumn(ParamsType);
  MinDegreeColumn(const MinDegreeColumn<IDType, NNZType, ValueType> &d);
  MinDegreeColumn(std::shared_ptr<ParamsType>);
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<utils::Extractable *> get_subs() override;
  static std::type_index get_id_static();

  //! Min Degree executor function that carries out function matching
  /*!
   *
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * features.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return min degree in the graph representation of `format`
   */
  NNZType *GetMinDegreeColumn(format::Format *format,
                     std::vector<context::Context *> contexts,
                     bool convert_input);
  std::
      tuple<std::vector<std::vector<format::Format *>>, NNZType *>
      //! Min Degree executor function that carries out function
      //! matching with cached output
      /*!
       *
       * @param format a single format pointer to any format
       * @param contexts vector of contexts that can be used for extracting
       * features.
       * @param convert_input whether or not to convert the input format if that
       * is needed.
       * @return min degree in the graph representation of `format`
       */
      GetMinDegreeColumnCached(format::Format *format,
                       std::vector<context::Context *> contexts,
                       bool convert_input);
  //! Min Degree implementation function for CSCs
  /*!
   *
   * @param formats A vector containing a single format pointer that should
   * point at a CSR object
   * @param params a Parameters pointer, though it
   * is not used in the function
   * @return min degree in the graph representations of 'format[0]'
   */
  static NNZType *GetMinDegreeColumnCSC(std::vector<format::Format *> formats,
                               utils::Parameters *params);
  ~MinDegreeColumn();

 protected:
  void Register();
};

}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/min_degree_column.cc"
#endif

#endif  // SPARSEBASE_PROJECT_MIN_DEGREE_COLUMN_H
