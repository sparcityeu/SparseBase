#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/object/object.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_MEDIAN_DEGREE_COLUMN_H
#define SPARSEBASE_PROJECT_MEDIAN_DEGREE_COLUMN_H

namespace sparsebase::feature {

//! An empty struct used for the parameters of MedianDegree
struct MedianDegreeColumnParams : utils::Parameters {};
//! Find the median degree of the graph representation of a format object
/*!
 *
 * @tparam FeatureType the type in which the distribution value are returned --
 * should be a floating type
 */
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class MedianDegreeColumn
    : public feature::FeaturePreprocessType<FeatureType *> {
 public:
  //! An empty struct used for the parameters of MedianDegree
  typedef MedianDegreeColumnParams ParamsType;
  MedianDegreeColumn();
  MedianDegreeColumn(MedianDegreeColumnParams);
  MedianDegreeColumn(const MedianDegreeColumn &);
  MedianDegreeColumn(std::shared_ptr<MedianDegreeColumnParams>);
  virtual std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input);
  virtual std::vector<std::type_index> get_sub_ids();
  virtual std::vector<utils::Extractable *> get_subs();
  static std::type_index get_id_static();

  //! Median degree generation executor function that carries out function
  //! matching
  /*!
   *
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return median degree in the graph representation of `format`
   */
  FeatureType *GetMedianDegreeColumn(format::Format *format,
                               std::vector<context::Context *> contexts,
                               bool convert_input);
  //! Median degree generation executor function that carries out function
  //! matching on a Graph
  /*!
   *
   * @param object a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return median degree in the graph
   */
  FeatureType *GetMedianDegreeColumn(
      object::Graph<IDType, NNZType, ValueType> *object,
      std::vector<context::Context *> contexts, bool convert_input);
  //! Median degree generation executor function that carries out function
  //! matching with cached outputs
  /*!
   * Generates the median degree of the passed format. If the input format
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
   * element is a pointer pointing to median degree
   */
  std::tuple<std::vector<std::vector<format::Format *>>, FeatureType *>
  GetMedianDegreeColumnCached(format::Format *format,
                        std::vector<context::Context *> contexts,
                        bool convert_input);

  static FeatureType
      *
      //! Median degree generation implementation function for CSRs
      /*!
       *
       * @param format a single format pointer to any format
       * @return median degree in the graph representation of `formats[0]`
       */
      GetMedianDegreeColumnCSC(std::vector<format::Format *> formats,
                         utils::Parameters *params);
  ~MedianDegreeColumn();

 protected:
  void Register();
};
}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/median_degree_column.cc"
#endif

#endif  // SPARSEBASE_PROJECT_MEDIAN_DEGREE_COLUMN_H
