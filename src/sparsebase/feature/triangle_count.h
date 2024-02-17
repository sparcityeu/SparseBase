#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_TRIANGLE_COUNT_H
#define SPARSEBASE_PROJECT_TRIANGLE_COUNT_H
namespace sparsebase::feature {

struct TriangleCountParams : utils::Parameters {
  bool countDirected = 0;
  TriangleCountParams() {}
  TriangleCountParams(bool countDirected) : countDirected(countDirected){}
};

//! Find the triangle count in the graph representation of a format object
template <typename IDType, typename NNZType, typename ValueType>
class TriangleCount : public feature::FeaturePreprocessType<int64_t *> {
 public:
  //! An empty struct used for the parameters of Triangle Count
  typedef TriangleCountParams ParamsType;
  TriangleCount();
  TriangleCount(ParamsType);
  TriangleCount(const TriangleCount<IDType, NNZType, ValueType> &d);
  TriangleCount(std::shared_ptr<ParamsType>);
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<utils::Extractable *> get_subs() override;
  static std::type_index get_id_static();

  //! Triangle Count executor function that carries out function matching
  /*!
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting features.
   * @param convert_input whether or not to convert the input format if that is needed.
   * @return triangle count in the graph representation of `format`
   */
  int64_t *GetTriangleCount(format::Format *format, std::vector<context::Context *> contexts,
                     bool convert_input);

  std::tuple<std::vector<std::vector<format::Format *>>, int64_t *>
  //! Triangle Count executor function that carries out function atching with cached output
  /*!
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting features.
   * @param convert_input whether or not to convert the input format if that is needed.
   * @return triangle count in the graph representation of `format`
   */
  GetTriangleCountCached(format::Format *format,std::vector<context::Context *> contexts,
                   bool convert_input);
  //! Triangle Count implementation function for CSRs
  /*!
   * @param formats A vector containing a single format pointer that should point at a CSR object
   * @param params a Parameters pointer, though it is not used in the function
   * @return triangle count in the graph representations of 'format[0]'
   */
  static int64_t *GetTriangleCountCSR(std::vector<format::Format *> formats,
                               utils::Parameters *params);
  ~TriangleCount();

 protected:
  void Register();
};

}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/triangle_count.cc"
#endif
#endif  // SPARSEBASE_PROJECT_TRIANGLE_COUNT_H
