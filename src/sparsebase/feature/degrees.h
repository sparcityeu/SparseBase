#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_DEGREES_H
#define SPARSEBASE_PROJECT_DEGREES_H
namespace sparsebase::feature {
//! Count the degrees of every vertex in the graph representation of a format
//! object
template <typename IDType, typename NNZType, typename ValueType>
class Degrees : public feature::FeaturePreprocessType<IDType *> {
 public:
  //! An empty struct used for the parameters of Degrees
  typedef utils::Parameters ParamsType;
  Degrees();
  Degrees(ParamsType);
  Degrees(const Degrees<IDType, NNZType, ValueType> &d);
  Degrees(std::shared_ptr<ParamsType>);
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

}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/degrees.cc"
#endif
#endif  // SPARSEBASE_PROJECT_DEGREES_H
