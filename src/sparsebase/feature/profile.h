#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_PROFILE_H
#define SPARSEBASE_PROJECT_PROFILE_H
namespace sparsebase::feature {

//! Compute profile in the graph representation of a format object
template <typename IDType, typename NNZType, typename ValueType>
class Profile : public feature::FeaturePreprocessType<IDType *> {
 public:
  //! An empty struct used for the parameters of Profile Feature
  typedef utils::Parameters ParamsType;
  Profile();
  Profile(ParamsType);
  Profile(const Profile<IDType, NNZType, ValueType> &d);
  Profile(std::shared_ptr<ParamsType>);
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<utils::Extractable *> get_subs() override;
  static std::type_index get_id_static();

  //! Profile executor function that carries out function matching
  /*!
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting features.
   * @param convert_input whether or not to convert the input format if that is needed.
   * @return Profile in the graph representation of `format`
   */
  IDType *GetProfile(format::Format *format, std::vector<context::Context *> contexts,
                     bool convert_input);

  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  //! Profile executor function that carries out function matching with cached output
  /*!
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting features.
   * @param convert_input whether or not to convert the input format if that is needed.
   * @return Profile in the graph representation of `format`
   */
  GetProfileCached(format::Format *format, std::vector<context::Context *> contexts,
                       bool convert_input);
  //! Profile implementation function for CSRs
  /*!
   * @param formats A vector containing a single format pointer that should point at a CSR object
   * @param params a Parameters pointer, though it is not used in the function
   * @return Profile in the graph representations of 'format[0]'
   */
  static IDType *GetProfileCSR(std::vector<format::Format *> formats, utils::Parameters *params);

  ~Profile();

 protected:
  void Register();
};

}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/profile.cc"
#endif
#endif  // SPARSEBASE_PROJECT_PROFILE_H
