#include <memory>

#include "sparsebase/utils/extractable.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_FEATURE_PREPROCESS_TYPE_H
#define SPARSEBASE_PROJECT_FEATURE_PREPROCESS_TYPE_H
namespace sparsebase::feature {
//! A class that does feature extraction.
/*!
 * An Extractable class that has a function matching
 * capability. In other words, an Extractable to which implementation functions
 * can be added and used. @tparam FeatureType the return type of feature
 * extraction
 */
template <typename FeatureType>
class FeaturePreprocessType
    : public utils::FunctionMatcherMixin<FeatureType, utils::Extractable> {
 public:
  std::shared_ptr<utils::Parameters> get_params() override;
  std::shared_ptr<utils::Parameters> get_params(std::type_index) override;
  void set_params(std::type_index, std::shared_ptr<utils::Parameters>) override;
  std::type_index get_id() override;
  ~FeaturePreprocessType();
};

}  // namespace sparsebase::feature

#ifdef _HEADER_ONLY
#include "sparsebase/feature/feature_preprocess_type.cc"
#endif

#endif  // SPARSEBASE_PROJECT_FEATURE_PREPROCESS_TYPE_H
