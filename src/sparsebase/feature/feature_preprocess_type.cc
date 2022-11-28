#include "sparsebase/feature/feature_preprocess_type.h"

#include <any>
#include <memory>
#include <unordered_map>

namespace sparsebase::feature {

template <typename FeatureType>
FeaturePreprocessType<FeatureType>::~FeaturePreprocessType() = default;

template <typename FeatureType>
std::shared_ptr<utils::Parameters>
FeaturePreprocessType<FeatureType>::get_params() {
  return this->params_;
}
template <typename FeatureType>
std::shared_ptr<utils::Parameters>
FeaturePreprocessType<FeatureType>::get_params(std::type_index t) {
  if (this->pmap_.find(t) != this->pmap_.end()) {
    return this->pmap_[t];
  } else {
    throw utils::FeatureParamsException(get_id().name(), t.name());
  }
}
template <typename FeatureType>
void FeaturePreprocessType<FeatureType>::set_params(
    std::type_index t, std::shared_ptr<utils::Parameters> p) {
  auto ids = this->get_sub_ids();
  if (std::find(ids.begin(), ids.end(), t) != ids.end()) {
    this->pmap_[t] = p;
  } else {
    throw utils::FeatureParamsException(get_id().name(), t.name());
  }
}
template <typename FeatureType>
std::type_index FeaturePreprocessType<FeatureType>::get_id() {
  return typeid(*this);
}

#if !defined(_HEADER_ONLY)
#include "init/feature_preprocess_type.inc"
template class FeaturePreprocessType<
    std::unordered_map<std::type_index, std::any>>;
#endif

}  // namespace sparsebase::feature