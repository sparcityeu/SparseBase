#include "feature_extractor.h"

#include <vector>

#include "sparsebase/feature/degree_distribution.h"
#include "sparsebase/feature/degrees.h"
#include "sparsebase/feature/degrees_degree_distribution.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/extractable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
FeatureExtractor<IDType, NNZType, ValueType, FeatureType>::FeatureExtractor() {
  auto degree_distribution =
      new feature::DegreeDistribution<IDType, NNZType, ValueType,
                                      FeatureType>();
  this->RegisterClass(degree_distribution->get_sub_ids(), degree_distribution);
  auto degrees = new Degrees<IDType, NNZType, ValueType>();
  this->RegisterClass(degrees->get_sub_ids(), degrees);

  auto degrees_degreedistribution =
      new feature::Degrees_DegreeDistribution<IDType, NNZType, ValueType,
                                              FeatureType>();
  this->RegisterClass(degrees_degreedistribution->get_sub_ids(),
                      degrees_degreedistribution);
}

#if !defined(_HEADER_ONLY)
#include "init/feature_extractor.inc"
#endif

}  // namespace sparsebase::feature