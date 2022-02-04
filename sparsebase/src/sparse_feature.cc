#include "sparsebase/sparse_feature.h"
#include <set>

using namespace std;

namespace sparsebase {

size_t FeatureHash::operator()(Feature f) const { return f; }


#ifdef NDEBUG
#include "init/sparse_feature.inc"
#else
template class FeatureValue<unsigned int>;
template class FeatureFunctor<unsigned int, unsigned int, float>;
template class BasicFeatureFunctor<unsigned int, unsigned int, float>;
template class SparseFeature<unsigned int, unsigned int, float>;
#endif
} // namespace sparsebase
