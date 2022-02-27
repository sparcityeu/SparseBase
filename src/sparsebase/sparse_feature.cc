#include "sparsebase/sparse_feature.h"
#include "sparsebase/sparse_preprocess.h"
#include <vector>

namespace sparsebase::feature {

template<typename ClassType, typename Key, typename KeyHash, typename KeyEqualTo>
void ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::
    RegisterClass(const std::vector<std::type_index> instants, ClassType val){
    this->map_[instants] = val;
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
Extractor<IDType, NNZType, ValueType, FeatureType>::Extractor(){
   std::vector<std::type_index> temp;
   auto lol = new preprocess::DegreeDistribution<IDType, NNZType, ValueType, FeatureType>();
   temp.push_back(std::type_index(typeid(lol)));
   this->RegisterClass(temp, lol);
}

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
std::vector<std::any> Extractor<IDType, NNZType, ValueType, FeatureType>::Extract(std::vector<Feature<FeatureType>> & fs,
                                                                                   format::Format * format) {
    // match and get classes for feature extraction
    std::vector<std::any> res;
    for(auto & f : fs){
      res.push_back(f->Extract(format));
    }
    return res;
}

#if !defined(_HEADER_ONLY)
#include "init/feature.inc"
#endif

} // namespace sparsebase