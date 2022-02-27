#ifndef _Feature_HPP
#define _Feature_HPP

#include "sparse_format.h"
#include "sparsebase/sparse_preprocess.h"
#include <unordered_map>
#include <tuple>
#include <vector>
#include <any>

namespace sparsebase::feature {

template<typename Interface>
struct Implementation
{
public:
  template<typename ConcreteType>
  Implementation(ConcreteType&& object)
      : storage{std::forward<ConcreteType>(object)}
        , getter{ [](std::any &storage) -> Interface& { return std::any_cast<ConcreteType&>(storage); } }
  {}

  Interface *operator->() { return &getter(storage); }

private:
  std::any storage;
  Interface& (*getter)(std::any&);
};

template<typename FeatureType>
using Feature = Implementation<preprocess::FType>;
//using FeatureType = Implementation<std::common_type>;

template<class ClassType,
          typename Key = std::vector<std::type_index>,
          typename KeyHash = preprocess::TypeIndexVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class ClassMatcherMixin {

protected:
  std::unordered_map<Key, 
                     ClassType,
                     KeyHash,
                     KeyEqualTo> map_;
  void RegisterClass(std::vector<std::type_index> instants, ClassType);

};


template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Extractor: public ClassMatcherMixin<preprocess::FType*> {
public:
  Extractor();
  std::vector<std::any> Extract(std::vector<Feature<FeatureType>> & features,
                                     format::Format * format);
};

} // namespace sparsebase::feature

#ifdef _HEADER_ONLY
#include "sparse_feature.cc"
#endif

#endif