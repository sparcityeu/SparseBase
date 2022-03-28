#ifndef _Feature_HPP
#define _Feature_HPP

#include "sparse_format.h"
#include "sparsebase/sparse_preprocess.h"
#include <unordered_map>
#include <tuple>
#include <vector>
#include <set>
#include <any>

namespace sparsebase::feature {

template<typename Interface>
struct Implementation
{
public:
  Implementation() = default;
  template<typename ConcreteType>
  explicit Implementation(ConcreteType&& object)
      : storage{std::forward<ConcreteType>(object)}
        , getter{ [](std::any &storage) -> Interface& { return std::any_cast<ConcreteType&>(storage); } }
  {}
  Implementation(const Implementation& object)
      : storage{object.storage}
      , getter{ object.getter }
  {}
  Implementation(Implementation&& object)
      : storage{std::move(object.storage)}
      , getter{ std::move(object.getter) }
  {}
  Implementation& operator=(Implementation other)
  {
    storage = other.storage;
    getter = other.getter;
    return *this;
  }

  Interface *operator->() { return &getter(storage); }

private:
  std::any storage;
  Interface& (*getter)(std::any&);
};

using Feature = Implementation<preprocess::ExtractableType>;

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
  std::tuple<ClassType, std::vector<std::type_index>> MatchClass(std::unordered_map<std::type_index, ClassType> & source, std::vector<std::type_index> & ordered, unsigned int K);
  void GetClassesHelper(std::unordered_map<std::type_index, ClassType> & source, std::vector<std::type_index> & ordered, std::vector<ClassType> & res);
  std::vector<ClassType> GetClasses(std::unordered_map<std::type_index, ClassType> & source);
};


template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Extractor: public ClassMatcherMixin<preprocess::ExtractableType*> {
public:
  Extractor();
  ~Extractor();
  std::unordered_map<std::type_index, std::any> Extract(std::vector<Feature> & features,
                                     format::Format * format);
  std::unordered_map<std::type_index, std::any> Extract(format::Format * format);
  void Add(Feature f);
  void Substract(Feature f);
  std::vector<std::type_index> GetList();
  void PrintFuncList();
private:
  std::unordered_map<std::type_index, preprocess::ExtractableType*> in_;
};

} // namespace sparsebase::feature

#ifdef _HEADER_ONLY
#include "sparse_feature.cc"
#endif

#endif