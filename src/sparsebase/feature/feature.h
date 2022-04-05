#ifndef SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_
#define SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_

#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
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
 noexcept       : storage{std::move(object.storage)}
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

class Extractor: public ClassMatcherMixin<preprocess::ExtractableType*> {
public:
  ~Extractor();
  static std::unordered_map<std::type_index, std::any> Extract(std::vector<Feature> & features,
                                                        format::Format * format, const std::vector<context::Context*> &);
  std::unordered_map<std::type_index, std::any> Extract(format::Format * format, const std::vector<context::Context*> &);
  void Add(Feature f);
  void Subtract(Feature f);
  std::vector<std::type_index> GetList();
  void PrintFuncList();
protected:
  Extractor() noexcept = default;
private:
  std::unordered_map<std::type_index, preprocess::ExtractableType*> in_;
};

template<typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureExtractor: public Extractor {
public:
  FeatureExtractor();
};

} // namespace sparsebase::format

#ifdef _HEADER_ONLY
#include "sparsebase/feature/feature.cc"
#endif

#endif // SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_