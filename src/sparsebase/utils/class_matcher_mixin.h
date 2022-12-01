#ifndef SPARSEBASE_PROJECT_CLASS_MATCHER_MIXIN_H
#define SPARSEBASE_PROJECT_CLASS_MATCHER_MIXIN_H

#include <typeindex>
#include <vector>

#include "utils.h"

namespace sparsebase::utils {

template <class ClassType, typename Key = std::vector<std::type_index>,
          typename KeyHash = utils::TypeIndexVectorHash,
          typename KeyEqualTo = std::equal_to<std::vector<std::type_index>>>
class ClassMatcherMixin {
#ifdef DEBUG
 public:
#else
 protected:
#endif

  std::unordered_map<Key, ClassType, KeyHash, KeyEqualTo> map_;
  void RegisterClass(std::vector<std::type_index> instants, ClassType);
  std::tuple<ClassType, std::vector<std::type_index>> MatchClass(
      std::unordered_map<std::type_index, ClassType> &source,
      std::vector<std::type_index> &ordered, unsigned int K);
  void GetClassesHelper(std::unordered_map<std::type_index, ClassType> &source,
                        std::vector<std::type_index> &ordered,
                        std::vector<ClassType> &res);
  std::vector<ClassType> GetClasses(
      std::unordered_map<std::type_index, ClassType> &source);
};
template <typename ClassType, typename Key, typename KeyHash,
          typename KeyEqualTo>
void ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::RegisterClass(
    const std::vector<std::type_index> instants, ClassType val) {
  this->map_.insert({instants, val});
}

template <typename ClassType, typename Key, typename KeyHash,
          typename KeyEqualTo>
std::tuple<ClassType, std::vector<std::type_index>>
ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::MatchClass(
    std::unordered_map<std::type_index, ClassType> &source,
    std::vector<std::type_index> &ordered, unsigned int K) {
  unsigned int N = source.size();
  std::string bitmask(K, 1);  // K leading 1's
  bitmask.resize(N, 0);       // N-K trailing 0's
  do {
    std::vector<std::type_index> temp;
    for (unsigned int i = 0; i < N; ++i)  // [0..N-1] integers
    {
      // check if comb exists
      if (bitmask[i]) {
        temp.push_back(ordered[i]);
      }
    }
    if (map_.find(temp) != map_.end()) {  // match found
      auto &merged = map_[temp];
      for (auto el : temp) {  // set params for the merged class
        auto tr = source[el];
        auto par = tr->get_params();
        merged->set_params(el, par);
      }
      std::vector<std::type_index> rem;
      for (unsigned int i = 0; i < N; ++i)  // return remaining
      {
        if (!bitmask[i]) {
          rem.push_back(ordered[i]);
        }
      }
      return std::make_tuple(merged, rem);
    }
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  return std::make_tuple(nullptr, ordered);
}

template <typename ClassType, typename Key, typename KeyHash,
          typename KeyEqualTo>
void ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::GetClassesHelper(
    std::unordered_map<std::type_index, ClassType> &source,
    std::vector<std::type_index> &ordered, std::vector<ClassType> &res) {
  if (ordered.empty()) {
    return;
  }
  bool found = false;
  // std::cout << "remaining size: " << ordered.size() << std::endl;
  // for (auto &el : ordered) {
  //   std::cout << el.name() << std::endl;
  // }
  // std::cout << std::endl;

  for (unsigned int c = source.size(); !found && c > 0; c--) {
    auto r = MatchClass(source, ordered, c);
    // std::cout << source.size() << " " << c << " " << std::get<0>(r)
    //           << std::endl;
    if (std::get<0>(r)) {
      res.push_back(std::get<0>(r));
      ordered = std::get<1>(r);
      found = true;
    }
  }

  // std::cout << "result size: " << res.size() << std::endl;
  //  for(auto &el : res){
  //    std::cout << el->get_id().name() << std::endl;
  //  }
  // std::cout << std::endl;
  GetClassesHelper(source, ordered, res);
}

template <typename ClassType, typename Key, typename KeyHash,
          typename KeyEqualTo>
std::vector<ClassType>
ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::GetClasses(
    std::unordered_map<std::type_index, ClassType> &source) {
  std::vector<ClassType> res;
  std::vector<std::type_index> ordered;
  for (auto &el : source) {
    ordered.push_back(std::get<0>(el));
  }
  std::sort(ordered.begin(), ordered.end());
  // std::cout << "Here: " << ordered.size() << std::endl;
  GetClassesHelper(source, ordered, res);
  return res;
}
}  // namespace sparsebase::utils
#endif  // SPARSEBASE_PROJECT_CLASS_MATCHER_MIXIN_H
