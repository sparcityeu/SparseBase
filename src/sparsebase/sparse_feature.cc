#include "sparsebase/sparse_feature.h"
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_exception.h"
#include <vector>

namespace sparsebase::feature {

template<typename ClassType, typename Key, typename KeyHash, typename KeyEqualTo>
void ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::
    RegisterClass(const std::vector<std::type_index> instants, ClassType val){
    this->map_.insert({instants, val});
}

template<typename ClassType, typename Key, typename KeyHash, typename KeyEqualTo>
std::tuple<ClassType, std::vector<std::type_index>> ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::
    MatchClass(std::unordered_map<std::type_index, ClassType> & source, std::vector<std::type_index> & ordered, unsigned int K){
  unsigned int N = source.size();
  std::string bitmask(K, 1); // K leading 1's
  bitmask.resize(N, 0); // N-K trailing 0's
  do {
    std::vector<std::type_index> temp;
    for (unsigned int i = 0; i < N; ++i) // [0..N-1] integers
    {
      //check if comb exists
      if (bitmask[i]) {
        temp.push_back(ordered[i]);
      }
    }
    if(map_.find(temp) != map_.end()){ //match found
      auto & merged = map_[temp];
      for(auto el : temp){ //set params for the merged class
        auto tr = source[el];
        auto par = tr->get_params();
        merged->set_params(el, par);
      }
      std::vector<std::type_index> rem;
      for (unsigned int i = 0; i < N; ++i) // return remaining
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

template<typename ClassType, typename Key, typename KeyHash, typename KeyEqualTo>
void ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::
    GetClassesHelper(std::unordered_map<std::type_index, ClassType> & source, std::vector<std::type_index> & ordered, std::vector<ClassType> & res){
    if(ordered.empty()){
      return;
    }
    bool found = false;
    for(unsigned int c = source.size(); !found && c > 0; c--){
      auto r = MatchClass(source, ordered, c);
      if(std::get<0>(r)){
        res.push_back(std::get<0>(r));
        ordered = std::get<1>(r);
        found = true;
      }
    }
    GetClassesHelper(source, ordered, res);
}

template<typename ClassType, typename Key, typename KeyHash, typename KeyEqualTo>
std::vector<ClassType> ClassMatcherMixin<ClassType, Key, KeyHash, KeyEqualTo>::
    GetClasses(std::unordered_map<std::type_index, ClassType> & source){
  std::vector<ClassType> res;
  std::vector<std::type_index> ordered;
  for(auto & el: source){
    ordered.push_back(std::get<0>(el));
  }
  std::sort(ordered.begin(), ordered.end());
  GetClassesHelper(source, ordered, res);
  return res;
}

void Extractor::PrintFuncList() {
  std::cout<< std::endl;
  std::cout << "Registered functions: " << std::endl;
  for(auto & cls : map_){
    for(auto el : cls.first){
      std::cout << el.name() << " ";
    }
    std::cout << "-> " << cls.second->get_feature_id().name() << std::endl;
  }
  std::cout<< std::endl;
}

std::unordered_map<std::type_index, std::any>Extractor::Extract(std::vector<Feature> & fs,
                                                                                   format::Format * format, const std::vector<context::Context*> & c) {
    std::unordered_map<std::type_index, std::any> res;
    for(auto & el : fs){
      auto t = el->Extract(format, c);
      res.merge(t);
    }
    return res;
}

std::unordered_map<std::type_index, std::any>Extractor::Extract(format::Format * format, const std::vector<context::Context*> & c){
  if(in_.empty())
    return {};
  // match and get classes for feature extraction
  std::vector<preprocess::ExtractableType*> cs = this->GetClasses(in_);
  std::unordered_map<std::type_index, std::any> res;
  std::cout << std::endl << "Classes used:" << std::endl;
  for(auto & el : cs){
    std::cout << el->get_feature_id().name() << std::endl;
    res.merge(el->Extract(format, c));
  }
  std::cout << std::endl;
  return res;
}

void Extractor::Add(Feature f){
  if(map_.find(f->get_sub_ids()) != map_.end()){ //check if the class is registered
    for(auto & cls : f->get_subs()){
      auto id = cls->get_feature_id();
      if(in_.find(id) == in_.end()){
        //in_[id] = cls;
        in_.insert({id, cls});
      }
    }
  }
  else{
    throw utils::FeatureException(f->get_feature_id().name(), typeid(this).name());
  }
}

void Extractor::Subtract(Feature f){
  for(auto id : f->get_sub_ids()){
    if(in_.find(id) != in_.end()){
      delete in_[id];
      in_.erase(id);
    }
  }
}

std::vector<std::type_index> Extractor::GetList(){
  std::vector<std::type_index> res;
  for(auto & el : in_){
    res.push_back(std::get<0>(el));
  }
  return res;
}

Extractor::~Extractor() {
  for(auto & el: in_){
    delete el.second;
  }
}

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
FeatureExtractor<IDType, NNZType, ValueType, FeatureType>::FeatureExtractor() {
  auto degree_distribution = new preprocess::DegreeDistribution<IDType, NNZType, ValueType, FeatureType>();
  this->RegisterClass(degree_distribution->get_sub_ids(), degree_distribution);
  auto degrees = new preprocess::Degrees<IDType, NNZType, ValueType>();
  this->RegisterClass(degrees->get_sub_ids(), degrees);

  auto degrees_degreedistribution = new preprocess::Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>();
  this->RegisterClass(degrees_degreedistribution->get_sub_ids(), degrees_degreedistribution);
}

#if !defined(_HEADER_ONLY)
#include "init/feature.inc"
#endif

} // namespace sparsebase