#ifndef _SPARSEFEATURE_HPP
#define _SPARSEFEATURE_HPP

#include "sparse_format.h"
#include <unordered_map>
#include <tuple>
#include <vector>
#include <any>

namespace sparsebase {

//! Enum keeping sparce features
enum Feature {
  sfNRNZ = 0,
};

struct FeatureHash {
  size_t operator()(Feature f) const;
};

template <typename IDType>
class FeatureValue {
  std::any* feature_;
  std::vector<IDType> dimension_;
public:
  FeatureValue() {
    Init();
  }
  virtual ~FeatureValue() {
    delete[] feature_;
  }
  virtual void Init() {
    IDType size = 1;
    if (dimension_.size() > 0) {
      for (auto dimensionIt = dimension_.begin(); dimensionIt != dimension_.end(); dimensionIt ++) {
        size *= *dimensionIt;
      }
    }
    feature_ = new std::any[size];
  }
  virtual unsigned int GetOrder() {
    return dimension_.size();
  }
  virtual std::vector<IDType> GetDimension() {
    return dimension_;
  }
  virtual std::any* Value() {
    return feature_;
  }
};


template <typename IDType, typename NNZType, typename ValueType>
class FeatureFunctor {
  std::unordered_map<
      Feature,
      FeatureValue<IDType> *,
      FeatureHash>
      features_;
  std::unordered_map<
      Feature,
      std::string,
      FeatureHash>
      feature_names_;
public:
  FeatureFunctor() {
  }
  virtual ~FeatureFunctor() {
  }
  virtual void Extract(format::Format *source) {
  }
  virtual FeatureValue<IDType>* GetFeature(Feature feature) {
    if (features_.find(feature) != features_.end()) {
      return features_[feature];
    } else {
      return nullptr;
    }
  }
  virtual std::string GetFeatureName(Feature feature) {
    if (feature_names_.find(feature) != feature_names_.end()) {
      return feature_names_[feature];
    } else {
      return std::string();
    }
  }
  virtual std::vector<Feature> ListFeatures() {
    std::vector<Feature> keys;
    keys.reserve(features_.size());
    for(auto f : features_) {
      keys.push_back(f.first);
    }
    return keys;
  }
  virtual void RegisterFeature(Feature feature, FeatureValue<IDType>* feature_value, std::string feature_name = std::string()) {
    if (features_.count(feature) == 0) {
      features_.emplace(feature, feature_value);
      feature_names_.emplace(feature, feature_name);
    }
  }
};

template <typename IDType, typename NNZType, typename ValueType>
class BasicFeatureFunctor : public FeatureFunctor<IDType, NNZType, ValueType> {
public:
  BasicFeatureFunctor() {
    this->RegisterFeature(sfNRNZ, new FeatureValue<IDType>(), "Number of real non-zero elements");
  }
  virtual ~BasicFeatureFunctor() {
  }
  virtual void Extract(format::Format *source) {
    *(this->GetFeature(sfNRNZ)->Value()) = 0.0;
  }
};

template <typename IDType, typename NNZType, typename ValueType>
class SparseFeature {
private:
  std::unordered_map<
      Feature,
      FeatureFunctor<IDType, NNZType, ValueType> *,
      FeatureHash>
      feature_map_;
  std::vector<FeatureFunctor<IDType, NNZType, ValueType> *> extractors_;

public:
  SparseFeature() {
    RegisterFeatureExtractionFunction(new BasicFeatureFunctor<IDType, NNZType, ValueType>());
  }
  ~SparseFeature() {
  }
  void RegisterFeatureExtractionFunction(
      FeatureFunctor<IDType, NNZType, ValueType> *feature_extractor) {
    if (!feature_extractor) {
      return;
    }
    std::vector<Feature> extractor_features = feature_extractor->ListFeatures();
    bool feature_added = false;
    for(auto feature : extractor_features) {
      if (feature_map_.count(feature) == 0) {
        feature_map_.emplace(feature, feature_extractor);
        feature_added = true;
      }
    }
    if ((!feature_added) && std::find(extractors_.begin(), extractors_.end(), feature_extractor) == extractors_.end()) {
      extractors_.push_back(feature_extractor);
    }
  }
  void Extract(format::Format *source) {
    for(auto extractor : extractors_) {
      extractor->Extract(source);
    }
  }
  FeatureValue<IDType> * GetFeature(Feature feature) {
    if (feature_map_.find(feature) != feature_map_.end()) {
      return feature_map_[feature]->GetFeature(feature);
    } else {
      return nullptr;
    }
  }
  std::string GetFeatureName(Feature feature) {
    if (feature_map_.find(feature) != feature_map_.end()) {
      return feature_map_[feature]->GetFeatureName(feature);
    } else {
      return std::string();
    }
  }
  std::vector<Feature> ListFeatures() {
    std::vector<Feature> keys;
    keys.reserve(feature_map_.size());
    for(auto f : feature_map_) {
      keys.push_back(f.first);
    }
    return keys;
  }
};

} // namespace sparsebase

#endif