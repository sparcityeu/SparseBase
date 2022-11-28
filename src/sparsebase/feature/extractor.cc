#include "sparsebase/feature/extractor.h"

#include <vector>

#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/extractable.h"

namespace sparsebase::feature {

void Extractor::PrintFuncList() {
  std::cout << std::endl;
  std::cout << "Registered functions: " << std::endl;
  for (auto &cls : map_) {
    for (auto el : cls.first) {
      std::cout << el.name() << " ";
    }
    std::cout << "-> " << cls.second->get_id().name() << std::endl;
  }
  std::cout << std::endl;
}

std::vector<utils::Extractable *> Extractor::GetFuncList() {
  std::vector<utils::Extractable *> res;
  for (auto &cls : map_) {
    res.push_back(cls.second);
  }
  return res;
}

std::unordered_map<std::type_index, std::any> Extractor::Extract(
    std::vector<Feature> &fs, format::Format *format,
    const std::vector<context::Context *> &c, bool convert_input) {
  std::unordered_map<std::type_index, std::any> res;
  for (auto &el : fs) {
    auto t = el->Extract(format, c, convert_input);
    res.merge(t);
  }
  return res;
}

std::unordered_map<std::type_index, std::any> Extractor::Extract(
    format::Format *format, const std::vector<context::Context *> &c,
    bool convert_input) {
  if (in_.empty()) return {};
  // match and get classes for format extraction
  std::vector<utils::Extractable *> cs = this->GetClasses(in_);
  std::unordered_map<std::type_index, std::any> res;
  // std::cout << std::endl << "Classes used:" << std::endl;
  for (auto &el : cs) {
    // std::cout << el->get_id().name() << std::endl;
    res.merge(el->Extract(format, c, convert_input));
  }
  // std::cout << std::endl;
  return res;
}

void Extractor::Add(Feature f) {
  if (map_.find(f->get_sub_ids()) !=
      map_.end()) {  // check if the class is registered
    for (auto &cls : f->get_subs()) {
      auto id = cls->get_id();
      if (in_.find(id) == in_.end()) {
        // in_[id] = cls;
        in_.insert({id, cls});
      }
    }
  } else {
    throw utils::FeatureException(f->get_id().name(), typeid(this).name());
  }
}

void Extractor::Subtract(Feature f) {
  for (auto id : f->get_sub_ids()) {
    if (in_.find(id) != in_.end()) {
      delete in_[id];
      in_.erase(id);
    }
  }
}

std::vector<std::type_index> Extractor::GetList() {
  std::vector<std::type_index> res;
  for (auto &el : in_) {
    res.push_back(std::get<0>(el));
  }
  return res;
}

Extractor::~Extractor() {
  for (auto &el : in_) {
    delete el.second;
  }
}

}  // namespace sparsebase::feature
