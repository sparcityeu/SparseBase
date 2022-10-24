#include "experiment.h"

namespace sparsebase::experiment {

void ConcreteExperiment::AddDataLoader(LoadDataFunction func, std::vector<std::pair<std::vector<std::string>, std::any>> targets){
  this->targets_.push_back(targets);
  this->dataLoaders_.emplace_back(func);
}

void ConcreteExperiment::AddKernel(std::string id, KernelFunction func, std::any params){
  this->kernels_.insert(std::make_pair(id, std::make_pair(func, params)));
}

void ConcreteExperiment::AddPreprocess(std::string id, PreprocessFunction func, std::any params){
  this->preprocesses_.insert(std::make_pair(id, std::make_pair(func, params)));
}

std::map<std::string, std::vector<double>> ConcreteExperiment::GetRunTimes(){
  return this->runtimes_;
}

std::map<std::string, std::vector<std::any>> ConcreteExperiment::GetResults(){
  return this->results_;
}

std::map<std::string, std::any> ConcreteExperiment::GetAuxiliary(){
  return this->auxiliary_;
}

void ConcreteExperiment::Run(unsigned int times, bool store_auxiliary) {
  for(unsigned int l = 0; l < this->dataLoaders_.size(); l++){
    auto loader = this->dataLoaders_[l];
    for(auto & [file_names, file_param] : this->targets_[l]){
      //auto file_name = t.first;
      //auto file_param = t.second;
      auto data = loader(file_names);
      std::string file_id;
      for(const auto & file_name: file_names){
        file_id.append("-");
        file_id.append(file_name);
      }
      if(store_auxiliary){
        for(const auto & d: data){
          auto aid = d.first;
          aid.append(",");
          aid.append(file_id);
          this->auxiliary_[aid] = d.second;
        }
      }
      for( const auto & [pid, ppair]: this->preprocesses_){
        auto pfunc = ppair.first;
        auto pparams = ppair.second;
        pfunc(data, file_param, pparams);
        for(const auto & [kid, kpair]: this->kernels_){
          auto kfunc = kpair.first;
          auto kparams = kpair.second;
          for(unsigned int i = 0; i < times; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto res = kfunc(data, file_param, pparams, kparams);
            auto end = std::chrono::high_resolution_clock::now();
            auto id = file_id;
            id.append(",");
            id.append(pid);
            id.append(",");
            id.append(kid);
            id.append(",");
            id.append(std::to_string(i));
            this->results_[id].push_back(res);
            std::chrono::duration<double> secs = end - start;
            this->runtimes_[id].push_back(secs.count());
          }
        }
        if(store_auxiliary){
          for(const auto & d: data){
            auto aid = d.first;
            aid.append(",");
            aid.append(file_id);
            if(this->auxiliary_.find(aid) == this->auxiliary_.end()){
              aid.append(",");
              aid.append(pid);
              this->auxiliary_[aid] = d.second;
            }
          }
        }
      }
    }
  }
}

} // sparsebase::experiment