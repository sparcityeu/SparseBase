#include "experiment.h"

namespace sparsebase::experiment {

void ConcreteExperiment::AddDataLoader(LoadDataFunction func, std::vector<std::pair<std::string, std::any>> targets){
  this->_targets.push_back(targets);
  this->_dataLoaders.emplace_back(func);
}

void ConcreteExperiment::AddKernel(std::string id, KernelFunction func, std::any params){
  this->_kernels.insert(std::make_pair(id, func));
  this->_kernel_parameters.insert(std::make_pair(id, params));
}

void ConcreteExperiment::AddPreprocess(std::string id, PreprocessFunction func){
  this->_preprocesses.insert(std::make_pair(id, func));
}

std::map<std::string, std::vector<double>> ConcreteExperiment::GetRunTimes(){
  return this->_runtimes;
}

std::map<std::string, std::vector<std::any>> ConcreteExperiment::GetResults(){
  return this->_results;
}

std::map<std::string, std::any> ConcreteExperiment::GetAuxiliary(){
  return this->_auxiliary;
}

void ConcreteExperiment::Run(unsigned int times, bool store_auxiliary) {
  for(unsigned int l = 0; l < this->_dataLoaders.size(); l++){
    auto loader = this->_dataLoaders[l];
    for(const auto & t: this->_targets[l]){
      auto file_name = t.first;
      auto file_param = t.second;
      auto data = loader(file_name);
      if(store_auxiliary){
        for(const auto & d: data){
          auto aid = d.first;
          aid.append(",");
          aid.append(file_name);
          this->_auxiliary[aid] = d.second;
        }
      }
      for(const auto & p: this->_preprocesses){
        auto pid = p.first;
        auto pfunc = p.second;
        pfunc(data);
        if(store_auxiliary){
          for(const auto & d: data){
            auto aid = d.first;
            aid.append(",");
            aid.append(file_name);
            if(this->_auxiliary.find(aid) == this->_auxiliary.end()){
              aid.append(",");
              aid.append(pid);
              this->_auxiliary[aid] = d.second;
            }
          }
        }
        for(const auto & k: this->_kernels){
          auto kid = k.first;
          auto kfunc = k.second;
          auto kparams = this->_kernel_parameters[kid];
          for(unsigned int i = 0; i < times; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto res = kfunc(data, file_param, kparams);
            auto end = std::chrono::high_resolution_clock::now();
            auto id = file_name;
            id.append(",");
            id.append(pid);
            id.append(",");
            id.append(kid);
            this->_results[id].push_back(res);
            std::chrono::duration<double> secs = end - start;
            this->_runtimes[id].push_back(secs.count());
          }
        }
      }
    }
  }
}

} // sparsebase::experiment