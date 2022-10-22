#include "experiment.h"

namespace sparsebase::experiment {

void ConcreteExperiment::AddDataLoader(LoadDataFunction func, std::vector<std::pair<std::vector<std::string>, std::any>> targets){
  this->_targets.push_back(targets);
  this->_dataLoaders.emplace_back(func);
}

void ConcreteExperiment::AddKernel(std::string id, KernelFunction func, std::any params){
  this->_kernels.insert(std::make_pair(id, std::make_pair(func, params)));
  //this->_kernel_parameters.insert(std::make_pair(id, params));
}

void ConcreteExperiment::AddPreprocess(std::string id, PreprocessFunction func, std::any params){
  this->_preprocesses.insert(std::make_pair(id, std::make_pair(func, params)));
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
    for(auto & [file_names, file_param] : this->_targets[l]){
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
          this->_auxiliary[aid] = d.second;
        }
      }
      for( const auto & [pid, ppair]: this->_preprocesses){
        auto pfunc = ppair.first;
        auto pparams = ppair.second;
        pfunc(data, pparams);
        for(const auto & [kid, kpair]: this->_kernels){
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
            this->_results[id].push_back(res);
            std::chrono::duration<double> secs = end - start;
            this->_runtimes[id].push_back(secs.count());
          }
        }
        if(store_auxiliary){
          for(const auto & d: data){
            auto aid = d.first;
            aid.append(",");
            aid.append(file_id);
            if(this->_auxiliary.find(aid) == this->_auxiliary.end()){
              aid.append(",");
              aid.append(pid);
              this->_auxiliary[aid] = d.second;
            }
          }
        }
      }
    }
  }
}

} // sparsebase::experiment