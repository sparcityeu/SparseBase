#include "experiment.h"

namespace sparsebase::experiment {

void ConcreteExperiment::AddDataLoader(LoadDataFunction func, std::vector<std::string> targets){
  this->_targets.push_back(targets);
  this->_dataLoaders.emplace_back(func);
}

void ConcreteExperiment::AddKernel(KernelFunction func, std::any params){
  this->_kernels.emplace_back(func);
  this->_kernel_parameters.emplace_back(params);
}

void ConcreteExperiment::AddPreprocess(PreprocessFunction func){
  this->_preprocesses.emplace_back(func);
}

std::vector<double> ConcreteExperiment::GetRunTimes(){
  return this->_runtimes;
}

std::vector<std::any> ConcreteExperiment::GetResults(){
  return this->_results;
}

void ConcreteExperiment::Run(unsigned int times) {
  for(unsigned int i = 0; i < times; i++){
    for(unsigned int l = 0; l < _dataLoaders.size(); l++){
      auto loader = _dataLoaders[l];
      for(auto t: _targets[l]){
        std::cout << "Load data " << std::endl;
        auto data = loader(t);
        for(auto p: _preprocesses){
          std::cout << "Preprocess " << std::endl;
          p(data);
          for(unsigned int ki = 0; ki < _kernels.size(); ki++){
            auto k = this->_kernels[ki];
            auto params = this->_kernel_parameters[ki];
            std::cout << "Kernel " << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = k(data, params);
            auto end = std::chrono::high_resolution_clock::now();
            this->_results.push_back(res);
            std::chrono::duration<double> secs = end - start;
            this->_runtimes.push_back(secs.count());
          }
        }
      }
    }
  }
}
}