#include "experiment.h"

namespace sparsebase::experiment {

void ConcreteExperiment::AddDataLoader(LoadDataFunction func, std::vector<std::string> targets){
  this->_targets.push_back(targets);
  this->_dataLoaders.push_back(func);
}

void ConcreteExperiment::AddKernel(KernelFunction func){
  this->_kernels.push_back(func);
}

void ConcreteExperiment::AddPreprocess(PreprocessFunction func){
  this->_preprocesses.push_back(func);
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
          data = p(data);
          for(auto k: _kernels){
            std::cout << "Kernel " << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto res = k(data);
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