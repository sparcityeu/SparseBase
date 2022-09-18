#include "experiment.h"

namespace sparsebase::experiment {

void ConcreteExperiment::AddDataGetter(GetDataFunction func){
    this->_dataGetters.push_back(func);
}

void ConcreteExperiment::AddKernel(KernelFunction func){
    this->_kernels.push_back(func);
}

void ConcreteExperiment::AddPreprocess(PreprocessFunction func){
    this->_preprocesses.push_back(func);
}

std::vector<double> ConcreteExperiment::GetRunTimes(){
    return this->_runTimes;
}

void ConcreteExperiment::Run(unsigned int times) {
    for(unsigned int i = 0; i < times; i++){
        for(auto f: _dataGetters){
            std::cout << "Get data " << std::endl;
            // get data and cache
            //auto data = f();
            for(auto p: _preprocesses){
              std::cout << "Preprocess " << std::endl;
              // preprocess
                // transform
                for(auto k: _kernels){
                  std::cout << "Kernel " << std::endl;
                  // launch sampler
                    // run kernel
                    // restore runtimes
                }
            }
        }
    }
}

}