#include "experiment.h"

namespace sparsebase::experiment {

void ConcreteExperiment::AddFormat(format::Format * format){
    this->_formats.push_back(format);
}

void ConcreteExperiment::AddKernel(KernelFunction * k){
    this->_kernels.push_back(k);
}

void ConcreteExperiment::AddProcess(preprocess::ExtractableType * process){
    this->_processes.push_back(process);
}

void ConcreteExperiment::AddContext(context::Context * context){
    this->_contexts.push_back(context);
}

std::vector<double> ConcreteExperiment::GetRunTimes(){
    return this->_runs;
}

void ConcreteExperiment::Run(){
    for(unsigned int i = 0; i < _params.num_runs; i++){
        for(auto f: _formats){
            for(auto p: _processes){
                // preprocess
                // transform
                for(auto k: _kernels){
                    // launch sampler
                    // run kernel
                    // collect metrics
                }
            }
        }
    }
}

}