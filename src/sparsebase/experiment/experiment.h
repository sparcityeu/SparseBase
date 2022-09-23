
/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_EXPERIMENT_EXPERIMENT_H_
#define SPARSEBASE_SPARSEBASE_EXPERIMENT_EXPERIMENT_H_

#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
#include <any>
#include <unordered_map>
#include <vector>
#include <chrono>

namespace sparsebase::experiment {

using LoadDataFunction = std::unordered_map<std::string, format::Format*> (std::string);
using PreprocessFunction = std::unordered_map<std::string, format::Format*> (std::unordered_map<std::string, format::Format*>);
using KernelFunction = std::any (std::unordered_map<std::string, format::Format*>);

class ExperimentType{
    public:
        virtual void Run(unsigned int times = 1) = 0;
        virtual void AddDataLoader(LoadDataFunction, std::vector<std::string> targets) = 0;
        virtual void AddPreprocess(PreprocessFunction) = 0;
        virtual void AddKernel(KernelFunction) = 0;
        virtual std::vector<double> GetRunTimes() = 0;
        virtual std::vector<std::any> GetResults() = 0;
};

struct ExperimentParams{
    unsigned int num_runs;
};

class ConcreteExperiment : public ExperimentType {
    public:
        void Run(unsigned int times = 1);
        virtual void AddDataLoader(LoadDataFunction, std::vector<std::string> targets);
        virtual void AddPreprocess(PreprocessFunction);
        virtual void AddKernel(KernelFunction);
        std::vector<double> GetRunTimes();
        std::vector<std::any> GetResults();
    private:
        ExperimentParams _params;
        std::vector<std::vector<std::string>> _targets;
        std::vector<std::function<KernelFunction>> _kernels;
        std::vector<std::function<LoadDataFunction>> _dataLoaders;
        std::vector<std::function<PreprocessFunction>> _preprocesses;
        std::vector<double> _runtimes;
        std::vector<std::any> _results;
};

//class PreprocessExperiment : public ConcreteExperiment {
    //public:
        //void Run()

//}; 


};

#endif