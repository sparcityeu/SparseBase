
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

namespace sparsebase::experiment {

// variadics or params struct
using GetDataFunction = std::unordered_map<std::string, format::Format*> (std::string);
// variadics or params struct
using PreprocessFunction = std::unordered_map<std::string, format::Format*> (std::unordered_map<std::string, format::Format*>);
// variadics or params struct
using KernelFunction = std::any (std::unordered_map<std::string, format::Format*>);

class ExperimentType{
    public:
        virtual void Run(unsigned int times = 1) = 0;
        virtual void AddDataGetter(GetDataFunction) = 0;
        virtual void AddPreprocess(PreprocessFunction) = 0;
        virtual void AddKernel(KernelFunction) = 0;
};

struct ExperimentParams{
    unsigned int num_runs;
};
class ConcreteExperiment : public ExperimentType {
    public:
        void Run(unsigned int times = 1);
        virtual void AddDataGetter(GetDataFunction);
        virtual void AddPreprocess(PreprocessFunction);
        virtual void AddKernel(KernelFunction);
        std::vector<double> GetRunTimes();
    private:
        ExperimentParams _params;
        std::vector<std::function<KernelFunction>> _kernels;
        std::vector<std::function<GetDataFunction>> _dataGetters;
        std::vector<std::function<PreprocessFunction>> _preprocesses;
        std::vector<double> _runTimes;
};

//class PreprocessExperiment : public ConcreteExperiment {
    //public:
        //void Run()

//}; 


};

#endif