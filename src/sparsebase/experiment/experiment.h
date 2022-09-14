
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
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace sparsebase::experiment {

struct KernelParams {};
using KernelFunction = std::any (std::vector<format::Format *> formats,
                                std::vector<context::Context *> contexts,
                                          KernelParams * params);

class ExperimentType{
    public:
        virtual void Run() = 0;
        virtual void AddFormat(format::Format *) = 0;
        virtual void AddContext(context::Context *) = 0;
        virtual void AddProcess(preprocess::ExtractableType *) = 0;
        virtual void AddKernel(KernelFunction) = 0;
        virtual std::vector<double> GetRunTimes() = 0;
};

struct ExperimentParams{
    unsigned int num_runs;
};
class ConcreteExperiment : public ExperimentType {
    public:
        void Run();
        void AddFormat(format::Format * format);
        void AddProcess(preprocess::ExtractableType * process);
        void AddContext(context::Context * context);
        void AddKernel(KernelFunction);
        std::vector<double> GetRunTimes();
    private:
        ExperimentParams _params;
        std::vector<KernelFunction *> _kernels;
        std::vector<format::Format *> _formats;
        std::vector<preprocess::ExtractableType*> _processes;
        std::vector<context::Context*> _contexts;
        std::vector<double> _runs;
};

//class PreprocessExperiment : public ConcreteExperiment {
    //public:
        //void Run()

//}; 


};

#endif