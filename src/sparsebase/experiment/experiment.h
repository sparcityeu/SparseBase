
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
        virtual void Run(unsigned int times) = 0;
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
        void Run(unsigned int times = 1) override;
        void AddDataLoader(LoadDataFunction, std::vector<std::string> targets) override;
        void AddPreprocess(PreprocessFunction) override;
        void AddKernel(KernelFunction) override;
        std::vector<double> GetRunTimes() override;
        std::vector<std::any> GetResults() override;
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

template< template<typename, typename, typename> typename FormatType, template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadCSR(std::string file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  FormatType<IDType, NNZType, VALType> * csr = reader.ReadCSR();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("graph", csr);
  auto format = csr->get_format_id();
  auto dimensions = csr->get_dimensions();
  //auto row_ptr2 = csr->get_row_ptr();
  //auto col2 = csr->get_col();
  //auto vals = csr->get_vals();
  std::cout << "Format: " << format.name() << std::endl;
  std::cout << "# of dimensions: " << dimensions.size() << std::endl;
  for (int i = 0; i < dimensions.size(); i++) {
    std::cout << "Dim " << i << " size " << dimensions[i] << std::endl;
  }
  return r;
};

template< template<typename, typename, typename> typename ReorderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> ReorderCSR(std::unordered_map<std::string, format::Format*> data) {
  std::cout << "PREPROCESS LAMBDA FUNCTION!!" << std::endl;
  std::unordered_map<std::string, format::Format*> r;
  context::CPUContext cpu_context;
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>({}, data["graph"]->As<format::CSR<IDType, NNZType, VALType>>(), {&cpu_context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<format::CSR>(perm, data["graph"]->As<format::CSR<IDType, NNZType, VALType>>(), {&cpu_context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  r.emplace("ordered", A_csc);
  return r;
};

};

#endif