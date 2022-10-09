
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
#include <map>
#include <vector>
#include <chrono>

namespace sparsebase::experiment {

using LoadDataFunction = std::unordered_map<std::string, format::Format*> (std::string &);
using PreprocessFunction = void (std::unordered_map<std::string, format::Format*> &);
using KernelFunction = std::any (std::unordered_map<std::string, format::Format*> &, std::any, std::any);

class ExperimentType{
    public:
        virtual void Run(unsigned int times, bool store_auxiliary) = 0;
        virtual void AddDataLoader(LoadDataFunction, std::vector<std::pair<std::string, std::any>> targets) = 0;
        virtual void AddPreprocess(std::string, PreprocessFunction) = 0;
        virtual void AddKernel(std::string, KernelFunction, std::any) = 0;
        virtual std::map<std::string, std::vector<double>> GetRunTimes() = 0;
        virtual std::map<std::string, std::vector<std::any>> GetResults() = 0;
        virtual std::map<std::string, std::any> GetAuxiliary() = 0;
};

class ConcreteExperiment : public ExperimentType {
    public:
        void Run(unsigned int times = 1, bool store_auxiliary = false) override;
        void AddDataLoader(LoadDataFunction, std::vector<std::pair<std::string, std::any>> targets) override;
        void AddPreprocess(std::string, PreprocessFunction) override;
        void AddKernel(std::string, KernelFunction, std::any) override;
        std::map<std::string, std::vector<double>> GetRunTimes() override;
        std::map<std::string, std::vector<std::any>> GetResults() override;
        std::map<std::string, std::any> GetAuxiliary() override;
    private:
        std::vector<std::vector<std::pair<std::string, std::any>>> _targets;
        std::unordered_map<std::string, std::function<KernelFunction>> _kernels;
        std::unordered_map<std::string, std::any> _kernel_parameters;
        std::vector<std::function<LoadDataFunction>> _dataLoaders;
        std::unordered_map<std::string, std::function<PreprocessFunction>> _preprocesses;
        std::map<std::string, std::any> _auxiliary;
        std::map<std::string, std::vector<double>> _runtimes;
        std::map<std::string, std::vector<std::any>> _results;
};

template< template<typename, typename, typename> typename FormatType, template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadFormat(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::FormatOrderTwo<IDType, NNZType, VALType> * coo = reader.ReadCOO();
  FormatType<IDType, NNZType, VALType> * format = coo->template Convert<FormatType>();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", format);
  return r;
}

template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadCSR(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::CSR<IDType, NNZType, VALType> * csr = reader.ReadCSR();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csr);
  return r;
}

template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadCOO(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::COO<IDType, NNZType, VALType> * csr = reader.ReadCOO();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csr);
  return r;
}

template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadCSC(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::COO<IDType, NNZType, VALType> * coo = reader.ReadCOO();
  format::CSC<IDType, NNZType, VALType> * csc = coo->template Convert<format::CSC>();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csc);
  return r;
}

template< template<typename, typename, typename> typename ReorderType, typename ContextType, typename IDType, typename NNZType, typename VALType>
void ReorderCSR(std::unordered_map<std::string, format::Format*> & data) {
  ContextType context;
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>({}, data["format"]->As<format::CSR<IDType, NNZType, VALType>>(), {&context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<format::CSR>(perm, data["format"]->As<format::CSR<IDType, NNZType, VALType>>(), {&context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  data.emplace("processed_format", A_csc);
}

template<typename IDType, typename NNZType, typename VALType>
void Pass(std::unordered_map<std::string, format::Format*> & data) {
  data["processed_format"] = data["format"];
}

template< template<typename, typename, typename> typename ReorderType, template<typename, typename, typename> typename FormatType, typename ContextType, typename IDType, typename NNZType, typename VALType>
void Reorder(std::unordered_map<std::string, format::Format*> & data) {
  ContextType context;
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>({}, data["format"]->As<FormatType<IDType, NNZType, VALType>>(), {&context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<FormatType>(perm, data["format"]->As<FormatType<IDType, NNZType, VALType>>(), {&context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  data.emplace("processed_format", A_csc);
}


} // sparsebase::experiment

#endif