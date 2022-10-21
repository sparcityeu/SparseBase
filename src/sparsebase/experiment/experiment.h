
/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at
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

//! Function template for loading data to the experiment.
//! All data that is going to be used in the experiment must be provided via a function that follows this template.
/*!
 *
 * \param file_paths File paths that store the target data.
 * \return data extracted from the files.
 */
using LoadDataFunction = std::function<std::unordered_map<std::string, format::Format*> (std::vector<std::string> & file_paths)>;
//! Function template for preprocess.
//! All preprocessing methods that are going to be applied on the data must follow this function definition.
/*!
 *
 * \param data Contains the data needed for the preprocessing. This map is returned by the DataLoader
 * \param params Preprocessing parameters.
 */
using PreprocessFunction = std::function<void (std::unordered_map<std::string, format::Format*> & data, std::any params)>;
//! Function template for kernel.
//! All kernel methods that are going to be run must follow this function definition.
/*!
 *
 * \param data Contains the data needed for the preprocessing. This map is returned by the DataLoader and extended by the preprocessing.
 * \param fparams Parameters specific to the target file.
 * \param kparams Parameters specific to the kernel.
 * \return The results of the kernel. The results are stored in _results.
 */
using KernelFunction = std::function<std::any (std::unordered_map<std::string, format::Format*> & data, std::any fparams, std::any kparams)>;

//! Abstract class that defines a common interface for experiments
class ExperimentType{
    public:
        //! Start the experiment.
        //! Each target will be preprocessed by all the preprocessing functions. Then every kernel will be applied to all of the newly created data.
        //! The cartesian product of the number of _targets, _preprocesses and _kernels iterations will be carried out.
        /*!
         *
         * \param times Specifies the number of samples.
         * \param store_auxiliary determines if the auxiliary data created as a by product of
         * the experiments is going to be stored or not.
         */
        virtual void Run(unsigned int times, bool store_auxiliary) = 0;
        //! Adds a dataLoader to the experiment.
        //! Each target data needs to be loaded to experiment via a LoadDataFunction.
        /*!
         *
         * \param func The function which follows the LoadDataFunction definition.
         * \param targets File paths and file specific parameters.
         */
        virtual void AddDataLoader(LoadDataFunction func, std::vector<std::pair<std::vector<std::string>, std::any>> targets) = 0;
        //! Adds a preprocessing to the experiment.
        //! Each preprocessing operation is added to experiment via a PreprocessFunction.
        /*!
         *
         * \param id Preprocessing ID is used to store results, runtimes and auxiliary data.
         * \param func The function which follows the PreprocessFunction definition.
         * \param params Parameters to be passed to the respective function.
         */
        virtual void AddPreprocess(std::string id, PreprocessFunction func, std::any params) = 0;
        //! Adds a kernel to the experiment.
        //! Each kernel is added to experiment via a KernelFunction.
        /*!
         *
         * \param id Kernel ID is used to store results, runtimes and auxiliary data.
         * \param func The function which follows the KernelFunction definition.
         */
        virtual void AddKernel(std::string id, KernelFunction func, std::any params) = 0;
        //!
        /*!
         *
         * \return Returns the runtimes of each kernel call.
         */
        virtual std::map<std::string, std::vector<double>> GetRunTimes() = 0;
        //!
        /*!
         *
         * \return Returns the results of each kernel call.
         */
        virtual std::map<std::string, std::vector<std::any>> GetResults() = 0;
        //!
        /*!
         *
         * \return Returns the auxiliary data stored during the experiment.
         */
        virtual std::map<std::string, std::any> GetAuxiliary() = 0;
};

class ConcreteExperiment : public ExperimentType {
    public:
        //! Start the experiment.
        //! Each target will be preprocessed by all the preprocessing functions. Then every kernel will be applied to all of the newly created data.
        //! The cartesian product of the number of _targets, _preprocesses and _kernels iterations will be carried out.
        /*!
         *
         * \param times Specifies the number of samples.
         * \param store_auxiliary determines if the auxiliary data created as a by product of
         * the experiments is going to be stored or not.
         */
        void Run(unsigned int times = 1, bool store_auxiliary = false) override;
        //! Adds a dataLoader to the experiment.
        //! Each target data needs to be loaded to experiment via a LoadDataFunction.
        /*!
         *
         * \param func The function which follows the LoadDataFunction definition.
         * \param targets File paths and file specific parameters.
         */
        void AddDataLoader(LoadDataFunction func, std::vector<std::pair<std::vector<std::string>, std::any>> targets) override;
        //! Adds a preprocessing to the experiment.
        //! Each preprocessing operation is added to experiment via a PreprocessFunction.
        /*!
         *
         * \param id Preprocessing id, used to store results, runtimes and auxiliary data.
         * \param func The function which follows the PreprocessFunction definition.
         */
        void AddPreprocess(std::string id, PreprocessFunction func, std::any params) override;
        //! Adds a kernel to the experiment.
        //! Each kernel is added to experiment via a KernelFunction.
        /*!
         *
         * \param id Kernel ID is used to store results, runtimes and auxiliary data.
         * \param func The function which follows the KernelFunction definition.
         */
        void AddKernel(std::string id, KernelFunction func, std::any params) override;
        //!
        /*!
         *
         * \return Returns the runtimes of each kernel call.
         */
        std::map<std::string, std::vector<double>> GetRunTimes() override;
        //!
        /*!
         *
         * \return Returns the results of each kernel call.
         */
        std::map<std::string, std::vector<std::any>> GetResults() override;
        //!
        /*!
         *
         * \return Returns the auxiliary data stored during the experiment.
         */
        std::map<std::string, std::any> GetAuxiliary() override;
    protected:
        //! Stores the file_paths of the files to be extracted,
        //! along with the file specific parameters attached.
        std::vector<std::vector<std::pair<std::vector<std::string>, std::any>>> _targets;
        //! Stores the kernel functions that are going to be run.
        std::unordered_map<std::string, std::pair<KernelFunction, std::any>> _kernels;
        ////! Stores the kernel params.
        //std::unordered_map<std::string, std::any> _kernel_parameters;
        //! Stores dataLoaders.
        std::vector<LoadDataFunction> _dataLoaders;
        //! Stores preprocesses.
        std::unordered_map<std::string, std::pair<PreprocessFunction, std::any>> _preprocesses;
        //! Stores auxiliary data.
        std::map<std::string, std::any> _auxiliary;
        //! Stores runtimes.
        //! The identifier is generated as a comma delimited string of target,preprocess_id, and kernel_id respectively.
        std::map<std::string, std::vector<double>> _runtimes;
        //! Stores results generated by the kernels.
        //! The identifier is generated as a comma delimited string of target,preprocess_id, and kernel_id respectively.
        std::map<std::string, std::vector<std::any>> _results;
};

//! Example dataLoader function.
//! Generic DataLoader.
template< template<typename, typename, typename> typename FormatType, template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::string, format::Format*> LoadFormat(std::vector<std::string> & file_names) {
  auto reader = ReaderType<IDType, NNZType, ValueType>(file_names[0]);
  format::FormatOrderTwo<IDType, NNZType, ValueType> * coo = reader.ReadCOO();
  FormatType<IDType, NNZType, ValueType> * format = coo->template Convert<FormatType>();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", format);
  return r;
}

//! Example dataLoader function.
//! CSR Loader.
template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::string, format::Format*> LoadCSR(std::vector<std::string> & file_names) {
  auto reader = ReaderType<IDType, NNZType, ValueType>(file_names[0]);
  format::CSR<IDType, NNZType, ValueType> * csr = reader.ReadCSR();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csr);
  return r;
}

//! Example dataLoader function.
//! COO Loader.
template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::string, format::Format*> LoadCOO(std::vector<std::string> & file_names) {
  auto reader = ReaderType<IDType, NNZType, ValueType>(file_names[0]);
  format::COO<IDType, NNZType, ValueType> * csr = reader.ReadCOO();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csr);
  return r;
}

//! Example dataLoader function.
//! CSC Loader.
template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::string, format::Format*> LoadCSC(std::vector<std::string> & file_names) {
  auto reader = ReaderType<IDType, NNZType, ValueType>(file_names[0]);
  format::COO<IDType, NNZType, ValueType> * coo = reader.ReadCOO();
  format::CSC<IDType, NNZType, ValueType> * csc = coo->template Convert<format::CSC>();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csc);
  return r;
}

//! example preprocessing function.
//! Generic reordering function for CSR format.
template< template<typename, typename, typename> typename ReorderType, typename ContextType, typename IDType, typename NNZType, typename ValueType>
void ReorderCSR(std::unordered_map<std::string, format::Format*> & data, std::any params) {
  ContextType context;
  auto p = std::any_cast<typename ReorderType<IDType, NNZType, ValueType>::ParamsType>(params);
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>(p, data["format"]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>(), {&context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<format::CSR>(perm, data["format"]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>(), {&context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  data.emplace("processed_format", A_csc);
}

//! Example preprocessing function.
//! Does nothing, can be used to run the experiment with the original data.
inline void Pass(std::unordered_map<std::string, format::Format*> & data, std::any params) {
  data["processed_format"] = data["format"];
}

//! Example preprocessing function.
//! Generic reordering function.
template< template<typename, typename, typename> typename ReorderType, template<typename, typename, typename> typename FormatType, typename ContextType, typename IDType, typename NNZType, typename ValueType>
void Reorder(std::unordered_map<std::string, format::Format*> & data, std::any params) {
  ContextType context;
  auto p = std::any_cast<typename ReorderType<IDType, NNZType, ValueType>::ParamsType>(params);
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>(p, data["format"]->AsAbsolute<FormatType<IDType, NNZType, ValueType>>(), {&context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<FormatType>(perm, data["format"]->AsAbsolute<FormatType<IDType, NNZType, ValueType>>(), {&context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  data.emplace("processed_format", A_csc);
}


} // sparsebase::experiment

#ifdef _HEADER_ONLY
#include "sparsebase/experiment/experiment.cc"
#endif

#endif