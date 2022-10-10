
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

//! Function template for loading data to the experiment.
//! All data that is going to be used in the experiment must be provided via a function that follows this template.
/*!
 *
 * \param file_path Path of the file to be read.
 * \return data extracted from the file.
 */
using LoadDataFunction = std::unordered_map<std::string, format::Format*> (std::string & file_path);
//! Function template for preprocess.
//! All preprocessing methods that are going to be applied on the data must follow this function definition.
/*!
 *
 * \param data Contains the data needed for the preprocessing. This map is returned by the DataLoader
 */
using PreprocessFunction = void (std::unordered_map<std::string, format::Format*> & data);
//! Function template for kernel.
//! All kernel methods that are going to be run must follow this function definition.
/*!
 *
 * \param data Contains the data needed for the preprocessing. This map is returned by the DataLoader and extended by the preprocessing.
 * \param fparams Parameters specific to the target file.
 * \param kparams Parameters specific to the kernel.
 * \return The results of the kernel. The results are stored in _results.
 */
using KernelFunction = std::any (std::unordered_map<std::string, format::Format*> & data, std::any fparams, std::any kparams);

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
        virtual void AddDataLoader(LoadDataFunction func, std::vector<std::pair<std::string, std::any>> targets) = 0;
        //! Adds a preprocessing to the experiment.
        //! Each preprocessing operation is added to experiment via a PreprocessFunction.
        /*!
         *
         * \param id Preprocessing ID is used to store results, runtimes and auxiliary data.
         * \param func The function which follows the PreprocessFunction definition.
         */
        virtual void AddPreprocess(std::string id, PreprocessFunction func) = 0;
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
        void AddDataLoader(LoadDataFunction func, std::vector<std::pair<std::string, std::any>> targets) override;
        //! Adds a preprocessing to the experiment.
        //! Each preprocessing operation is added to experiment via a PreprocessFunction.
        /*!
         *
         * \param id Preprocessing id, used to store results, runtimes and auxiliary data.
         * \param func The function which follows the PreprocessFunction definition.
         */
        void AddPreprocess(std::string id, PreprocessFunction func) override;
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
        std::vector<std::vector<std::pair<std::string, std::any>>> _targets;
        //! Stores the kernel functions that are going to be run.
        std::unordered_map<std::string, std::function<KernelFunction>> _kernels;
        //! Stores the kernel params.
        std::unordered_map<std::string, std::any> _kernel_parameters;
        //! Stores dataLoaders.
        std::vector<std::function<LoadDataFunction>> _dataLoaders;
        //! Stores preprocesses.
        std::unordered_map<std::string, std::function<PreprocessFunction>> _preprocesses;
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
template< template<typename, typename, typename> typename FormatType, template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadFormat(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::FormatOrderTwo<IDType, NNZType, VALType> * coo = reader.ReadCOO();
  FormatType<IDType, NNZType, VALType> * format = coo->template Convert<FormatType>();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", format);
  return r;
}

//! Example dataLoader function.
//! CSR Loader.
template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadCSR(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::CSR<IDType, NNZType, VALType> * csr = reader.ReadCSR();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csr);
  return r;
}

//! Example dataLoader function.
//! COO Loader.
template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadCOO(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::COO<IDType, NNZType, VALType> * csr = reader.ReadCOO();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csr);
  return r;
}

//! Example dataLoader function.
//! CSC Loader.
template< template<typename, typename, typename> typename ReaderType, typename IDType, typename NNZType, typename VALType>
std::unordered_map<std::string, format::Format*> LoadCSC(std::string & file_name) {
  auto reader = ReaderType<IDType, NNZType, VALType>(file_name);
  format::COO<IDType, NNZType, VALType> * coo = reader.ReadCOO();
  format::CSC<IDType, NNZType, VALType> * csc = coo->template Convert<format::CSC>();
  std::unordered_map<std::string, format::Format*> r;
  r.emplace("format", csc);
  return r;
}

//! example preprocessing function.
//! Generic reordering function for CSR format.
template< template<typename, typename, typename> typename ReorderType, typename ContextType, typename IDType, typename NNZType, typename VALType>
void ReorderCSR(std::unordered_map<std::string, format::Format*> & data) {
  ContextType context;
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>({}, data["format"]->As<format::CSR<IDType, NNZType, VALType>>(), {&context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<format::CSR>(perm, data["format"]->As<format::CSR<IDType, NNZType, VALType>>(), {&context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  data.emplace("processed_format", A_csc);
}

//! Example preprocessing function.
//! Does nothing, can be used to run the experiment with the original data.
template<typename IDType, typename NNZType, typename VALType>
void Pass(std::unordered_map<std::string, format::Format*> & data) {
  data["processed_format"] = data["format"];
}

//! Example preprocessing function.
//! Generic reordering function.
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