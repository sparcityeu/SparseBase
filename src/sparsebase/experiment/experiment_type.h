#ifndef SPARSEBASE_PROJECT_EXPERIMENT_TYPE_H
#define SPARSEBASE_PROJECT_EXPERIMENT_TYPE_H

#include <any>
#include <chrono>
#include <map>
#include <unordered_map>
#include <vector>

#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

namespace sparsebase::experiment {

//! Function template for loading data to the experiment.
//! All data that is going to be used in the experiment must be provided via a
//! function that follows this template.
/*!
 *
 * \param file_paths File paths that store the target data.
 * \return data extracted from the files.
 */
using LoadDataFunction =
    std::function<std::unordered_map<std::string, format::Format*>(
        std::vector<std::string>& file_paths)>;
//! Function template for preprocess.
//! All preprocessing methods that are going to be applied on the data must
//! follow this function definition.
/*!
 *
 * \param data Contains the data needed for the preprocessing. This map is
 * returned by the DataLoader \param fparams Parameters specific to the target
 * file. \param params Preprocessing parameters.
 */
using PreprocessFunction =
    std::function<void(std::unordered_map<std::string, format::Format*>& data,
                       std::any& fparams, std::any& params)>;
//! Function template for kernel.
//! All kernel methods that are going to be run must follow this function
//! definition.
/*!
 *
 * \param data Contains the data needed for the preprocessing. This map is
 * returned by the DataLoader and extended by the preprocessing. \param fparams
 * Parameters specific to the target file. \param pparams Parameters specific to
 * the preprocessing. \param kparams Parameters specific to the kernel. \return
 * The results of the kernel. The results are stored in _results.
 */
using KernelFunction = std::function<std::any(
    std::unordered_map<std::string, format::Format*>& data, std::any& fparams,
    std::any& pparams, std::any& kparams)>;

//! Abstract class that defines a common interface for experiments
class ExperimentType {
 public:
  //! Start the experiment.
  //! Each target will be preprocessed by all the preprocessing functions. Then
  //! every kernel will be applied to all of the newly created data. The
  //! cartesian product of the number of _targets, _preprocesses and _kernels
  //! iterations will be carried out.
  /*!
   *
   * \param times Specifies the number of samples.
   * \param store_auxiliary determines if the auxiliary data created as a by
   * product of the experiments is going to be stored or not.
   */
  virtual void Run(unsigned int times, bool store_auxiliary) = 0;
  //! Adds a dataLoader to the experiment.
  //! Each target data needs to be loaded to experiment via a LoadDataFunction.
  /*!
   *
   * \param func The function which follows the LoadDataFunction definition.
   * \param targets File paths and file specific parameters.
   */
  virtual void AddDataLoader(
      LoadDataFunction func,
      std::vector<std::pair<std::vector<std::string>, std::any>> targets) = 0;
  //! Adds a preprocessing to the experiment.
  //! Each preprocessing operation is added to experiment via a
  //! PreprocessFunction.
  /*!
   *
   * \param id Preprocessing ID is used to store results, runtimes and auxiliary
   * data. \param func The function which follows the PreprocessFunction
   * definition. \param params Parameters to be passed to the respective
   * function.
   */
  virtual void AddPreprocess(std::string id, PreprocessFunction func,
                             std::any params) = 0;
  //! Adds a kernel to the experiment.
  //! Each kernel is added to experiment via a KernelFunction.
  /*!
   *
   * \param id Kernel ID is used to store results, runtimes and auxiliary data.
   * \param func The function which follows the KernelFunction definition.
   */
  virtual void AddKernel(std::string id, KernelFunction func,
                         std::any params) = 0;
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
}  // namespace sparsebase::experiment
#endif  // SPARSEBASE_PROJECT_EXPERIMENT_TYPE_H
