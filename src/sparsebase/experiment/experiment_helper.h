#include "sparsebase/experiment/experiment_type.h"
#include "sparsebase/format/format.h"
#include <unordered_map>
#include <string>
#include <vector>

#ifndef SPARSEBASE_PROJECT_EXPERIMENT_HELPER_H
#define SPARSEBASE_PROJECT_EXPERIMENT_HELPER_H

namespace sparsebase::experiment {

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
void ReorderCSR(std::unordered_map<std::string, format::Format*> & data, std::any fparams, std::any params) {
  ContextType context;
  auto p = std::any_cast<typename ReorderType<IDType, NNZType, ValueType>::ParamsType>(params);
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>(p, data["format"]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>(), {&context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<format::CSR>(perm, data["format"]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>(), {&context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  data.emplace("processed_format", A_csc);
}

//! Example preprocessing function.
//! Does nothing, can be used to run the experiment with the original data.
inline void Pass(std::unordered_map<std::string, format::Format*> & data, std::any fparams, std::any params) {
  data["processed_format"] = data["format"];
}

//! Example preprocessing function.
//! Generic reordering function.
template< template<typename, typename, typename> typename ReorderType, template<typename, typename, typename> typename FormatType, typename ContextType, typename IDType, typename NNZType, typename ValueType>
void Reorder(std::unordered_map<std::string, format::Format*> & data, std::any fparams, std::any params) {
  ContextType context;
  auto p = std::any_cast<typename ReorderType<IDType, NNZType, ValueType>::ParamsType>(params);
  auto *perm = preprocess::ReorderBase::Reorder<ReorderType>(p, data["format"]->AsAbsolute<FormatType<IDType, NNZType, ValueType>>(), {&context}, true);
  auto * A_reordered = preprocess::ReorderBase::Permute2D<FormatType>(perm, data["format"]->AsAbsolute<FormatType<IDType, NNZType, ValueType>>(), {&context}, true);
  auto *A_csc = A_reordered->template Convert<format::CSR>();
  data.emplace("processed_format", A_csc);
}
}

#endif  // SPARSEBASE_PROJECT_EXPERIMENT_HELPER_H
