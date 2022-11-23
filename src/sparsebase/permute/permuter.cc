#include "sparsebase/permute/permuter.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/array.h"

namespace sparsebase::permute {
template <typename InputFormatType, typename ReturnFormatType>
std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
Permuter<InputFormatType, ReturnFormatType>::GetPermutationCached(format::Format *format,
std::vector<context::Context *> contexts,
bool convert_input) {
//if (dynamic_cast<InputFormatType *>(format) == nullptr)
//  throw utils::TypeException(format->get_name(),
//                             InputFormatType::get_name_static());
return this->CachedExecute(this->params_.get(), contexts,
    convert_input, false, format);
}

template <typename InputFormatType, typename ReturnFormatType>
std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
Permuter<InputFormatType, ReturnFormatType>::GetPermutationCached(format::Format *format, utils::Parameters *params,
    std::vector<context::Context *> contexts,
bool convert_input) {
//if (dynamic_cast<InputFormatType *>(format) == nullptr)
//  throw utils::TypeException(format->get_name(),
//                             InputFormatType::get_name_static());
return this->CachedExecute(params, contexts, convert_input,
false, format);
}

template <typename InputFormatType, typename ReturnFormatType>
ReturnFormatType *
Permuter<InputFormatType, ReturnFormatType>::GetPermutation(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  ////if (dynamic_cast<InputFormatType *>(format) == nullptr)
  ////  throw utils::TypeException(format->get_name(),
  ////                             InputFormatType::get_name_static());
  return this->Execute(this->params_.get(), contexts,
                       convert_input, format);
}

template <typename InputFormatType, typename ReturnFormatType>
ReturnFormatType *
Permuter<InputFormatType, ReturnFormatType>::GetPermutation(
    format::Format *format, utils::Parameters *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  ////if (dynamic_cast<InputFormatType *>(format) == nullptr)
  ////  throw utils::TypeException(format->get_name(),
  ////                             InputFormatType::get_name_static());
  return this->Execute(params, contexts, convert_input,
                       format);
}

template <typename InputFormatType, typename ReturnFormtType>
Permuter<InputFormatType,
    ReturnFormtType>::~Permuter() = default;


#if !defined(_HEADER_ONLY)
#include "init/permuter.inc"
#endif

}