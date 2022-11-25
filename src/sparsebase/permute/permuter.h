#include <any>
#include <cmath>
#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#ifndef SPARSEBASE_PROJECT_PERMUTER_H
#define SPARSEBASE_PROJECT_PERMUTER_H

namespace sparsebase::permute {
//! Permutes a format according to an inverse permutation of its rows/columns
template <typename InputFormatType, typename ReturnFormatType>
class Permuter : public utils::FunctionMatcherMixin<ReturnFormatType *> {
 public:
  Permuter() {
    static_assert(std::is_base_of<format::Format, InputFormatType>::value,
                  "Permuter must take as input a Format object");
    static_assert(std::is_base_of<format::Format, ReturnFormatType>::value,
                  "Permuter must return a Format object");
  }
  //! Permute `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return a transformed Format object
   */
  ReturnFormatType *GetPermutation(format::Format *csr,
                                   std::vector<context::Context *>,
                                   bool convert_input);
  //! Permute `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param params a polymorphic pointer at a params object
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return a transformed Format object
   */
  ReturnFormatType *GetPermutation(format::Format *csr,
                                   utils::Parameters *params,
                                   std::vector<context::Context *>,
                                   bool convert_input);
  //! Permute `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * a transformed Format object.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
  GetPermutationCached(format::Format *csr,
                       std::vector<context::Context *> contexts,
                       bool convert_input);
  //! Permute `format` to a new format according to an inverse permutation
  //! using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be transformed.
   * @param params a polymorphic pointer at a params object
   * @param contexts vector of contexts that can be used for generating
   * transformation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * a transformed Format object.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, ReturnFormatType *>
  GetPermutationCached(format::Format *csr, utils::Parameters *params,
                       std::vector<context::Context *> contexts,
                       bool convert_input);
  virtual ~Permuter();
};

}  // namespace sparsebase::permute
#ifdef _HEADER_ONLY
#include "sparsebase/permute/permuter.cc"
#endif

#endif  // SPARSEBASE_PROJECT_PERMUTER_H
