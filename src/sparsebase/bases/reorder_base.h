#include "sparsebase/context/cpu_context.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/permute/permute_order_one.h"
#include "sparsebase/permute/permute_order_two.h"
#include "sparsebase/reorder/rcm_reorder.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/amd_reorder.h"
#include "sparsebase/reorder/generic_reorder.h"
#include "sparsebase/reorder/gray_reorder.h"
#include "sparsebase/reorder/metis_reorder.h"
#include "sparsebase/reorder/rabbit_reorder.h"
#include "sparsebase/reorder/reorder_heatmap.h"
#include "sparsebase/reorder/reorderer.h"

#ifndef SPARSEBASE_PROJECT_REORDER_BASE_H
#define SPARSEBASE_PROJECT_REORDER_BASE_H

namespace sparsebase::bases {

//! A class containing the interface for reordering and permuting data.
/*!
 * The class contains all the functionalities needed for reordering. That
 * includes a function generate reordering permutations from data, functions to
 * permute data using a permutation vector, and a function to inverse the
 * permutation of data. In the upcoming release, ReorderBase will include
 * functions to extract feeatures from permutation vectors and permuted data.
 */
class ReorderBase {
 public:
  //! Generates a permutation array from a FormatOrderTwo object using the
  //! Reordering class `Reordering`.
  /*!
   *
   * @tparam Reordering a reordering class defining a reordering algorithm. For
   * a full list of available reordering algorithms, please check
   * [here](../pages/getting_started/available.html).
   * @param params a struct containing the parameters specific for the
   * reordering algorithm `Reordering`. Please check the documentation of each
   * reordering for the specifications of its parameters.
   * @param format FormatOrderTwo object to be used to generate permutation
   * array.
   * @param contexts vector of contexts that can be used for permutation.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the permutation array.
   */
  template <template <typename, typename, typename> typename Reordering,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static AutoIDType *Reorder(
      typename Reordering<AutoIDType, AutoNNZType, AutoValueType>::ParamsType
      params,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input) {
    static_assert(
        std::is_base_of_v<reorder::Reorderer<AutoIDType>,
        Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
    "You must pass a reordering function (with base Reorderer) "
    "to ReorderBase::Reorder");
    static_assert(
        !std::is_same_v<reorder::GenericReorder<AutoIDType, AutoNNZType, AutoValueType>,
        Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
    "You must pass a reordering function (with base Reorderer) "
    "to ReorderBase::Reorder");
    Reordering<AutoIDType, AutoNNZType, AutoValueType> reordering(params);
    return reordering.GetReorder(format, contexts, convert_input);
  }
  //! Generates a permutation array from a FormatOrderTwo object using the
  //! Reordering class `Reordering` with cached output.
  /*!
   *
   * @tparam Reordering a reordering class defining a reordering algorithm. For
   * a full list of available reordering algorithms, please check: xxx
   * @param params a struct containing the parameters specific for the
   * reordering algorithm `Reordering`. Please check the documentation of each
   * reordering for the specifications of its parameters.
   * @param format FormatOrderTwo object to be used to generate permutation
   * array.
   * @param contexts vector of contexts that can be used for permutation.
   * @return An std::pair with the second element being the permutation array,
   * and the first being a vector of all the formats generated by converting the
   * input (if such conversions were needed to execute the permutation).
   */
  template <template <typename, typename, typename> typename Reordering,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
      AutoValueType> *>,
  AutoIDType *>
  ReorderCached(
      typename Reordering<AutoIDType, AutoNNZType, AutoValueType>::ParamsType
      params,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts) {
    static_assert(
        std::is_base_of_v<reorder::Reorderer<AutoIDType>,
        Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
        "You must pass a reordering function (with base Reorderer) "
        "to ReorderBase::Reorder");
    static_assert(
        !std::is_same_v<reorder::GenericReorder<AutoIDType, AutoNNZType, AutoValueType>,
        Reordering<AutoIDType, AutoNNZType, AutoValueType>>,
        "You must pass a reordering function (with base Reorderer) "
        "to ReorderBase::Reorder");
    Reordering<AutoIDType, AutoNNZType, AutoValueType> reordering(params);
    auto output = reordering.GetReorderCached(format, contexts, true);
    std::vector<
    format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
                                                    converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    return std::make_pair(converted_formats, std::get<1>(output));
  }

  //! Permute a two-dimensional format row- and column-wise using a single
  //! permutation array for both axes.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
      typename ReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *Permute2D(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 ordering);
    auto out_format = perm.GetPermutation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format row- and column-wise using a single
  //! permutation array for both axes with cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param ordering Permutation
   * array to use when permuting rows and columns. @param format object to be
   * permuted. @param contexts vector of contexts that can be used for
   * permutation. @param convert_output if true, the returned object will be
   * converted to `ReturnFormatType`. Otherwise, the returned object will be
   * cast to `ReturnFormatType`, and if the cast fails, an exception of type
   * `sparsebase::utils::TypeException`. @return An std::pair with the second
   * element being the permuted format, and the first being a vector of all the
   * formats generated by converting the input (if such conversions were needed
   * to execute the permutation). By default, the permuted object is returned as
   * a pointer at a generic FormatOrderTwo object. However, if the user passes a
   * concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the
   * returned format will be converted to that type. If not, the returned object
   * will only be cast to that type (if casting fails, an exception of type
   * utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
      typename ReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
      AutoValueType> *>,
  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DCached(
      AutoIDType *ordering,
  format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 ordering);
    auto output = perm.GetPermutationCached(format, contexts, true);
    std::vector<
    format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
                                                    converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<ReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a two-dimensional format row- and column-wise using a permutation
  //! array for each axis with cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param row_ordering
   * Permutation array to use when permuting rows. @param col_ordering
   * Permutation array to use when permuting col. @param format object to be
   * permuted. @param contexts vector of contexts that can be used for
   * permutation. @param convert_output if true, the returned object will be
   * converted to `ReturnFormatType`. Otherwise, the returned object will be
   * cast to `ReturnFormatType`, and if the cast fails, an exception of type
   * `sparsebase::utils::TypeException`. @return An std::pair with the second
   * element being the permuted format, and the first being a vector of all the
   * formats generated by converting the input (if such conversions were needed
   * to execute the permutation). By default, the permuted object is returned as
   * a pointer at a generic FormatOrderTwo object. However, if the user passes a
   * concrete FormatOrderTwo class as the templated parameter
   * `ReturnFormatType`, e.g. format::CSR, then if `convert_output` is true, the
   * returned format will be converted to that type. If not, the returned object
   * will only be cast to that type (if casting fails, an exception of type
   * utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
      typename ReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
      AutoValueType> *>,
  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DRowColumnWiseCached(
      AutoIDType *row_ordering, AutoIDType *col_ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
  std::vector<context::Context *> contexts, bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(row_ordering,
                                                                 col_ordering);
    auto output = perm.GetPermutationCached(format, contexts, true);
    std::vector<
    format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
                                                    converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<ReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a two-dimensional format row- and column-wise using a permutation
  //! array for each axis.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
      typename ReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *
  Permute2DRowColumnWise(
      AutoIDType *row_ordering, AutoIDType *col_ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(row_ordering,
                                                                 col_ordering);
    auto out_format = perm.GetPermutation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format row-wise using a permutation array.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
      typename ReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *
  Permute2DRowWise(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 nullptr);
    auto out_format = perm.GetPermutation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format row-wise using a permutation array with
  //! cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param ordering Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_output if true,
   * the returned object will be converted to `ReturnFormatType`. Otherwise, the
   * returned object will be cast to `ReturnFormatType`, and if the cast fails,
   * an exception of type `sparsebase::utils::TypeException`. @return An
   * std::pair with the second element being the permuted format, and the first
   * being a vector of all the formats generated by converting the input (if
   * such conversions were needed to execute the permutation). By default, the
   * permuted object is returned as a pointer at a generic FormatOrderTwo
   * object. However, if the user passes a concrete FormatOrderTwo class as the
   * templated parameter `ReturnFormatType`, e.g. format::CSR, then if
   * `convert_output` is true, the returned format will be converted to that
   * type. If not, the returned object will only be cast to that type (if
   * casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
      typename RelativeReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<
      std::vector<
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>,
  RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DRowWiseCached(
      AutoIDType *ordering,
  format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(ordering,
                                                                 nullptr);
    auto output = perm.GetPermutationCached(format, contexts, true);
    std::vector<
    format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
                                                    converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                  RelativeReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<RelativeReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<RelativeReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a two-dimensional format column-wise using a permutation array.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderTwo object. However, if the user passes a concrete
   * FormatOrderTwo class as the templated parameter `ReturnFormatType`, e.g.
   * format::CSR, then if `convert_output` is true, the returned format will be
   * converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename, typename, typename>
      typename ReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *
  Permute2DColWise(
      AutoIDType *ordering,
      format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_input,
      bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(nullptr,
                                                                 ordering);
    auto out_format = perm.GetPermutation(format, contexts, convert_input);
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *output;
    if constexpr (std::is_same_v<
                  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a two-dimensional format column-wise using a permutation array
  //! with cached output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderTwo. Defines the
   * return pointer type. Default is FormatOrderTwo. @param ordering Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_output if true,
   * the returned object will be converted to `ReturnFormatType`. Otherwise, the
   * returned object will be cast to `ReturnFormatType`, and if the cast fails,
   * an exception of type `sparsebase::utils::TypeException`. @return An
   * std::pair with the second element being the permuted format, and the first
   * being a vector of all the formats generated by converting the input (if
   * such conversions were needed to execute the permutation). By default, the
   * permuted object is returned as a pointer at a generic FormatOrderTwo
   * object. However, if the user passes a concrete FormatOrderTwo class as the
   * templated parameter `ReturnFormatType`, e.g. format::CSR, then if
   * `convert_output` is true, the returned format will be converted to that
   * type. If not, the returned object will only be cast to that type (if
   * casting fails, an exception of type utils::TypeException will be thrown).
   */
  template <template <typename, typename, typename>
      typename ReturnFormatType = format::FormatOrderTwo,
      typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderTwo<AutoIDType, AutoNNZType,
      AutoValueType> *>,
  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> *>
  Permute2DColWiseCached(
      AutoIDType *ordering,
  format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *format,
      std::vector<context::Context *> contexts, bool convert_output = false) {
    permute::PermuteOrderTwo<AutoIDType, AutoNNZType, AutoValueType> perm(nullptr,
                                                                 ordering);
    auto output = perm.GetPermutationCached(format, contexts, true);
    std::vector<
    format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>
                                                    converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<
              format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType> * output_format;
    if constexpr (std::is_same_v<
                  ReturnFormatType<AutoIDType, AutoNNZType, AutoValueType>,
                  format::FormatOrderTwo<AutoIDType, AutoNNZType,
                  AutoValueType>>)
    output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format =
            std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format =
            std::get<1>(output)->template As<ReturnFormatType>();
    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Permute a one-dimensional format using a permutation array.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderOne. Defines the
   * return pointer type. Default is FormatOrderOne. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_input whether or
   * not to convert the input format if that is needed. @param convert_output if
   * true, the returned object will be converted to `ReturnFormatType`.
   * Otherwise, the returned object will be cast to `ReturnFormatType`, and if
   * the cast fails, an exception of type `sparsebase::utils::TypeException`.
   * @return The permuted format. By default, the function returns a pointer at
   * a generic FormatOrderOne object. However, if the user passes a concrete
   * FormatOrderOne class as the templated parameter `ReturnFormatType`, e.g.
   * format::Array, then if `convert_output` is true, the returned format will
   * be converted to that type. If not, the returned object will only be cast to
   * that type (if casting fails, an exception of type utils::TypeException will
   * be thrown).
   */
  template <template <typename>
      typename ReturnFormatType = format::FormatOrderOne,
      typename AutoIDType, typename AutoValueType>
  static ReturnFormatType<AutoValueType> *Permute1D(
      AutoIDType *ordering, format::FormatOrderOne<AutoValueType> *format,
      std::vector<context::Context *> context, bool convert_inputs,
      bool convert_output = false) {
    permute::PermuteOrderOne<AutoIDType, AutoValueType> perm(ordering);
    auto out_format = perm.GetPermutation(format, context, convert_inputs);
    ReturnFormatType<AutoValueType> * output;
    if constexpr (std::is_same_v<ReturnFormatType<AutoValueType>,
                  format::FormatOrderOne<AutoValueType>>)
    output = out_format;
    else {
      if (convert_output)
        output = out_format->template Convert<ReturnFormatType>();
      else
        output = out_format->template As<ReturnFormatType>();
    }
    return output;
  }

  //! Permute a one-dimensional format using a permutation array with cached
  //! output.
  /*!
   *
   * @tparam ReturnFormatType a child class of type FormatOrderOne. Defines the
   * return pointer type. Default is FormatOrderOne. @param order Permutation
   * array. @param format object to be permuted. @param contexts vector of
   * contexts that can be used for permutation. @param convert_output if true,
   * the returned object will be converted to `ReturnFormatType`. Otherwise, the
   * returned object will be cast to `ReturnFormatType`, and if the cast fails,
   * an exception of type `sparsebase::utils::TypeException`. @return An
   * std::pair with the second element being the permuted format, and the first
   * being a vector of all the formats generated by converting the input (if
   * such conversions were needed to execute the permutation). By default, the
   * permuted object is returned as a pointer at a generic FormatOrderOne
   * object. However, if the user passes a FormatOrderOne class as the templated
   * parameter `ReturnFormatType`, e.g. format::Array, then if `convert_output`
   * is true, the returned format will be converted to that type. If not, the
   * returned object will only be cast to that type (if casting fails, an
   * exception of type utils::TypeException will be thrown).
   */
  template <template <typename>
      typename ReturnFormatType = format::FormatOrderOne,
      typename AutoIDType, typename AutoValueType>
  static std::pair<std::vector<format::FormatOrderOne<AutoValueType> *>,
  ReturnFormatType<AutoValueType> *>
  Permute1DCached(AutoIDType *ordering,
  format::FormatOrderOne<AutoValueType> *format,
      std::vector<context::Context *> context,
  bool convert_output = false) {
    permute::PermuteOrderOne<AutoIDType, AutoValueType> perm(ordering);
    auto output = perm.GetPermutationCached(format, context, true);
    std::vector<format::FormatOrderOne<AutoValueType> *> converted_formats;
    std::transform(
        std::get<0>(output)[0].begin(), std::get<0>(output)[0].end(),
        std::back_inserter(converted_formats),
        [](format::Format *intermediate_format) {
          return static_cast<format::FormatOrderOne<AutoValueType> *>(
              intermediate_format);
        });
    ReturnFormatType<AutoValueType> * output_format;
    if constexpr (std::is_same_v<ReturnFormatType<AutoValueType>,
                  format::FormatOrderOne<AutoValueType>>)
    output_format = std::get<1>(output);
    else {
      if (convert_output)
        output_format = std::get<1>(output)->template Convert<ReturnFormatType>();
      else
        output_format = std::get<1>(output)->template As<ReturnFormatType>();

    }
    return std::make_pair(converted_formats, output_format);
  }

  //! Takes a permutation array and its length and inverses it.
  /*!
   * Takes a permutation array and its length and inverses it. If a format `A`
   * was permuted with `perm` into object `B`, then permuting `B` with the
   * inverse permutation returns its order to `A`.
   * @param perm a permutation array of length `length`
   * @param length the length of the permutation array
   * @return a permutation array of length `length` that is the inverse of
   * `perm`, i.e. can be used to reverse a permutation done by `perm`.
   */
  template <typename AutoIDType, typename AutoNumType>
  static AutoIDType *InversePermutation(AutoIDType *perm, AutoNumType length) {
    static_assert(std::is_integral_v<AutoNumType>,
                  "Length of the permutation array must be an integer");
    auto inv_perm = new AutoIDType[length];
    for (AutoIDType i = 0; i < length; i++) {
      inv_perm[perm[i]] = i;
    }
    return inv_perm;
  }
  //! Calculates density of non-zeros of a 2D format on a num_parts * num_parts grid
  /*!
 * Splits the input 2D matrix into a grid of size num_parts * num_parts containing an
 * equal number of rows and columns, and calculates the density of non-zeros in each
 * cell in the grid relative to the total number of non-zeros in the matrix, given that the
 * matrix was reordered according to a permutation matrix.
 * Returns the densities as a dense array (FormatOrderOne) of size num_parts * num_parts where
 * the density at cell [i][j] in the 2D grid is located at index [i*num_parts+j] in the
 * grid. The density values sum up to 1.
  * @tparam FloatType type used to represent the densities of non-zeros.
  * @param format the 2D matrix to calculate densities for.
  * @param permutation the permutation array containing the reordering of rows and columns.
  * @param num_parts number of parts to split rows/columns over
  * @param contexts vector of contexts that can be used for permutation.
  * @param convert_input whether or not to convert the input format if that is needed.
  * @return a format::Array containing the densities of the cells in the num_parts * num_parts 2D grid.
  */
  template <typename FloatType,  typename AutoIDType, typename AutoNNZType, typename AutoValueType>
  static sparsebase::format::Array<FloatType>* Heatmap(format::FormatOrderTwo<AutoIDType, AutoNNZType, AutoValueType> * format, format::FormatOrderOne<AutoIDType>* permutation_r, format::FormatOrderOne<AutoIDType>* permutation_c, int num_parts,
                                                       std::vector<context::Context *> contexts, bool convert_input) {
    reorder::ReorderHeatmap<AutoIDType, AutoNNZType, AutoValueType, FloatType> heatmapper(num_parts);
    format::FormatOrderOne<FloatType>* arr = heatmapper.Get(format, permutation_r, permutation_c, contexts, convert_input);
    return arr->template Convert<sparsebase::format::Array>();
  }
};
}

#endif  // SPARSEBASE_PROJECT_REORDER_BASE_H
