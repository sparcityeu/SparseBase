/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_

#include "sparsebase/config.h"
#include <functional>
#include <tuple>
#include <unordered_map>
#ifdef USE_CUDA
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/utils/converter/cuda/converter.cuh"
#endif

// Forward declerations for the `Convert` functions in sparsebase/format/format.h
namespace sparsebase {
namespace utils {
namespace converter {
class Converter;
template <class ConverterType> 
class ConverterImpl;
template <typename IDType, typename NNZType, typename ValueType>
class ConverterOrderTwo;
template <typename ValueType>
class ConverterOrderOne;
    
} // namespace converter
} // namespace utils
} // namespace sparsebase

#include "sparsebase/format/format.h"
namespace sparsebase {

namespace utils {

namespace converter {

//! A conversion schema is a way to store a conversion to a particular type and
//! context
typedef std::vector<std::tuple<bool, std::type_index, context::Context *>>
    ConversionSchemaConditional;

//! Any function that matches the signature of this definition can be registered
//! in Converter instances
using ConditionalConversionFunction =
    std::function<format::Format *(format::Format *, context::Context *)>;

//! A function type that returns true if the relevant conversion function can
//! convert between the contexts in the parameters
using EdgeConditional =
    std::function<bool(context::Context *, context::Context *)>;

//! Base class for all the converter classes
class Converter {
private:
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         std::vector<std::tuple<
                             EdgeConditional, ConditionalConversionFunction>>>>
      conditional_map_;
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         std::vector<std::tuple<
                             EdgeConditional, ConditionalConversionFunction>>>>
      conditional_move_map_;
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         std::vector<std::tuple<
                             EdgeConditional, ConditionalConversionFunction>>>>
      *get_conversion_map(bool is_move_conversion);

public:
  void RegisterConditionalConversionFunction(
      std::type_index from_type, std::type_index to_type,
      ConditionalConversionFunction conv_func, EdgeConditional edge_condition,
      bool is_move_conversion = false);

  /*!
   * Helper function used to automatically select the correct conversion
   * function from all the registered functions
   * @param from_type type_index for the source instance
   * @param from_context context for the source instance
   * @param to_type type_index for the target instance
   * @param to_context context for the target instance
   * @param is_move_conversion if true a move conversion function will be
   * returned (moves the arrays instead of copying them when possible)
   * @return an std::function instance that can perform the desired conversion
   */
  ConditionalConversionFunction
  GetConversionFunction(std::type_index from_type,
                        context::Context *from_context, std::type_index to_type,
                        context::Context *to_context,
                        bool is_move_conversion = false);

  /*!
   * Converts the source to to_type by automatically selecting and using a
   * registered conversion function
   * @param source a pointer to the source Format instance
   * @param to_type a type_index for the type to convert to (for example:
   * std::typeid(COO<...>) can be used)
   * @param to_context context used for the conversion (see the context
   * namespace for more information)
   * @param is_move_conversion if true the underlying arrays will be moved
   * instead of being copied
   * @return a pointer to the converted Format instance
   */
  format::Format *Convert(format::Format *source, std::type_index to_type,
                          context::Context *to_context,
                          bool is_move_conversion = false);

  /*! An overload of the Convert function where the to_type parameter is
   * provided as a template Usage: Convert<COO<...>>(...) Note that sometimes it
   * might be necessary to use the "template" keyword before the function name
   * in the call.
   */
  template <typename FormatType>
  FormatType *Convert(format::Format *source, context::Context *to_context,
                      bool is_move_conversion = false) {
    auto *res = this->Convert(source, FormatType::get_format_id_static(),
                              to_context, is_move_conversion);
    return res->template As<FormatType>();
  }

  std::tuple<bool, context::Context *>
  CanConvert(std::type_index from_type, context::Context *from_context,
             std::type_index to_type,
             const std::vector<context::Context *> &to_contexts,
             bool is_move_conversion = false);

  //! Returns true if a conversion from (from_type, from_context) to (to_type,
  //! to_context) is possible
  bool CanConvert(std::type_index from_type, context::Context *from_context,
                  std::type_index to_type, context::Context *to_context,
                  bool is_move_conversion = false);

  /*! Removes conversion functions that convert from `from_type` to `to_type` from
   * the conversion map
   * @param from_type type_index of the original format type
   * @param to_type type_index of the destination format type
   * @param move_conversion whether the conversions to be removed are move conversions or not
   */
  void ClearConversionFunctions(std::type_index from_type, std::type_index to_type, bool move_conversion = false);
  
  /*! Removes all conversion functions from the current converter
   * @param move_conversion whether the conversions to be removed are move conversions or not
   */
  void ClearConversionFunctions(bool move_conversion = false);
  
  std::vector<format::Format *>
  ApplyConversionSchema(ConversionSchemaConditional cs,
                        std::vector<format::Format *> packed_sfs,
                        bool is_move_conversion = false);
  virtual std::type_index get_converter_type() = 0;
  virtual Converter *Clone() const = 0;
  virtual void Reset() = 0;
  virtual ~Converter();
};

/*!
 * Intermediate class used to implement CRTP (curiously recurring template
 * pattern). Under most circumstances, users don't need to interact with this
 * class unless they are defining their own converter classes.
 *
 * @tparam ConverterType Concrete type of the Converter class that is derived
 * from this class
 */
template <class ConverterType> class ConverterImpl : public Converter {
public:
  //! Returns the type_index for the Converter instance
  virtual std::type_index get_converter_type() { return typeid(ConverterType); }
};

//! An instance of this class can be used to convert between order two formats
//! (CSR and COO)
template <typename IDType, typename NNZType, typename ValueType>
class ConverterOrderTwo
    : public ConverterImpl<ConverterOrderTwo<IDType, NNZType, ValueType>> {
public:
  ConverterOrderTwo();
  virtual Converter *Clone() const;
  virtual void Reset();
};

//! An instance of this class can be used to convert between order one formats
//! (Array)
template <typename ValueType>
class ConverterOrderOne : public ConverterImpl<ConverterOrderOne<ValueType>> {
public:
  ConverterOrderOne();
  virtual Converter *Clone() const;
  virtual void Reset();
};

} // namespace converter

} // namespace utils

} // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparsebase/utils/converter/converter.cc"
#ifdef USE_CUDA
#include "cuda/converter.cu"
#endif
#endif

#endif // SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_
