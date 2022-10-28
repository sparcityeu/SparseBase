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
#include <optional>

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




//! Any function that matches the signature of this definition can be registered
//! in Converter instances

typedef std::function<format::Format *(format::Format *, context::Context *)>
    ConversionFunction;


//! A function type that returns true if the relevant conversion function can
//! convert between the contexts in the parameters
typedef
std::function<bool(context::Context *, context::Context *)>
    ConversionCondition;

//! A single conversion step composed of a conversion function and a context to use for the function
typedef std::tuple<ConversionFunction, context::Context *, utils::CostType> ConversionStep;

//! A chain of conversion functions-context pairs to be applied sequentially to a single formats
typedef
std::optional<std::tuple<std::vector<ConversionStep>, utils::CostType>>
    ConversionChain;

//! A vector of conversion chains, each for an independent format
typedef std::vector<ConversionChain> ConversionSchema;

//! A map containing conversion functions between format classes. Each function is
//! associated with a conditional function.
typedef std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         std::vector<std::tuple<ConversionCondition, ConversionFunction>>>> ConversionMap;
//! Base class for all the converter classes
class Converter {
private:
  ConversionMap copy_conversion_map_;
  ConversionMap move_conversion_map_;
  //! Returns one of the member conversion maps
  /*!
   *
   * @param is_move_conversion to get the move ConversionMap or the copy one.
   * @return
   */
  ConversionMap *get_conversion_map(bool is_move_conversion);

  //! Returns a conversion path from from_type to to_type using contexts in to_contexts.
  /*!
   * \param from_type type of source format
   * \param from_context context of source format
   * \param to_type format type to convert to
   * \param to_contexts contexts available for conversion
   * \param map the map containing conversion functions available
   * \return a vector of tuples corresponding to conversion steps. Each is a tuple of function
   * and context pair. If an empty vector is returned, a conversion is not possible.
  */
  static std::vector<ConversionStep> ConversionBFS(std::type_index from_type, context::Context* from_context, std::type_index to_type, const std::vector<context::Context*>& to_contexts, ConversionMap* map);
public:
  //! Register a conversion function from one type to another.
  /*!
   * Registers a conversion function from one type to another. The conversion will be conditional on `edge_condition` which is
   * a function that returns a boolean based on the source and destination contexts of the conversion.
   * @param from_type std::type_index of the source format
   * @param to_type std::type_index of the destination format
   * @param conv_func a conversion function taking an input Format* and a destination context.
   * @param edge_condition an std::function containing a boolean function with two Context* parameters for the source and destination
   * contexts. Returns true if the conversion should happen based on context inputs.
   * @param is_move_conversion whether this conversion is a move conversion.
   */
  void RegisterConversionFunction(
      std::type_index from_type, std::type_index to_type,
                                             ConversionFunction conv_func,
                                  ConversionCondition edge_condition,
      bool is_move_conversion = false);

  //! Converts a source format to a destination format.
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

  //! Converts a source format to a destination format with cached output.
  /*!
   * Converts the source to to_type and returns all the formats created
   * during the conversion process.
   * \param source a pointer to the source Format instance
   * \param to_type a type_index for the type to convert to (for example:
   *  std::typeid(COO<...>) can be used)
   * \param to_context context used for the conversion (see the context
   * namespace for more information)
   * \param is_move_conversion if true the underlying arrays will be moved
   * instead of being copied
   * \return a vector of formats, with the last (back) format being the
   * target format, and the ones before it being intermediate ones.
   */
  std::vector<format::Format *>ConvertCached(format::Format *source, std::type_index to_type,
                          context::Context *to_context,
                          bool is_move_conversion = false);

  //! Converts a source format to a destination format.
  /*!
   * Converts the source to to_type by automatically selecting and using a
   * registered conversion function
   * @param source a pointer to the source Format instance
   * @param to_type a type_index for the type to convert to (for example:
   * std::typeid(COO<...>) can be used)
   * @param to_contexts contexts that may be used for the conversion (see the context
   * namespace for more information)
   * @param is_move_conversion if true the underlying arrays will be moved
   * instead of being copied
   * @return a pointer to the converted Format instance
   */
  format::Format *Convert(format::Format *source, std::type_index to_type,
                          std::vector<context::Context *>to_contexts,
                          bool is_move_conversion = false);

  //! Converts the source to to_type and returns all the formats created
  //! during the conversion process.
  /*!
  * \param source a pointer to the source Format instance
  * \param to_type a type_index for the type to convert to (for example:
  *  std::typeid(COO<...>) can be used)
  * \param to_contexts contexts that may be used for the conversion (see the context
  * namespace for more information)
  * \param is_move_conversion if true the underlying arrays will be moved
  * instead of being copied
  * \return a vector of formats, with the last (back) format being the
  * target format, and the ones before it being intermediate ones.
  */
  std::vector<format::Format *>ConvertCached(format::Format *source, std::type_index to_type,
                                             std::vector<context::Context *>to_context,
                                             bool is_move_conversion = false);
   //! Converts a format to another format and returns a cast pointer to the destination format.
   /*! An overload of the Convert function where the to_type parameter is
    * provided as a template Usage: Convert<COO<...>>(...) Note that sometimes it
    * might be necessary to use the "template" keyword before the function name
    * in the call.
    * @tparam FormatType the destination format.
    * @param source input format object.
    * @param to_context context to be used for conversion
    * @param is_move_conversion whether this is a move conversion
    * @return a pointer at `FormatType` object.
    */
  template <typename FormatType>
  FormatType *Convert(format::Format *source, context::Context *to_context,
                      bool is_move_conversion = false) {
    auto *res = this->Convert(source, FormatType::get_format_id_static(),
                              to_context, is_move_conversion);
    return res->template AsAbsolute<FormatType>();
  }

  //! Converts a format to another format and returns a cast pointer to the destination format.
  /*! An overload of the Convert function where the to_type parameter is
   * provided as a template Usage: Convert<COO<...>>(...) Note that sometimes it
   * might be necessary to use the "template" keyword before the function name
   * in the call.
   * @tparam FormatType the destination format.
   * @param source input format object.
   * @param to_contexts contexts that may be used for the conversion (see the context
   * namespace for more information)
   * @param is_move_conversion whether this is a move conversion
   * @return a pointer at `FormatType` object.
   */
  template <typename FormatType>
  FormatType *Convert(format::Format *source, std::vector<context::Context *>to_contexts,
                      bool is_move_conversion = false) {
    auto *res = this->Convert(source, FormatType::get_format_id_static(),
                              to_contexts, is_move_conversion);
    return res->template AsAbsolute<FormatType>();
  }

  //! Returns a conversion chain needed to convert between two formats
  /*!
   * Takes a single (format, context) input and returns the chain of conversions needed to
   * convert the input to the type `to_type`. The chain is optional, meaning it could be empty
   * if no conversions exist.
   * @param from_type type of source format
   * @param from_context context of source format
   * @param to_type format type to convert to
   * @param to_contexts contexts available for conversion
   * @param is_move_conversion whether to find move conversions or normal conversions
   * @return an optional chain of conversion. If nothing is returned, then no conversions are available. Otherwise,
   * will return a tuple with its second element being the total conversion cost, and its first element a vector of
   * (ConversionFunction, Context) pairs, where the input must be passed through this chain starting from its front
   * to its back to become `to_type`.
   */
  ConversionChain
  GetConversionChain(std::type_index from_type, context::Context *from_context,
             std::type_index to_type,
             const std::vector<context::Context *> &to_contexts,
             bool is_move_conversion = false);

  //! Returns true if a conversion from (from_type, from_context) to (to_type,
  //! to_context) is possible
  /*!
   *
   * @param from_type source format type_index
   * @param from_context source context
   * @param to_type destination format type_index
   * @param to_context destination context
   * @param is_move_conversion whether it is a move conversion
   * @return a boolean indicating whether a conversion is possible
   */
  bool CanConvert(std::type_index from_type, context::Context *from_context,
                  std::type_index to_type, context::Context *to_context,
                  bool is_move_conversion = false);

  //! Returns true if a conversion from (from_type, from_context) to (to_type,
  //! to_context) is possible
  /*!
   *
   * @param from_type source format type_index
   * @param from_context source context
   * @param to_type destination format type_index
   * \param to_contexts contexts available for conversion
   * @param is_move_conversion whether it is a move conversion
   * @return a boolean indicating whether a conversion is possible
   */
  bool CanConvert(std::type_index from_type, context::Context *from_context,
                  std::type_index to_type, const std::vector<context::Context *>&to_contexts,
                  bool is_move_conversion = false);
  //! Clears all conversion functions from one format to another.
  /*! Removes conversion functions that convert from `from_type` to `to_type` from
   * the conversion map
   * @param from_type type_index of the original format type
   * @param to_type type_index of the destination format type
   * @param move_conversion whether the conversions to be removed are move conversions or not
   */
  void ClearConversionFunctions(std::type_index from_type, std::type_index to_type, bool move_conversion = false);

  //! Removes all conversion functions from current converter.
  /*! Removes all conversion functions from the current converter in all directions.
   * @param move_conversion whether the conversions to be removed are move conversions or not
   */
  void ClearConversionFunctions(bool move_conversion = false);

  //! Applies a conversion schema to convert one format to another.
  /*!
   * Takes a conversion schema to convert all the formats in `packed_sfs`, carries out the conversions,
   * and returns a chain of converted formats for each format. Chain contains original format, all intermediate formats,
   * and last format in the chain.
   * @param cs ConversionSchema containing the an optional chain of conversion functions for every format in `packed_sfs`
   * @param packed_sfs formats to convert
   * @param clear_intermediate delete intermediate formats between the starting and destination one.
   * @return a vector of size `packed_sfs.size()`. Each element is a vector of size >= 1 containing the starting format
   * in the conversion chain, every intermediate format, and the final format. The first format is the original and the last
   * is the output format in each chain.
   */
  static std::vector<std::vector<format::Format *>>
  ApplyConversionSchema(const ConversionSchema & cs,
                        const std::vector<format::Format *> &packed_sfs, bool clear_intermediate);

   //! Takes a conversion chain and a format and applies that chain on the format
   //! to produce some output format.
   /*!
    * \param chain a conversion chain containing conversion functions and contexts to use for each function.
    * \param clear_intermediate delete intermediate formats between the starting and destination one.
    * \return a vector of format with the first being the original format, the last being the target format,
    * and the rest being intermediate formats. If a conversion is empty or false, only returns the original format.
    */
  static std::vector<format::Format*> ApplyConversionChain(const ConversionChain& chain,
                                              format::Format*, bool clear_intermediate);

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
  void ResetConverterOrderTwo();
  virtual void Reset();
};


//! An instance of this class can be used to convert between order one formats
//! (Array)
template <typename ValueType>
class ConverterOrderOne : public ConverterImpl<ConverterOrderOne<ValueType>> {
public:
  ConverterOrderOne();
  virtual Converter *Clone() const;
  void ResetConverterOrderOne();
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
