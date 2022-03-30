#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "sparse_format.h"
#include "config.h"
#include <tuple>
#include <unordered_map>
#include <functional>
#ifdef CUDA
#include "cuda/format.cuh"
#include "cuda/converter.cuh"
#endif

namespace sparsebase {

namespace utils {

typedef std::vector<std::tuple<bool, std::type_index, context::Context*>> ConversionSchemaConditional;

using ConditionalConversionFunction = std::function<format::Format*(format::Format*, context::Context*)>;
using EdgeConditional = std::function<bool(context::Context*, context::Context*)>;
class Converter {
private:
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         std::vector<std::tuple<EdgeConditional, ConditionalConversionFunction>>>>
      conditional_map_;
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         std::vector<std::tuple<EdgeConditional, ConditionalConversionFunction>>>>
      conditional_move_map_;
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         std::vector<std::tuple<EdgeConditional, ConditionalConversionFunction>>>> *get_conversion_map(bool is_move_conversion);

public:
  void RegisterConditionalConversionFunction(
      std::type_index from_type, 
      std::type_index to_type,
      ConditionalConversionFunction conv_func,
      EdgeConditional edge_condition,
      bool is_move_conversion = false);
  ConditionalConversionFunction
  GetConversionFunction(std::type_index from_type, context::Context* from_context, 
                        std::type_index to_type, context::Context* to_context, 
                        bool is_move_conversion = false);
  format::Format *Convert(format::Format *source, std::type_index to_type, context::Context* to_context,
                          bool is_move_conversion = false);
  template <typename FormatType>
  FormatType *Convert(format::Format *source, context::Context* to_context, bool is_move_conversion = false) {
    auto *res = this->Convert(source, FormatType::get_format_id_static(), to_context, is_move_conversion);
    return res->template As<FormatType>();
  }
  std::tuple<bool, context::Context*> CanConvert(std::type_index from_type, context::Context* from_context, std::type_index to_type,
                  std::vector<context::Context*> to_contexts,
                  bool is_move_conversion = false);
  bool CanConvert(std::type_index from_type, context::Context* from_context, std::type_index to_type, context::Context* to_context,
                  bool is_move_conversion = false);
  std::vector<format::Format *>
  ApplyConversionSchema(ConversionSchemaConditional cs,
                        std::vector<format::Format *> packed_sfs,
                        bool is_move_conversion = false);
  virtual std::type_index get_converter_type() = 0;
  virtual utils::Converter* Clone() const = 0;
  virtual void Reset() = 0;
  virtual ~Converter();
};

template <class ConverterType>
class ConverterImpl : public Converter{
public:
  virtual std::type_index get_converter_type(){
    return typeid(ConverterType);
  }
};

template <typename IDType, typename NNZType, typename ValueType>
class OrderTwoConverter : public ConverterImpl<OrderTwoConverter<IDType, NNZType, ValueType>>{
public:
  OrderTwoConverter();
  virtual utils::Converter* Clone() const;
  virtual void Reset();
};

template <typename ValueType>
class OrderOneConverter : public ConverterImpl<OrderOneConverter<ValueType>>{
public:
  OrderOneConverter();
  virtual utils::Converter* Clone() const;
  virtual void Reset();
};

} // namespace utils

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparse_converter.cc"
#ifdef CUDA
#include "cuda/converter.cu"
#endif
#endif

#endif
