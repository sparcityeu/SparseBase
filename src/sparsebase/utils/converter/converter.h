#ifndef SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_

#include "sparsebase/config.h"
#include "sparsebase/format/format.h"
#include <functional>
#include <tuple>
#include <unordered_map>
#ifdef CUDA
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/utils/converter/cuda/converter.cuh"
#endif

namespace sparsebase {

namespace utils {

namespace converter {

typedef std::vector<std::tuple<bool, std::type_index, context::Context *>>
    ConversionSchemaConditional;

using ConditionalConversionFunction =
    std::function<format::Format *(format::Format *, context::Context *)>;
using EdgeConditional =
    std::function<bool(context::Context *, context::Context *)>;
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
  ConditionalConversionFunction
  GetConversionFunction(std::type_index from_type,
                        context::Context *from_context, std::type_index to_type,
                        context::Context *to_context,
                        bool is_move_conversion = false);
  format::Format *Convert(format::Format *source, std::type_index to_type,
                          context::Context *to_context,
                          bool is_move_conversion = false);
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
             std::vector<context::Context *> to_contexts,
             bool is_move_conversion = false);
  bool CanConvert(std::type_index from_type, context::Context *from_context,
                  std::type_index to_type, context::Context *to_context,
                  bool is_move_conversion = false);
  std::vector<format::Format *>
  ApplyConversionSchema(ConversionSchemaConditional cs,
                        std::vector<format::Format *> packed_sfs,
                        bool is_move_conversion = false);
  virtual std::type_index get_converter_type() = 0;
  virtual Converter *Clone() const = 0;
  virtual void Reset() = 0;
  virtual ~Converter();
};

template <class ConverterType> class ConverterImpl : public Converter {
public:
  virtual std::type_index get_converter_type() { return typeid(ConverterType); }
};

template <typename IDType, typename NNZType, typename ValueType>
class ConverterOrderTwo
    : public ConverterImpl<ConverterOrderTwo<IDType, NNZType, ValueType>> {
public:
  ConverterOrderTwo();
  virtual Converter *Clone() const;
  virtual void Reset();
};

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
#ifdef CUDA
#include "cuda/converter.cu"
#endif
#endif

#endif // SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_
