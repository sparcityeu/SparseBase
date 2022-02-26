#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "sparse_format.h"
#include "config.h"
#include <tuple>
#include <unordered_map>
#include <functional>

namespace sparsebase {

namespace utils {

typedef std::vector<std::tuple<bool, std::type_index>> ConversionSchema;

using ConversionFunction = std::function<format::Format*(format::Format*)>;

template <typename IDType, typename NNZType, typename ValueType>
class Converter {
private:
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         ConversionFunction>>
      conversion_map_;
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         ConversionFunction>>
      move_conversion_map_;

  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         ConversionFunction>> *
  get_conversion_map(bool is_move_conversion);

public:
  Converter();
  ~Converter();
  void RegisterConversionFunction(
      std::type_index from_type, std::type_index to_type,
      ConversionFunction conv_func,
      bool is_move_conversion = false);
  ConversionFunction
  GetConversionFunction(std::type_index from_type, std::type_index to_type,
                        bool is_move_conversion = false);
  format::Format *Convert(format::Format *source, std::type_index to_type,
                          bool is_move_conversion = false);
  template <typename FormatType>
  FormatType *Convert(format::Format *source, bool is_move_conversion = false) {
    auto *res = this->Convert(source, FormatType::get_format_id_static(), is_move_conversion);
    return res->template As<FormatType>();
  }
  bool CanConvert(std::type_index from_type, std::type_index to_type,
                  bool is_move_conversion = false);
  std::vector<format::Format *>
  ApplyConversionSchema(ConversionSchema cs,
                        std::vector<format::Format *> packed_sfs,
                        bool is_move_conversion = false);
};

} // namespace utils

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparse_converter.cc"
#endif

#endif
