#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "sparse_format.h"
#include <tuple>
#include <unordered_map>


namespace sparsebase {

namespace utils {

typedef std::vector<std::tuple<bool, std::type_index>> ConversionSchema;

template <typename IDType, typename NNZType, typename ValueType>
class ConversionFunctor {
public:
  virtual format::Format *operator()(format::Format *source) { return nullptr; }
};

template <typename IDType, typename NNZType, typename ValueType>
class CsrCooFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
  format::Format *operator()(format::Format *source);
};

template <typename IDType, typename NNZType, typename ValueType>
class CooCsrFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
  format::Format *operator()(format::Format *source);
};

template <typename IDType, typename NNZType, typename ValueType>
class Converter {
private:
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         ConversionFunctor<IDType, NNZType, ValueType> *>>
      conversion_map_;

public:
  Converter();
  ~Converter();
  void RegisterConversionFunction(
      std::type_index from_type, std::type_index to_type,
      ConversionFunctor<IDType, NNZType, ValueType> *conv_func);
  ConversionFunctor<IDType, NNZType, ValueType> *
  GetConversionFunction(std::type_index from_type, std::type_index to_type);
  format::Format *Convert(format::Format *source, std::type_index to_type);
  template <typename FormatType> FormatType *Convert(format::Format *source) {
    auto *res = this->Convert(source, FormatType::get_format_id_static());
    return res->template As<FormatType>();
  }
  bool CanConvert(std::type_index from_type, std::type_index to_type);
  std::vector<format::Format *> ApplyConversionSchema(ConversionSchema cs,
                                              std::vector<format::Format *> packed_sfs);
};

} // namespace utils

} // namespace sparsebase

#endif