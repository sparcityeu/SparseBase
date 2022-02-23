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
class CsrCooMoveFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
  format::Format *operator()(format::Format *source);
};

template <typename IDType, typename NNZType, typename ValueType>
class CooCsrFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
  format::Format *operator()(format::Format *source);
};

template <typename IDType, typename NNZType, typename ValueType>
class CooCsrMoveFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
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
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         ConversionFunctor<IDType, NNZType, ValueType> *>>
      move_conversion_map_;

  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index,
                         ConversionFunctor<IDType, NNZType, ValueType> *>> *
  get_conversion_map(bool is_move_conversion);

public:
  Converter();
  ~Converter();
  void RegisterConversionFunction(
      std::type_index from_type, std::type_index to_type,
      ConversionFunctor<IDType, NNZType, ValueType> *conv_func,
      bool is_move_conversion = false);
  ConversionFunctor<IDType, NNZType, ValueType> *
  GetConversionFunction(std::type_index from_type, std::type_index to_type,
                        bool is_move_conversion = false);
  format::Format *Convert(format::Format *source, std::type_index to_type,
                          bool is_move_conversion = false);
  template <typename FormatType>
  FormatType *Convert(format::Format *source, bool is_move_conversion = false) {
    auto *res = this->Convert(source, FormatType::get_format_id_static());
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

#endif