#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "sparse_format.h"
#include <unordered_map>
#include <tuple>

using namespace sparsebase::format;

namespace sparsebase {

namespace utils {

typedef std::vector<std::tuple<bool, std::type_index>> ConversionSchema;

template <typename IDType, typename NNZType, typename ValueType>
class ConversionFunctor {
public:
  virtual Format<IDType, NNZType, ValueType> *
  operator()(Format<IDType, NNZType, ValueType> *source) {
    return nullptr;
  }
};

template <typename IDType, typename NNZType, typename ValueType>
class CsrCooFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
    Format<IDType, NNZType, ValueType> *
  operator()(Format<IDType, NNZType, ValueType> *source);
};

template <typename IDType, typename NNZType, typename ValueType>
class CooCsrFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
    Format<IDType, NNZType, ValueType> *
  operator()(Format<IDType, NNZType, ValueType> *source);
};

template <typename IDType, typename NNZType, typename ValueType> class SparseConverter {
private:
  std::unordered_map<
      std::type_index,
      std::unordered_map<std::type_index, ConversionFunctor<IDType, NNZType, ValueType> *>>
      conversion_map_;

public:
  SparseConverter();
  ~SparseConverter();
  void RegisterConversionFunction(
          std::type_index from_format, std::type_index to_format,
      ConversionFunctor<IDType, NNZType, ValueType> *conv_func);
  ConversionFunctor<IDType, NNZType, ValueType> *
  GetConversionFunction(std::type_index from_format, std::type_index to_format);
    Format<IDType, NNZType, ValueType> *
  Convert(Format<IDType, NNZType, ValueType> *source, std::type_index to_format);
  bool CanConvert(std::type_index from_format, std::type_index to_format);
  std::vector<Format<IDType, NNZType, ValueType> *> ApplyConversionSchema(
      ConversionSchema cs,
      std::vector<Format<IDType, NNZType, ValueType> *> packed_sfs);
};

} // namespace utils

} // namespace sparsebase

#endif