#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "sparse_format.h"
#include <unordered_map>
#include <tuple>

namespace sparsebase {

typedef std::vector<std::tuple<bool, Format>> ConversionSchema;
struct FormatHash {
  size_t operator()(Format f) const;
};

template <typename IDType, typename NNZType, typename ValueType>
class ConversionFunctor {
public:
  virtual SparseFormat<IDType, NNZType, ValueType> *
  operator()(SparseFormat<IDType, NNZType, ValueType> *source) {
    return nullptr;
  }
};

template <typename IDType, typename NNZType, typename ValueType>
class CsrCooFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
  SparseFormat<IDType, NNZType, ValueType> *
  operator()(SparseFormat<IDType, NNZType, ValueType> *source);
};

template <typename IDType, typename NNZType, typename ValueType>
class CooCsrFunctor : public ConversionFunctor<IDType, NNZType, ValueType> {
public:
  SparseFormat<IDType, NNZType, ValueType> *
  operator()(SparseFormat<IDType, NNZType, ValueType> *source);
};

template <typename IDType, typename NNZType, typename ValueType> class SparseConverter {
private:
  std::unordered_map<
      Format,
      std::unordered_map<Format, ConversionFunctor<IDType, NNZType, ValueType> *,
                         FormatHash>,
      FormatHash>
      conversion_map_;

public:
  SparseConverter();
  ~SparseConverter();
  void RegisterConversionFunction(
      Format from_format, Format to_format,
      ConversionFunctor<IDType, NNZType, ValueType> *conv_func);
  ConversionFunctor<IDType, NNZType, ValueType> *
  GetConversionFunction(Format from_format, Format to_format);
  SparseFormat<IDType, NNZType, ValueType> *
  Convert(SparseFormat<IDType, NNZType, ValueType> *source, Format to_format);
  bool CanConvert(Format from_format, Format to_format);
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> ApplyConversionSchema(
      ConversionSchema cs,
      std::vector<SparseFormat<IDType, NNZType, ValueType> *> packed_sfs);
};

} // namespace sparsebase

#endif