#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "sparse_format.hpp"
#include <unordered_map>

namespace sparsebase {

typedef std::vector<std::tuple<bool, Format>> ConversionSchema;
struct FormatHash {
  size_t operator()(Format f) const;
};

template <typename ID, typename NumNonZeros, typename Value>
class ConversionFunctor {
public:
  virtual SparseFormat<ID, NumNonZeros, Value> *
  operator()(SparseFormat<ID, NumNonZeros, Value> *source) {
    return nullptr;
  }
};

template <typename ID, typename NumNonZeros, typename Value>
class CsrCooFunctor : public ConversionFunctor<ID, NumNonZeros, Value> {
public:
  SparseFormat<ID, NumNonZeros, Value> *
  operator()(SparseFormat<ID, NumNonZeros, Value> *source);
};

template <typename ID, typename NumNonZeros, typename Value>
class CooCsrFunctor : public ConversionFunctor<ID, NumNonZeros, Value> {
public:
  SparseFormat<ID, NumNonZeros, Value> *
  operator()(SparseFormat<ID, NumNonZeros, Value> *source);
};

template <typename ID, typename NumNonZeros, typename Value> class SparseConverter {
private:
  std::unordered_map<
      Format,
      std::unordered_map<Format, ConversionFunctor<ID, NumNonZeros, Value> *,
                         FormatHash>,
      FormatHash>
      conversion_map_;

public:
  SparseConverter();
  ~SparseConverter();
  void RegisterConversionFunction(
      Format from_format, Format to_format,
      ConversionFunctor<ID, NumNonZeros, Value> *conv_func);
  ConversionFunctor<ID, NumNonZeros, Value> *
  GetConversionFunction(Format from_format, Format to_format);
  SparseFormat<ID, NumNonZeros, Value> *
  Convert(SparseFormat<ID, NumNonZeros, Value> *source, Format to_format);
  bool CanConvert(Format from_format, Format to_format);
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> ApplyConversionSchema(
      ConversionSchema cs,
      std::vector<SparseFormat<ID, NumNonZeros, Value> *> packed_sfs);
};

} // namespace sparsebase

#endif