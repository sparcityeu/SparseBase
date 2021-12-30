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
      ConversionMap;

public:
  SparseConverter();
  ~SparseConverter();
  void register_conversion_function(
      Format from_format, Format to_format,
      ConversionFunctor<ID, NumNonZeros, Value> *conv_func);
  ConversionFunctor<ID, NumNonZeros, Value> *
  get_conversion_function(Format from_format, Format to_format);
  SparseFormat<ID, NumNonZeros, Value> *
  convert(SparseFormat<ID, NumNonZeros, Value> *source, Format to_format);
  bool can_convert(Format from_format, Format to_format);
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> apply_conversion_schema(
      ConversionSchema cs,
      std::vector<SparseFormat<ID, NumNonZeros, Value> *> packed_sfs);
};

} // namespace sparsebase

#endif