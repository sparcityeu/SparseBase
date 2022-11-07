#ifndef SPARSEBASE_PROJECT_CONVERTER_ORDER_TWO_H
#define SPARSEBASE_PROJECT_CONVERTER_ORDER_TWO_H

#include "sparsebase/config.h"
#include "sparsebase/converter/converter.h"

namespace sparsebase::converter {

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
}
#ifdef _HEADER_ONLY
#include "converter_order_two.cc"
#endif
#endif  // SPARSEBASE_PROJECT_CONVERTER_ORDER_TWO_H
