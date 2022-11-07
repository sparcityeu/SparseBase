#ifndef SPARSEBASE_PROJECT_CONVERTER_ORDER_ONE_H
#define SPARSEBASE_PROJECT_CONVERTER_ORDER_ONE_H
#include "converter.h"
#include "sparsebase/config.h"
namespace sparsebase::converter{

//! An instance of this class can be used to convert between order one formats
//! (Array)
template <typename ValueType>
class ConverterOrderOne : public ConverterImpl<ConverterOrderOne<ValueType>> {
 public:
  ConverterOrderOne();
  virtual Converter *Clone() const;
  void ResetConverterOrderOne();
  virtual void Reset();
};

}
#ifdef _HEADER_ONLY
#include "converter_order_one.cc"
#endif
#endif  // SPARSEBASE_PROJECT_CONVERTER_ORDER_ONE_H
