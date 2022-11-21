#include "sparsebase/format/format_order_one.h"
#include "sparsebase/converter/converter_store.h"
#include "sparsebase/converter/converter_order_one.h"
namespace sparsebase::format {

template <typename ValueType>
FormatOrderOne<ValueType>::FormatOrderOne(){
  this->set_converter(converter::ConverterStore::GetStore().get_converter<converter::ConverterOrderOne<ValueType>>());
}
#ifndef _HEADER_ONLY
#include "init/format_order_one.inc"
#endif
}
