#include "sparsebase/format/format_order_two.h"

#include "sparsebase/converter/converter_order_two.h"
#include "sparsebase/converter/converter_store.h"
namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
FormatOrderTwo<IDType, NNZType, ValueType>::FormatOrderTwo() {
  this->set_converter(
      converter::ConverterStore::GetStore()
          .get_converter<
              converter::ConverterOrderTwo<IDType, NNZType, ValueType>>());
}

#ifndef _HEADER_ONLY
#include "init/format_order_two.inc"
#endif
}  // namespace sparsebase::format
