#include <iostream>

#include "sparsebase/converter/converter.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/object/object.h"

using namespace std;
using namespace sparsebase;
using namespace sparsebase::format;
using namespace sparsebase::object;

class MyFormat
    : public utils::IdentifiableImplementation<MyFormat,
                                               format::FormatImplementation> {
 public:
  MyFormat() {
    order_ = 2;
    dimension_ = {4, 4};
  }
  virtual Format *Clone() const override { return nullptr; }
};

Format *COOToMyFormat(Format *source, context::Context *) {
  return new MyFormat;
}

int main(int argc, char *argv[]) {
  // Custom Format creation and casting
  Format *format = new MyFormat;
  MyFormat *my_format = format->AsAbsolute<MyFormat>();

  // Using it in an Object
  Graph<int, int, int> *graph = new Graph<int, int, int>(my_format);

  // Custom conversion using the custom format
  converter::ConverterOrderTwo<int, int, int> converter;
  converter.RegisterConversionFunction(
      COO<int, int, int>::get_id_static(), MyFormat::get_id_static(),
      COOToMyFormat,
      [](context::Context *, context::Context *) { return true; });
  int row[6] = {0, 0, 1, 1, 2, 2};
  int col[6] = {0, 1, 1, 2, 3, 3};
  int vals[6] = {10, 20, 30, 40, 50, 60};
  context::CPUContext cpu_context;
  COO<int, int, int> *coo = new COO<int, int, int>(6, 6, 6, row, col, vals);
  MyFormat *my_format2 = converter.Convert<MyFormat>(coo, &cpu_context);
}
