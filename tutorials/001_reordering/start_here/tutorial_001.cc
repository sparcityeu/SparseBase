#include "sparsebase/format/csr.h"
#include "sparsebase/bases/iobase.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/reorder/rcm_reorder.h"
#include "sparsebase/context/cpu_context.h"
#include <string>
#include <iostream>

typedef unsigned int id_type;
typedef unsigned int nnz_type;
typedef void value_type;

using namespace sparsebase;
using namespace io;
using namespace bases;
using namespace reorder;
using namespace format;

int main(int argc, char * argv[]){
  if (argc < 2){
    std::cout << "Please enter the name of the edgelist file as a parameter\n";
    return 1;
  }


  ///// YOUR CODE GOES HERE /////

  //////////////////////////////

  return 0;
}
