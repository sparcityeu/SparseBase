#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/iobase.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/partition/patoh_partition.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"
#ifdef USE_CUDA
#include "sparsebase/converter/converter_cuda.cuh"
#include "sparsebase/converter/converter_order_one_cuda.cuh"
#include "sparsebase/converter/converter_order_two_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"
#endif

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";


using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::partition;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
#ifdef USE_PATOH
TEST(PatohPartition, BasicTest) {
  //std::cout << "Hello" << std::endl;
  PatohPartition<int, int, void> partitioner;
  // This is a temporary solution intended to be replaced by the Downloaders once finished
  auto coo = io::IOBase::ReadMTXToCOO<int,int,void>(FILE_NAME);
  PatohPartitionParams params;
  params.num_partitions = 2;
  auto part2 = partitioner.Partition(coo, &params, {&cpu_context}, true);
  check_partition(part2, n, 2);
  params.num_partitions = 4;
  auto part4 = partitioner.Partition(coo, &params, {&cpu_context}, true);
  check_partition(part4, n, 4);
  delete coo;
}
#endif
