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
#include "sparsebase/partition/metis_partition.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"
#ifdef USE_CUDA
#include "sparsebase/converter/converter_cuda.cuh"
#include "sparsebase/converter/converter_order_one_cuda.cuh"
#include "sparsebase/converter/converter_order_two_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#endif

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::partition;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
#ifdef USE_METIS
TEST(MetisPartition, BasicTest) {
  if (typeid(metis::idx_t) == typeid(int)) {
    MetisPartition<int, int, int> partitioner;
    MetisPartitionParams params;
    params.num_partitions = 2;
    auto part2 =
        partitioner.Partition(&global_coo, &params, {&cpu_context}, true);
    check_partition(part2, n, (int)2);
    params.num_partitions = 4;
    auto part4 =
        partitioner.Partition(&global_coo, &params, {&cpu_context}, true);
    check_partition(part4, n, (int)4);
  } else {
    MetisPartition<int64_t, int64_t, int64_t> partitioner;
    auto global_coo_64_bit =
        global_coo.Convert<sparsebase::format::COO, int64_t, int64_t, int64_t>(
            false);
    MetisPartitionParams params;
    params.num_partitions = 2;
    auto part2 =
        partitioner.Partition(global_coo_64_bit, &params, {&cpu_context}, true);
    check_partition(part2, (int64_t)n, (int64_t)2);
    params.num_partitions = 4;
    auto part4 =
        partitioner.Partition(global_coo_64_bit, &params, {&cpu_context}, true);
    check_partition(part4, (int64_t)n, (int64_t)4);
  }
}

#endif
