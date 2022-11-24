#include <iostream>
#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
#include "sparsebase/format/cuda_array_cuda.cuh"

using namespace sparsebase;
using namespace converter;
using namespace format;
using namespace context;

#include "common.inc"
#include "common_cuda.inc"

TEST_F(CUDAFormatsFixture, CUDAArray) {
ConversionChain chain;
format::Format* output_format;
// Can't convert for bad context
chain = c1.GetConversionChain(array->get_id(), array->get_context(),
                              CUDAArray<int>::get_id_static(), {&cpu});
EXPECT_EQ(chain.has_value(), false);
// Can convert
chain =
c1.GetConversionChain(array->get_id(), array->get_context(),
                      CUDAArray<int>::get_id_static(), {&gpu0});
ASSERT_EQ(chain.has_value(), true);
EXPECT_EQ((std::get<0>(*chain)).size(), 1);
EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_id()),
CUDAContext::get_id_static());
output_format = (std::get<0>(std::get<0>(*chain)[0]))(array, &gpu0);
EXPECT_EQ(output_format->get_id(),
(CUDAArray<int>::get_id_static()));
compare_arrays_cuda_cpu(
    output_format->AsAbsolute<CUDAArray<int>>()->get_vals(),
    array->get_vals(), array->get_num_nnz());
// Can't convert to Array bad context
chain = c1.GetConversionChain(CUDAArray<int>::get_id_static(),
                              output_format->get_context(),
                              Array<int>::get_id_static(), {&gpu0});
EXPECT_EQ(chain.has_value(), false);
// Can convert to Array
chain = c1.GetConversionChain(CUDAArray<int>::get_id_static(),
                              output_format->get_context(),
                              Array<int>::get_id_static(), {&cpu});
EXPECT_EQ(chain.has_value(), true);
EXPECT_EQ((std::get<0>(*chain)).size(), 1);
EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_id()),
CPUContext::get_id_static());
format::Format* output_format_csr;
output_format_csr =
(std::get<0>(std::get<0>(*chain)[0]))(output_format, &cpu);
EXPECT_EQ(output_format_csr->get_id(),
(Array<int>::get_id_static()));
compare_arrays_cuda_cpu(
    output_format->AsAbsolute<CUDAArray<int>>()->get_vals(),
    output_format_csr->AsAbsolute<Array<int>>()->get_vals(),
    output_format->get_num_nnz());
delete output_format;
delete output_format_csr;
}

TEST_F(CUDAFormatsFixture, CUDAArrayCached) {
ConversionChain chain;
std::vector<format::Format*> output_formats;
// Can't convert for bad context
EXPECT_THROW((c1.ConvertCached(array, CUDAArray<int>::get_id_static(),
    &cpu, false)),
utils::ConversionException);

output_formats = c1.ConvertCached(
    array, CUDAArray<int>::get_id_static(), &gpu0, false);
EXPECT_EQ(output_formats.size(), 1);
compare_arrays_cuda_cpu(
    output_formats[0]->AsAbsolute<CUDAArray<int>>()->get_vals(),
    array->get_vals(), array->get_num_nnz());
delete output_formats[0];

output_formats = c1.ConvertCached(
    array, CUDAArray<int>::get_id_static(), {&gpu0, &cpu}, false);
EXPECT_EQ(output_formats.size(), 1);
compare_arrays_cuda_cpu(
    output_formats[0]->AsAbsolute<CUDAArray<int>>()->get_vals(),
    array->get_vals(), array->get_num_nnz());
}

TEST_F(CUDAFormatsFixture, CUDAArrayMultiContext) {
ConversionChain chain;
format::Format* output_format;
// Can't convert for bad context
EXPECT_THROW(
(c1.Convert(array, CUDAArray<int>::get_id_static(), &cpu, false)),
utils::ConversionException);

auto cuda_array = c1.Convert<CUDAArray<int>>(array, {&gpu0, &cpu}, false);
compare_arrays_cuda_cpu(cuda_array->get_vals(), array->get_vals(),
    array->get_num_nnz());
delete cuda_array;

output_format = c1.Convert(array, CUDAArray<int>::get_id_static(),
                           {&gpu0, &cpu}, false);
compare_arrays_cuda_cpu(
    output_format->AsAbsolute<CUDAArray<int>>()->get_vals(),
    array->get_vals(), array->get_num_nnz());
delete output_format;
cuda_array = array->Convert<CUDAArray>({&gpu0, &cpu}, false);
compare_arrays_cuda_cpu(cuda_array->get_vals(), array->get_vals(),
    array->get_num_nnz());
delete cuda_array;
}
