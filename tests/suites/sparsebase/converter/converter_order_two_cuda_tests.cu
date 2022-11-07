#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
using namespace sparsebase;
using namespace converter;
using namespace format;
using namespace context;
#include "common.inc"
#include "common_cuda.inc"
TEST_F(CUDAFormatsFixture, CUDACSR) {
ConversionChain chain;
format::Format* output_format;
// Can't convert for bad context
chain = c2.GetConversionChain(csr->get_id(), csr->get_context(),
                              CUDACSR<TYPE>::get_id_static(), {&cpu});
EXPECT_EQ(chain.has_value(), false);
// Can convert
chain = c2.GetConversionChain(csr->get_id(), csr->get_context(),
                              CUDACSR<TYPE>::get_id_static(), {&gpu0});
EXPECT_EQ(chain.has_value(), true);
EXPECT_EQ((std::get<0>(*chain)).size(), 1);
EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_id()),
CUDAContext::get_id_static());
output_format = (std::get<0>(std::get<0>(*chain)[0]))(csr, &gpu0);
EXPECT_EQ(output_format->get_id(),
(CUDACSR<TYPE>::get_id_static()));
compare_cuda_cpu_csr(output_format->AsAbsolute<CUDACSR<TYPE>>(), csr);
// Can't convert to CSR bad context
chain = c2.GetConversionChain(CUDACSR<TYPE>::get_id_static(),
                              output_format->get_context(),
                              CSR<TYPE>::get_id_static(), {&gpu0});
EXPECT_EQ(chain.has_value(), false);
// Can convert to CSR
chain = c2.GetConversionChain(CUDACSR<TYPE>::get_id_static(),
                              output_format->get_context(),
                              CSR<TYPE>::get_id_static(), {&cpu});
EXPECT_EQ(chain.has_value(), true);
EXPECT_EQ((std::get<0>(*chain)).size(), 1);
EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_id()),
CPUContext::get_id_static());
format::Format* output_format_csr;
output_format_csr =
(std::get<0>(std::get<0>(*chain)[0]))(output_format, &cpu);
EXPECT_EQ(output_format_csr->get_id(),
(CSR<TYPE>::get_id_static()));
compare_cuda_cpu_csr(output_format->AsAbsolute<CUDACSR<TYPE>>(),
output_format_csr->AsAbsolute<CSR<TYPE>>());
delete output_format;
delete output_format_csr;
}
