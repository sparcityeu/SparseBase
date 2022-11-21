#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
#include "common.inc"

using namespace sparsebase;


TEST(Format, Is) {
  int *new_csr_row_ptr = new int[5];
  int *new_csr_col = new int[4];
  int *new_csr_vals = new int[4];
  std::copy(csr_row_ptr, csr_row_ptr + 5, new_csr_row_ptr);
  std::copy(csr_col, csr_col + 4, new_csr_col);
  std::copy(csr_vals, csr_vals + 4, new_csr_vals);

  // Construct an owned CSR
  auto *csr = new format::CSR<int, int, int>(
      4, 4, new_csr_row_ptr, new_csr_col, new_csr_vals,
      format::kOwned);

  bool res = csr->Is<format::CSR<int, int, int>>();
  EXPECT_TRUE(res);
  res = csr->Is<format::CSR<int, int, int> *>();
  EXPECT_TRUE(res);

  res = csr->Is<format::COO<int, int, int>>();
  EXPECT_FALSE(res);
  res = csr->Is<format::CSR<int, int, float>>();
  EXPECT_FALSE(res);

  delete csr;
}
class TestFormat : utils::IdentifiableImplementation<TestFormat, format::FormatImplementation> {
 public:
  TestFormat() {
    this->context_ = std::unique_ptr<context::Context>(new context::CPUContext);
  }
  void ResetContext() {
    this->context_ = std::unique_ptr<context::Context>(new context::CPUContext);
    utils::OnceSettable<std::unique_ptr<context::Context>> xx;
  }
  Format *Clone() const { return nullptr; }
};

TEST(Context, reset) {
  TestFormat tf;
  EXPECT_THROW(tf.ResetContext(),
               utils::AttemptToReset<std::unique_ptr<context::Context>>);
}