#include "gtest/gtest.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
const int n = 3;
const int nnz = 4;
int row_ptr[n + 1] = {0, 2, 3, 4};
int col_ptr[n + 1] = {0, 2, 3, 4};
int row_ind[nnz] = {1, 2, 0, 0};
int vals_csr[nnz] = {3, 4, 1, 2};
int cols[nnz] = {1, 2, 0, 0};
int rows[nnz] = {0, 0, 1, 2};
float distribution[n] = {2.0 / nnz, 1.0 / nnz, 1.0 / nnz};
int degrees[n] = {2, 1, 1};
int vals[nnz] = {1, 2, 3, 4};
/*
0 1 2
3 0 0
4 0 0
*/

float heatmap_no_order_true[3*3] = {0, 0.25, 0.25, 0.25, 0, 0, 0.25, 0, 0};

// row reordering
// r_reorder_vector[i] = j -> new row j used to be at location i
int r_reorder_vector[3] = {1, 2, 0};
int r_row_ptr[n + 1] = {0, 1, 3, 4};
int r_rows[nnz] = {0, 1, 1, 2};
int r_cols[nnz] = {0, 1, 2, 0};
int r_vals[nnz] = {4, 1, 2, 3};

// column reordering
int c_reorder_vector[3] = {2, 0, 1};
int c_row_ptr[n + 1] = {0, 2, 3, 4};
int c_rows[nnz] = {0, 0, 1, 2};
int c_cols[nnz] = {0, 1, 2, 2};
int c_vals[nnz] = {1, 2, 3, 4};

// row and column reordering
int rc_row_ptr[n + 1] = {0, 1, 3, 4};
int rc_rows[nnz] = {0, 1, 1, 2};
int rc_cols[nnz] = {2, 0, 1, 2};
int rc_vals[nnz] = {4, 1, 2, 3};
/*
0 0 4
1 2 0
0 0 3
*/
float heatmap_rc_order_true[3*3] = {0, 0, 0.25, 0.25, 0.25, 0, 0, 0, 0.25};

const int array_length = 3;
int inverse_perm_array[array_length] = {2, 0, 1};
int perm_array[array_length] = {1, 2, 0};
float original_array[array_length] = {0.0, 0.1, 0.2};
float reordered_array[array_length] = {0.1, 0.2, 0.0};
format::Array<float> orig_arr(n, original_array, format::kNotOwned);
format::Array<float> inv_arr(n, reordered_array, format::kNotOwned);
format::CSR<int, int, int> global_csr(n, n, row_ptr, cols, vals,
                                      format::kNotOwned);
format::CSC<int, int, int> global_csc(n, n, col_ptr, row_ind, vals_csr,
                                      format::kNotOwned);
format::COO<int, int, int> global_coo(n, n, nnz, rows, cols, vals,
                                      format::kNotOwned);
sparsebase::context::CPUContext cpu_context;

template <typename IDType>
void check_degree_ordering(IDType *order, IDType n, IDType *row_ptr,
                           bool ascending = true) {
  auto *permutation = new IDType[n];
  for (IDType i = 0; i < n; i++) {
    permutation[order[i]] = i;
  }
  bool order_is_correct = true;
  std::set<IDType> check;
  for (IDType new_u = 0; new_u < n - 1 && order_is_correct; new_u++) {
    IDType u = permutation[new_u];
    EXPECT_EQ(check.find(u), check.end());
    check.insert(u);
    IDType v = permutation[new_u + 1];
    if (ascending)
      EXPECT_LE(row_ptr[u + 1] - row_ptr[u], row_ptr[v + 1] - row_ptr[v]);
    else
      EXPECT_GE(row_ptr[u + 1] - row_ptr[u], row_ptr[v + 1] - row_ptr[v]);
  }
  IDType v = permutation[n - 1];
  EXPECT_EQ(check.find(v), check.end());
  check.insert(v);
  delete[] permutation;
}
template <typename IDType>
void check_reorder(IDType *order, IDType n) {
  std::set<IDType> vertices;
  for (IDType i = 0; i < n; i++) {
    EXPECT_EQ(vertices.find(order[i]), vertices.end());
    vertices.insert(order[i]);
  }
}

template <typename IDType>
void check_partition(IDType* part, IDType n, IDType np){
  for (IDType i = 0; i < n; i++) {
    EXPECT_LT(part[i], np);
    EXPECT_GE(part[i], 0);
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void compare_csr(format::CSR<IDType, NNZType, ValueType> *correct,
                 format::CSR<IDType, NNZType, ValueType> *testing) {
  auto correct_row_ptr = correct->get_row_ptr();
  auto correct_col = correct->get_col();
  auto testing_row_ptr = testing->get_row_ptr();
  auto testing_col = testing->get_col();

  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(correct_row_ptr[i], testing_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(correct_col[i], testing_col[i]);
  }
}
template <typename IDType, typename NNZType, typename ValueType>
void compare_csc(format::CSC<IDType, NNZType, ValueType> *correct,
                 format::CSC<IDType, NNZType, ValueType> *testing) {
  auto correct_col_ptr = correct->get_col_ptr();
  auto correct_row = correct->get_row();
  auto testing_col_ptr = testing->get_col_ptr();
  auto testing_row = testing->get_row();

  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(correct_col_ptr[i], testing_col_ptr[i]);
  }
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(correct_row[i], testing_row[i]);
  }
}
template <typename V, typename E, typename O, typename L>
void confirm_renumbered_csr(V *xadj, V *renumbered_xadj, E *adj,
                            E *renumbered_adj, O *inverse_order, L n) {
  auto order = new E[n];
  for (L i = 0; i < n; i++) {
    order[inverse_order[i]] = i;
  }
  for (L i = 0; i < n; i++) {
    EXPECT_EQ(xadj[i + 1] - xadj[i], renumbered_xadj[inverse_order[i] + 1] -
                                         renumbered_xadj[inverse_order[i]]);
    std::set<V> edges;
    for (E edge = xadj[i]; edge < xadj[i + 1]; edge++) {
      edges.insert(inverse_order[adj[edge]]);
    }
    for (E edge = renumbered_xadj[inverse_order[i]];
         edge < renumbered_xadj[inverse_order[i] + 1]; edge++) {
      EXPECT_NE(edges.find(renumbered_adj[edge]), edges.end());
    }
  }
}
void CompareVectorsOfTypeIndex(std::vector<std::type_index> i1,
                               std::vector<std::type_index> i2) {
  ASSERT_EQ(i1.size(), i2.size());
  std::sort(i1.begin(), i1.end());
  std::sort(i2.begin(), i2.end());
  for (int i = 0; i < i1.size(); i++) {
    EXPECT_EQ(i1[i], i2[i]);
  }
}
#define EXECUTE_AND_DELETE(call) \
  [&]() {                        \
    auto output = (call);        \
    delete[] output;             \
  }();

template <typename a, typename b, typename c>
class TestFormat
        : public sparsebase::utils::IdentifiableImplementation<
                TestFormat<a, b, c>, sparsebase::format::FormatOrderTwo<a, b, c>> {
    format::Format *Clone() const override { return nullptr; }
};

template <typename a>
class TestFormatOrderOne
        : public sparsebase::utils::IdentifiableImplementation<
                TestFormatOrderOne<a>, sparsebase::format::FormatOrderOne<a>> {
    format::Format *Clone() const override { return nullptr; }
};

