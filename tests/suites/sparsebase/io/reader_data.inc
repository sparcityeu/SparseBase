// Both files should result in the COO arrays below
// (With the convert_to_zero_index option set to true for mtx)
int row_ptr[6]{0, 0, 1, 2, 3, 5};
int row[5]{1, 2, 3, 4, 4};
int col[5]{0, 1, 0, 2, 3};
int one_row_one_col[5]{1, 3, 4, 5, 8};
float one_row_one_col_vals[10]{0, 0.1, 0, 0.3, 0.2, 0.4, 0, 0, 0.5, 0};
int one_row_one_col_length = 10;
float vals[5]{0.1, 0.3, 0.2, 0.4, 0.5};
int row_ptr_symm[6]{0, 3, 5, 7, 9, 11};
int row_symm[11]{0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
int col_symm[11]{0, 1, 3, 0, 2, 1, 4, 0, 4, 2, 3};
float vals_symm[11]{0.7, 0.1, 0.2, 0.1, 0.3, 0.3, 0.4, 0.2, 0.5, 0.4, 0.5};
const std::string mtx_data =
    R"(%%MatrixMarket matrix coordinate pattern general
%This is a comment
5 5 5
2 1
4 1
3 2
5 3
5 4
)";

const std::string mtx_symm_data =
    R"(%%MatrixMarket matrix coordinate pattern symmetric
%This is a comment
5 5 6
1 1
2 1
4 1
3 2
5 3
5 4
)";

const std::string mtx_data_with_values =
    R"(%%MatrixMarket matrix coordinate real general
%This is a comment
5 5 5
2 1 0.1
4 1 0.2
3 2 0.3
5 3 0.4
5 4 0.5
)";

const std::string mtx_data_with_values_array =
    R"(%%MatrixMarket matrix array real general
%This is a comment
5 5
0
0
0
0
0
0.1
0
0
0
0
0
0.3
0
0
0
0.2
0
0
0
0
0
0
0.4
0.5
0
)";

const std::string mtx_data_array =
    R"(%%MatrixMarket matrix array real general
%This is a comment
5 5
0
0
0
0
0
0.1
0
0
0
0
0
0.3
0
0
0
0.2
0
0
0
0
0
0
0.4
0.5
0
)";

const std::string mtx_symm_data_with_values =
    R"(%%MatrixMarket matrix coordinate real symmetric
%This is a comment
5 5 6
1 1 0.7
2 1 0.1
4 1 0.2
3 2 0.3
5 3 0.4
5 4 0.5
)";

const std::string mtx_data_one_col_with_values =
    R"(%%MatrixMarket matrix coordinate real general
%This is a comment
10 1 5
2 1 0.1
4 1 0.3
5 1 0.2
6 1 0.4
9 1 0.5
)";

const std::string mtx_data_one_row_with_values =
    R"(%%MatrixMarket matrix coordinate real general
%This is a comment
1 10 5
1 2 0.1
1 4 0.3
1 5 0.2
1 6 0.4
1 9 0.5
)";

const std::string mtx_array_data_one_col_with_values =
    R"(%%MatrixMarket matrix array real general
%This is a comment
10 1
0
0.1
0
0.3
0.2
0.4
0
0
0.5
0
)";

const std::string mtx_array_data_one_row_with_values =
    R"(%%MatrixMarket matrix array real general
%This is a comment
1 10
0
0.1
0
0.3
0.2
0.4
0
0
0.5
0
)";
const std::string edge_list_data = R"(1 0
3 0
2 1
4 2
4 3
)";

const std::string edge_list_data_with_values = R"(1 0 0.1
3 0 0.2
2 1 0.3
4 2 0.4
4 3 0.5
)";

//Metis Graph Reader

//Data_1: https://people.sc.fsu.edu/~jburkardt/data/metis_graph/tiny_03.graph
int metis_n_1 = 7;
int metis_m_1 = 22;
int metis_ncon_1 = 1;
int metis_row_1[22]{1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7};
int metis_col_1[22]{2, 3, 5, 1, 3, 4, 1, 2, 4, 5, 2, 3, 6, 7, 1, 3, 6, 4, 5, 7, 4, 6};
int metis_val_1[22]{1, 2, 1, 1, 2, 1, 2, 2, 2, 3, 1, 2, 2, 5, 1, 3, 2, 2, 2, 6, 5, 6};
int metis_vertex_weights_1[8]{0, 4, 2, 5, 3, 1, 6, 2};

const std::string metis_graph_1 = R"( 7 11 11
4    5 1   3 2   2 1
2    1 1   3 2   4 1
5    5 3   4 2   2 2   1 2
3    2 1   3 2   6 2   7 5
1    1 1   3 3   6 2
6    5 2   4 2   7 6
2    6 6   4 5
)";

//Data_2: https://people.sc.fsu.edu/~jburkardt/data/metis_graph/tiny_04.graph
int metis_n_2 = 7;
int metis_m_2 = 22;
int metis_ncon_2 = 3;
int metis_row_2[22]{1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7};
int metis_col_2[22]{2, 3, 5, 1, 3, 4, 1, 2, 4, 5, 2, 3, 6, 7, 1, 3, 6, 4, 5, 7, 4, 6};
int metis_vertex_weights_2[8][3]{{0, 0, 0}, {1, 2, 0}, {0, 2, 2}, {4, 1, 1}, {2, 2, 3}, {1, 1, 1}, {2, 2, 1}, {1, 2, 1}};
const std::string metis_graph_2 = R"( 7 11 10  3
1 2 0    5  3  2
0 2 2    1  3  4
4 1 1    5  4  2  1
2 2 3    2  3  6  7
1 1 1    1  3  6
2 2 1    5  4  7
1 2 1    6  4
)";

