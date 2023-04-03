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

//PATOH HyperGraph Reader

int hypergraph1_n = 11;
int hypergraph1_m = 31;
int hypergraph1_vertex_size = 12;
int hypergraph1_xpins[12]{0,5,7,11,13,15,19,21,25,27,29,31};
int hypergraph1_pins[31]{2, 3, 5, 6, 9, 0, 1, 0, 1, 2, 3, 1, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 10, 11, 8, 10, 8, 11, 2, 5};
int hypergraph1_netWeights[31]{1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, 1, 1};

const std::string hypergraph_1 = R"(% base:(0/1) #cells #nets #pins
0 12 11 31 
% pins of each net in the hypergraph
2 3 5 6 9
0 1
0 1 2 3
1 3
4 5
4 5 6 7
6 7
8 9 10 11
8 10
8 11
2 5)";
