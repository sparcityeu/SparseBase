/*
    For three symmetric cases (asymmetric, symmetric, skew-
    symmetric), we list:
    1. The raw pointers used to construct that matrix into
    COO/CSR
    2. The output from using these raw arrays in Scipy
*/
////////// Asymmetric sparse matrix
//// Raw pointers
int row_ptr[6]{0, 0, 1, 2, 3, 5};
int row[5]{1, 2, 3, 4, 4};
int col[5]{0, 1, 0, 2, 3};
float vals[5]{0.1, 0.3, 0.2, 0.4, 0.5};
//// Scipy output
/* COO
``` python
print(coo_array((vals, (row, col)), shape=(5,5)).toarray())
Output: [[0.  0.  0.  0.  0. ]
        [0.1 0.  0.  0.  0. ]
        [0.  0.3 0.  0.  0. ]
        [0.2 0.  0.  0.  0. ]
        [0.  0.  0.4 0.5 0. ]]
*/
/* CSR
>>> print(csr_array((vals, col, row_ptr), shape=(5,5)).toarray())

Output: [[0.  0.  0.  0.  0. ]
         [0.1 0.  0.  0.  0. ]
         [0.  0.3 0.  0.  0. ]
         [0.2 0.  0.  0.  0. ]
         [0.  0.  0.4 0.5 0. ]]
*/

////////// Symmetric sparse matrix
//// Raw pointers
int row_ptr_symm[6]{0, 3, 5, 7, 9, 11};
int row_symm[11]{0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
int col_symm[11]{0, 1, 3, 0, 2, 1, 4, 0, 4, 2, 3};
float vals_symm[11]{0.7, 0.1, 0.2, 0.1, 0.3, 0.3, 0.4, 0.2, 0.5, 0.4, 0.5};
//// Scipy output
/* COO
>>> print(coo_array((vals_symm, (row_symm, col_symm)), shape=(5,5)).toarray())

Output: [[0.7 0.1 0.  0.2 0. ]
         [0.1 0.  0.3 0.  0. ]
         [0.  0.3 0.  0.  0.4]
         [0.2 0.  0.  0.  0.5]
         [0.  0.  0.4 0.5 0. ]]
*/
/* CSR
>>> print(csr_array((vals_symm, col_symm, row_ptr_symm), shape=(5,5)).toarray())

Output: [[0.7 0.1 0.  0.2 0. ]
         [0.1 0.  0.3 0.  0. ]
         [0.  0.3 0.  0.  0.4]
         [0.2 0.  0.  0.  0.5]
         [0.  0.  0.4 0.5 0. ]]
*/

////////// Skew-symmetric sparse matrix
//// Raw pointers
int row_skew_symm[10]{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
int col_skew_symm[10]{1, 3, 0, 2, 1, 4, 0, 4, 2, 3};
float vals_skew_symm[10]{-0.1, -0.2, 0.1, -0.3, 0.3, -0.4, 0.2, -0.5, 0.4, 0.5};
//// Scipy output
/* COO
>>> print(coo_array((vals_skew_symm, (row_skew_symm, col_skew_symm)), shape=(5,5)).toarray())

Output: [[ 0.  -0.1  0.  -0.2  0. ]
         [ 0.1  0.  -0.3  0.   0. ]
         [ 0.   0.3  0.   0.  -0.4]
         [ 0.2  0.   0.   0.  -0.5]
         [ 0.   0.   0.4  0.5  0. ]]
*/
/*
Script used for (for reference):
``` python
import numpy as np
from scipy.sparse import coo_array, csr_array
row_ptr = np.array([0, 0, 1, 2, 3, 5])
row = np.array([1, 2, 3, 4, 4])
col = np.array([0, 1, 0, 2, 3])
vals = np.array([0.1, 0.3, 0.2, 0.4, 0.5])
row_ptr_symm = np.array([0, 3, 5, 7, 9, 11])
row_symm = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
col_symm = np.array([0, 1, 3, 0, 2, 1, 4, 0, 4, 2, 3])
vals_symm = np.array([0.7, 0.1, 0.2, 0.1, 0.3, 0.3, 0.4, 0.2, 0.5, 0.4, 0.5])
row_skew_symm = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
col_skew_symm = np.array([1, 3, 0, 2, 1, 4, 0, 4, 2, 3])
vals_skew_symm = np.array([-0.1, -0.2, 0.1, -0.3, 0.3, -0.4, 0.2, -0.5, 0.4, 0.5])

(coo_array((vals, (row, col)), shape=(5,5)).toarray())
(csr_array((vals, col, row_ptr), shape=(5,5)).toarray())
(coo_array((vals_symm, (row_symm, col_symm)), shape=(5,5)).toarray())
(csr_array((vals_symm, col_symm, row_ptr_symm), shape=(5,5)).toarray())

(coo_array((vals_skew_symm, (row_skew_symm, col_skew_symm)), shape=(5,5)).toarray())
```

*/

float one_row_one_col_vals[10]{0, 0.1, 0, 0.3, 0.2, 0.4, 0, 0, 0.5, 0};
int one_row_one_col_length = 10;

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
/* Scipy output (adjacency matrix):
array([[0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 1., 1., 0.]])
*/
// CSR = (row_ptr, col, nullptr)
// COO = (row, col, nullptr)

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
/* Scipy output:
array([[1., 1., 0., 1., 0.],
       [1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 1.],
       [1., 0., 0., 0., 1.],
       [0., 0., 1., 1., 0.]])
*/
// CSR = (row_ptr_symm, col_symm, nullptr)
// COO = (row_symm, col_symm, nullptr)

const std::string mtx_skew_symm_data =
    R"(%%MatrixMarket matrix coordinate pattern skew-symmetric
%This is a comment
5 5 5
2 1
4 1
3 2
5 3
5 4
)";
/* Scipy output (coo.A):
array([[ 0., -1.,  0., -1.,  0.],
       [ 1.,  0., -1.,  0.,  0.],
       [ 0.,  1.,  0.,  0., -1.],
       [ 1.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  1.,  1.,  0.]])
*/
// COO = (row_skew-symm, col_skew_symm, nullptr)

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
/* Scipy output:
array([[0. , 0. , 0. , 0. , 0. ],
       [0.1, 0. , 0. , 0. , 0. ],
       [0. , 0.3, 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. ],
       [0. , 0. , 0.4, 0.5, 0. ]])
*/
// COO = (row, col, vals))
// CSR = (row_ptr, col, vals))

const std::string mtx_data_with_values_array =
    R"(%%MatrixMarket matrix array real general
%This is a comment
5 5
0
0.1
0
0.2
0
0
0
0.3
0
0
0
0
0
0
0.4
0
0
0
0
0.5
0
0
0
0
0
)";
/* Scipy output:
array([[0. , 0. , 0. , 0. , 0. ],
       [0.1, 0. , 0. , 0. , 0. ],
       [0. , 0.3, 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. ],
       [0. , 0. , 0.4, 0.5, 0. ]])
*/
// COO = (row, col, vals))
// CSR = (row_ptr, col, vals))

const std::string mtx_data_array =
    R"(%%MatrixMarket matrix array real general
%This is a comment
5 5
0
0.1
0
0.2
0
0
0
0.3
0
0
0
0
0
0
0.4
0
0
0
0
0.5
0
0
0
0
0
)";
/* Scipy output:
array([[0. , 0. , 0. , 0. , 0. ],
       [0.1, 0. , 0. , 0. , 0. ],
       [0. , 0.3, 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. ],
       [0. , 0. , 0.4, 0.5, 0. ]])
*/
// COO = (row, col, vals))
// CSR = (row_ptr, col, vals))

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
/* Scipy output:
array([[0.7, 0.1, 0. , 0.2, 0. ],
       [0.1, 0. , 0.3, 0. , 0. ],
       [0. , 0.3, 0. , 0. , 0.4],
       [0.2, 0. , 0. , 0. , 0.5],
       [0. , 0. , 0.4, 0.5, 0. ]])
*/
// COO = (row_ptr_symm, col_symm, vals_symm))
// CSR = (row_symm, col_symm, vals_symm))

const std::string mtx_skew_symm_data_with_values =
    R"(%%MatrixMarket matrix coordinate real skew-symmetric
%This is a comment
5 5 5
2 1 0.1
4 1 0.2
3 2 0.3
5 3 0.4
5 4 0.5
)";
/* Scipy output:
array([[ 0. , -0.1,  0. , -0.2,  0. ],
       [ 0.1,  0. , -0.3,  0. ,  0. ],
       [ 0. ,  0.3,  0. ,  0. , -0.4],
       [ 0.2,  0. ,  0. ,  0. , -0.5],
       [ 0. ,  0. ,  0.4,  0.5,  0. ]])
*/
// COO = (row_skew_symm, col_skew_symm, vals_skew_symm)

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
/* Scipy output:
array([[0. ],
       [0.1],
       [0. ],
       [0.3],
       [0.2],
       [0.4],
       [0. ],
       [0. ],
       [0.5],
       [0. ]])
*/

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
/* Scipy output:
array([[0. , 0.1, 0. , 0.3, 0.2, 0.4, 0. , 0. , 0.5, 0. ]])
*/

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
/* Scipy output:
array([[0. ],
       [0.1],
       [0. ],
       [0.3],
       [0.2],
       [0.4],
       [0. ],
       [0. ],
       [0.5],
       [0. ]])
*/

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
/* Scipy output:
array([[0. , 0.1, 0. , 0.3, 0.2, 0.4, 0. , 0. , 0.5, 0. ]])
*/
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


//PATOH HyperGraph Reader

int hypergraph1_n = 11;
int hypergraph1_m = 31;
int hypergraph1_constraint_num = 1;
int hypergraph1_base_type = 0;
int hypergraph1_vertex_size = 12;
int hypergraph1_xpins[12]{0,5,7,11,13,15,19,21,25,27,29,31};
int hypergraph1_pins[31]{2, 3, 5, 6, 9, 0, 1, 0, 1, 2, 3, 1, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 10, 11, 8, 10, 8, 11, 2, 5};
int hypergraph1_netWeights[11]{1,1,1,1,1,1,1,1,1,1,1};
int hypergraph1_cellWeights[12]{1,1,1,1,1,1,1,1,1,1,1,1};
int hypergraph1_xnets[13]{0,2,5,8,11,13,17,20,22,25,27,29,31};
int hypergraph1_cells[31]{1, 2, 1, 2, 3, 0, 2, 10, 0, 2, 3, 4, 5, 0, 4, 5, 10, 0, 5, 6, 5, 6, 7, 8, 9, 0, 7, 7, 8, 7, 9};

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

int hypergraph2_n = 9;
int hypergraph2_m = 28;
int hypergraph2_vertex_size = 8;
int hypergraph2_xpins[10]{0,5,9,13,15,17,20,23,26,28};
int hypergraph2_pins[28]{8,6,3,5,2,4,5,1,7,4,2,5,7,4,7,3,5,8,2,4,6,5,2,5,7,2,8,4};
int hypergraph2_netWeights[9]{10,15,13,18,25,20,14,27,29};
int hypergraph2_cellWeights[8]{1,1,1,1,1,1,1,1};
int hypergraph2_xnets[9]{0,1,6,8,13,19,21,25,28};
int hypergraph2_cells[28]{2,1,3,6,7,8,1,5,2,3,4,6,9,1,2,3,5,7,8,1,7,2,3,4,8,1,6,9};

const std::string hypergraph_2 = R"(% base:(0/1) #cells #nets #pins
1 8 9 28 2
10 8 6 3 5 2
15 4 5 1 7
13 4 2 5 7
18 4 7
25 3 5
20 8 2 4
14 6 5 2
27 5 7 2
29 8 4)";

int hypergraph3_n = 9;
int hypergraph3_m = 28;
int hypergraph3_vertex_size = 8;
int hypergraph3_xpins[10]{0,5,9,13,15,17,20,23,26,28};
int hypergraph3_pins[28]{8,6,3,5,2,4,5,1,7,4,2,5,7,4,7,3,5,8,2,4,6,5,2,5,7,2,8,4};
int hypergraph3_netWeights[9]{1,1,1,1,1,1,1,1,1};
int hypergraph3_cellWeights[8]{80,85,30,55,42,39,90,102};
int hypergraph3_xnets[9]{0,1,6,8,13,19,21,25,28};
int hypergraph3_cells[28]{2,1,3,6,7,8,1,5,2,3,4,6,9,1,2,3,5,7,8,1,7,2,3,4,8,1,6,9};

const std::string hypergraph_3 = R"(1 8 9 28 1
8 6 3 5 2
4 5 1 7
4 2 5 7
4 7
3 5
8 2 4
6 5 2
5 7 2
8 4
80 85 30 55 42 39 90 102)";

int hypergraph4_n = 9;
int hypergraph4_m = 28;
int hypergraph4_vertex_size = 8;
int hypergraph4_xpins[10]{0,5,9,13,15,17,20,23,26,28};
int hypergraph4_pins[28]{8,6,3,5,2,4,5,1,7,4,2,5,7,4,7,3,5,8,2,4,6,5,2,5,7,2,8,4};
int hypergraph4_netWeights[9]{10,15,13,18,25,20,14,27,29};
int hypergraph4_cellWeights[8]{80,85,30,55,42,39,90,102};
int hypergraph4_xnets[9]{0,1,6,8,13,19,21,25,28};
int hypergraph4_cells[28]{2,1,3,6,7,8,1,5,2,3,4,6,9,1,2,3,5,7,8,1,7,2,3,4,8,1,6,9};

const std::string hypergraph_4 = R"(1 8 9 28 3
10 8 6 3 5 2
15 4 5 1 7
13 4 2 5 7
18 4 7
25 3 5
20 8 2 4
14 6 5 2
27 5 7 2
29 8 4
80 85 30 55 42 39 90 102)";