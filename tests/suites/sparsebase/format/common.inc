// The arrays defined here are for two matrices
// One in csr format one in coo format
// These are known to be equivalent (converted using scipy)
int coo_row[4]{0, 0, 1, 3};
int coo_col[4]{0, 2, 1, 3};
int coo_vals[4]{4, 5, 7, 9};
int csr_row_ptr[5]{0, 2, 3, 3, 4};
int csr_col[4]{0, 2, 1, 3};
int csr_vals[4]{4, 5, 7, 9};
int csc_col_ptr[5]{0, 1, 2, 3, 4};
int csc_row[4]{0, 1, 0, 3};
int csc_vals[4]{4, 7, 5, 9};

