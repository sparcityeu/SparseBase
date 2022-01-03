#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"

using namespace std;
using namespace sparsebase;

int main(){

    int row[6] = {0, 0, 1, 1, 2, 2};
    int col[6] = {0, 1, 1, 2, 3, 3};
    int vals[6] = {10, 20, 30, 40, 50, 60};

    COO<int,int,int>* coo = new COO<int,int,int>(6, 6, 6, row, col, vals);

    auto converter = new SparseConverter<int,int,int>();
    auto csr = converter->Convert(coo,kCSRFormat);
    auto csr2 = dynamic_cast<CSR<int,int,int>*>(csr);

    auto dims = csr2->get_dimensions();
    int n = dims[0];
    int m = dims[1];
    int nnz = csr->get_num_nnz();

    cout << "CSR" << endl;

    for(int i=0; i<nnz; i++)
        cout << csr2->vals_[i] << ",";
    cout << endl;

    for(int i=0; i<nnz; i++)
        cout << csr2->col_[i] << ",";
    cout << endl;
    
    for(int i=0; i<n+1; i++)
        cout << csr2->row_ptr_[i] << ",";
    cout << endl;
    
    cout << endl;

    auto coo2 = converter->Convert(csr,kCOOFormat);

    auto coo3 = dynamic_cast<COO<int,int,int>*>(coo2);

    cout << "COO" << endl;

    for(int i=0; i<nnz; i++)
        cout << coo3->vals_[i] << ",";
    cout << endl;

    for(int i=0; i<nnz; i++)
        cout << coo3->row_[i] << ",";
    cout << endl;
    
    for(int i=0; i<nnz; i++)
        cout << coo3->col_[i] << ",";
    cout << endl;

    delete coo;
    delete converter;
    delete csr2;
    delete coo3;
}

