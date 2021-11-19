#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseConverter.hpp"

using namespace std;
using namespace sparsebase;

int main(){

    int adj[4] = {0, 1, 2, 3};
    int is[4] = {0, 1, 2, 3};
    int vals[4] = {1, 1, 1, 1};

    COO<int,int,int>* coo = new COO<int,int,int>(4,4,4,adj,is,vals);

    auto converter = new SparseConverter<int,int>();
    auto csr = converter->convert(coo,CSR_f);
    auto csr2 = dynamic_cast<CSR<int,int,int>*>(csr);

    for(int i=0; i<4; i++)
        cout << csr2->xadj[i] << ",";
    cout << endl;
    
    for(int i=0; i<4; i++)
        cout << csr2->adj[i] << ",";
    cout << endl;
    
    for(int i=0; i<4; i++)
        cout << csr2->vals[i] << ",";
    cout << endl;
    cout << endl;

    auto coo2 = converter->convert(csr,COO_f);

    auto coo3 = dynamic_cast<COO<int,int,int>*>(coo2);
    
    for(int i=0; i<4; i++)
        cout << coo3->adj[i] << ",";
    cout << endl;
    
    for(int i=0; i<4; i++)
        cout << coo3->is[i] << ",";
    cout << endl;
    
    for(int i=0; i<4; i++)
        cout << coo3->vals[i] << ",";
    cout << endl;
}

