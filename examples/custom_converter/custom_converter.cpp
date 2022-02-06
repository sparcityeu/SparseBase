#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"

using namespace std;
using namespace sparsebase;

template<typename IDType, typename NNZType, typename VALType>
class MyFunctor : public ConversionFunctor<IDType, NNZType, VALType>{
    SparseFormat<IDType, NNZType, VALType> * operator()(SparseFormat<IDType, NNZType, VALType> *source) {
        return nullptr;
    }
};

int main(){

    int row[6] = {0, 0, 1, 1, 2, 2};
    int col[6] = {0, 1, 1, 2, 3, 3};
    int vals[6] = {10, 20, 30, 40, 50, 60};

    COO<int,int,int>* coo = new COO<int,int,int>(6, 6, 6, row, col, vals);

    auto converter = new SparseConverter<int,int,int>();

    converter->RegisterConversionFunction(kCOOFormat, kCSRFormat, new MyFunctor<int,int,int>());

    auto csr = converter->Convert(coo, kCSRFormat);
    cout << csr << endl;

    delete coo;
    delete converter;
    delete csr;
}

