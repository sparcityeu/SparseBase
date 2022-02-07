#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"

using namespace std;
using namespace sparsebase;

template<typename IDType, typename NNZType, typename VALType>
class MyFunctor : public utils::ConversionFunctor<IDType, NNZType, VALType>{
    format::Format * operator()(format::Format *source) {
        return nullptr;
    }
};

int main(){

    int row[6] = {0, 0, 1, 1, 2, 2};
    int col[6] = {0, 1, 1, 2, 3, 3};
    int vals[6] = {10, 20, 30, 40, 50, 60};

    format::COO<int,int,int>* coo = new format::COO<int,int,int>(6, 6, 6, row, col, vals);

    auto converter = new utils::Converter<int,int,int>();

    converter->RegisterConversionFunction(format::COO<int,int, int>::get_format_id_static(), format::CSR<int,int, int>::get_format_id_static(), new MyFunctor<int,int,int>());

    auto csr = converter->Convert(coo, format::CSR<int,int, int>::get_format_id_static());
    cout << csr << endl;

    delete coo;
    delete converter;
    delete csr;
}

