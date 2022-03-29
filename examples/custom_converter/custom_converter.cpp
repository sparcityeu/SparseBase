#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"

using namespace std;
using namespace sparsebase;

template<typename IDType, typename NNZType, typename VALType>
format::Format * MyFunction(format::Format *source, context::Context*) {
    return nullptr;
}

int main(){

    int row[6] = {0, 0, 1, 1, 2, 2};
    int col[6] = {0, 1, 1, 2, 3, 3};
    int vals[6] = {10, 20, 30, 40, 50, 60};

    format::COO<int,int,int>* coo = new format::COO<int,int,int>(6, 6, 6, row, col, vals);
    context::CPUContext cpu_context;

    auto converter = new utils::OrderTwoConverter<int,int,int>();

    converter->RegisterConditionalConversionFunction(
        format::COO<int, int, int>::get_format_id_static(),
        format::CSR<int, int, int>::get_format_id_static(),
        MyFunction<int, int, int>,
        [](context::Context *, context::Context *) -> bool { return true; });

    auto csr = converter->Convert(coo, format::CSR<int,int, int>::get_format_id_static(), &cpu_context);
    cout << csr << endl;

    delete coo;
    delete converter;
    delete csr;
}

