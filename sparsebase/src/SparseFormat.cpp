#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseException.hpp"

namespace sparsebase
{

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::~AbstractSparseFormat(){};
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::AbstractSparseFormat() {}

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    unsigned int AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_order()
    {
        return order;
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    Format AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_format()
    {
        return format;
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    std::vector<ID_t> AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_dimensions()
    {
        return dimension;
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    NNZ_t AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_num_nnz()
    {
        return nnz;
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    NNZ_t * AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_row_ptr()
    {
        throw InvalidDataMember(to_string(get_format()), string("row_ptr"));
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_col()
    {
        throw InvalidDataMember(to_string(get_format()), string("col"));
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_row()
    {
        throw InvalidDataMember(to_string(get_format()), string("is"));
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    VAL_t * AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_vals()
    {
        throw InvalidDataMember(to_string(get_format()), string("vals"));
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t ** AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_ind()
    {
        throw InvalidDataMember(to_string(get_format()), string("ind"));
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    COO<ID_t, NNZ_t, VAL_t>::COO()
    {
        this->order = 2;
        this->format = Format::COO_f;
        this->dimension = std::vector<ID_t>(2, 0);
        this->nnz = 0;
        col = nullptr;
        vals = nullptr;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    COO<ID_t, NNZ_t, VAL_t>::COO(ID_t _n, ID_t _m, NNZ_t _nnz, ID_t *_row, ID_t *_col, VAL_t *_vals)
    {
        col = _col;
        row = _row;
        vals = _vals;
        this->nnz = _nnz;
        this->format = Format::CSR_f;
        this->order = 2;
        this->dimension = {_n, _m};
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    Format COO<ID_t,NNZ_t,VAL_t>::get_format(){
        return COO_f;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * COO<ID_t, NNZ_t, VAL_t>::get_col()
    {
        return col;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * COO<ID_t, NNZ_t, VAL_t>::get_row()
    {
        return row;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    VAL_t * COO<ID_t, NNZ_t, VAL_t>::get_vals()
    {
        return vals;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    COO<ID_t, NNZ_t, VAL_t>::~COO(){};

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    CSR<ID_t, NNZ_t, VAL_t>::CSR()
    {
        this->order = 2;
        this->format = Format::CSR_f;
        this->dimension = std::vector<ID_t>(2, 0);
        this->nnz = 0;
        this->col = nullptr;
        this->row_ptr = nullptr;
        this->vals = nullptr;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    CSR<ID_t, NNZ_t, VAL_t>::CSR(ID_t _n, ID_t _m, NNZ_t *_row_ptr, ID_t *_col, VAL_t *_vals)
    {
        this->row_ptr = _row_ptr;
        this->col = _col;
        this->vals = _vals;
        this->format = Format::CSR_f;
        this->order = 2;
        this->dimension = {_n, _m};
        this->nnz = this->row_ptr[this->dimension[0]];
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    Format CSR<ID_t,NNZ_t,VAL_t>::get_format(){
        return CSR_f;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * CSR<ID_t, NNZ_t, VAL_t>::get_col()
    {
        return col;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * CSR<ID_t, NNZ_t, VAL_t>::get_row_ptr()
    {
        return row_ptr;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    VAL_t * CSR<ID_t, NNZ_t, VAL_t>::get_vals()
    {
        return vals;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    CSR<ID_t, NNZ_t, VAL_t>::~CSR(){}

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    CSF<ID_t, NNZ_t, VAL_t>::CSF(unsigned int order)
    {
        //init CSF
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t ** CSF<ID_t, NNZ_t, VAL_t>::get_ind()
    {
        return ind;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    VAL_t * CSF<ID_t, NNZ_t, VAL_t>::get_vals()
    {
        return vals;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    CSF<ID_t, NNZ_t, VAL_t>::~CSF(){};

    template class COO<int, int, int>;
    template class COO<unsigned int, unsigned int, unsigned int>;
    template class COO<unsigned int, unsigned int, void>;
    template class CSR<unsigned int, unsigned int, unsigned int>;
    template class CSR<unsigned int, unsigned int, void>;
    template class CSR<int, int, int>;
};