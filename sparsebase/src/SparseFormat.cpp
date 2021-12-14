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
    NNZ_t * AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_xadj()
    {
        throw InvalidDataMember(to_string(get_format()), string("xadj"));
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_adj()
    {
        throw InvalidDataMember(to_string(get_format()), string("adj"));
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * AbstractSparseFormat<ID_t, NNZ_t, VAL_t>::get_is()
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
        adj = nullptr;
        vals = nullptr;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    COO<ID_t, NNZ_t, VAL_t>::COO(ID_t _n, ID_t _m, NNZ_t _nnz, ID_t *_adj, ID_t *_is, VAL_t *_vals)
    {
        adj = _adj;
        is = _is;
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
    ID_t * COO<ID_t, NNZ_t, VAL_t>::get_adj()
    {
        return adj;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * COO<ID_t, NNZ_t, VAL_t>::get_is()
    {
        return is;
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
        this->adj = nullptr;
        this->xadj = nullptr;
        this->vals = nullptr;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    CSR<ID_t, NNZ_t, VAL_t>::CSR(ID_t _n, ID_t _m, NNZ_t *_xadj, ID_t *_adj, VAL_t *_vals)
    {
        this->xadj = _xadj;
        this->adj = _adj;
        this->vals = _vals;
        this->format = Format::CSR_f;
        this->order = 2;
        this->dimension = {_n, _m};
        this->nnz = this->xadj[this->dimension[0]];
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    Format CSR<ID_t,NNZ_t,VAL_t>::get_format(){
        return CSR_f;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * CSR<ID_t, NNZ_t, VAL_t>::get_adj()
    {
        return adj;
    }
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t * CSR<ID_t, NNZ_t, VAL_t>::get_xadj()
    {
        return xadj;
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