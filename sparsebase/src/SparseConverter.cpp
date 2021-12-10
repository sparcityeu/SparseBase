#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseConverter.hpp"
#include <iostream>

using namespace std;

namespace sparsebase
{
    
    size_t format_hash::operator()(Format f) const{
        return f;
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    SparseFormat<ID_t, NNZ_t> * CsrCooFunctor<ID_t, NNZ_t, VAL_t>::operator()(SparseFormat<ID_t, NNZ_t> *source){
        CSR<ID_t,NNZ_t,NNZ_t> *csr = dynamic_cast<CSR<ID_t,NNZ_t,NNZ_t> *>(source);
        COO<ID_t, NNZ_t, NNZ_t> *coo = new COO<ID_t,NNZ_t,NNZ_t>();

        vector<ID_t> dimensions = csr->get_dimensions();
        ID_t n = dimensions[0];
        ID_t m = dimensions[1];
        NNZ_t nnz = csr->get_num_nnz();

        coo->adj = new ID_t[nnz];
        coo->is = new ID_t[nnz];
        if (csr->vals != nullptr)
            coo->vals = new NNZ_t[nnz];
        else
            coo->vals = nullptr;

        ID_t count = 0;        
        for(ID_t i=0; i<n; i++){
            ID_t start = csr->xadj[i];
            ID_t end = csr->xadj[i+1];

            for(ID_t j=start; j<end; j++){
                coo->adj[count] = i;
                count++;
            }
        }

        for(ID_t i=0; i<m; i++){
            coo->is[i] = csr->adj[i];
        }

        if (csr->vals != nullptr)
            for(NNZ_t i=0; i<nnz; i++){
                coo->vals[i] = csr->vals[i];
            }

        vector<ID_t> dims{n,m};
        coo->dimension = dims;
        coo->nnz = nnz;

        return coo;
    }


    // Ai -> row indices -> adj
    // Aj -> col indices -> is
    // Ax -> nnz values -> vals

    // Bp -> row -> xadj
    // Bj -> col -> adj
    // Bx -> nnz -> vals    
    template <typename ID_t, typename NNZ_t, typename VAL_t>
    SparseFormat<ID_t, NNZ_t> * CooCsrFunctor<ID_t, NNZ_t, VAL_t>::operator()(SparseFormat<ID_t, NNZ_t> *source)
    {
        COO<ID_t,NNZ_t,NNZ_t> *coo = dynamic_cast<COO<ID_t,NNZ_t,NNZ_t> *>(source);

        vector<ID_t> dimensions = coo->get_dimensions();
        ID_t n = dimensions[0];
        ID_t m = dimensions[1];
        NNZ_t nnz = coo->get_num_nnz();

        ID_t* xadj = new ID_t[n+1];
        ID_t* adj = new ID_t[m];
        NNZ_t* vals;
        if (coo->vals != nullptr)
            vals = new NNZ_t[nnz];
        else
            vals = nullptr;

        fill(xadj, xadj + n + 1, 0);
        fill(adj, adj + m, 0);
        if (coo->vals != nullptr)
            fill(vals, vals + nnz, 0);

        // We need to ensure that they are sorted
        // Maybe add a sort check and then not do this if it is already sorted
        vector<pair<ID_t,ID_t>> edges;
        for(ID_t i=0; i<nnz; i++){
            edges.emplace_back(coo->adj[i], coo->is[i]);
        }
        sort(edges.begin(), edges.end(), less<pair<ID_t,ID_t>>());


        for(ID_t i=0; i<m; i++){
            adj[i] = edges[i].second;
            xadj[edges[i].first]++;
        }

        for (ID_t i = 1; i <= n; i++)
        {
          xadj[i] += xadj[i - 1];
        }

        for (ID_t i = n; i > 0; i--)
        {
          xadj[i] = xadj[i - 1];
        }
        xadj[0] = 0;


        if (coo->vals != nullptr)
            for(NNZ_t i=0; i<nnz; i++){
                vals[i] = coo->vals[i];
            }
        
        auto csr =  new CSR<ID_t, NNZ_t, NNZ_t>(n, m, xadj, adj, vals);
        return csr;
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    SparseConverter<ID_t, NNZ_t, VAL_t>::SparseConverter()
    {
        this->register_conversion_function(COO_f, CSR_f, new CooCsrFunctor<ID_t,NNZ_t,VAL_t>());
        this->register_conversion_function(CSR_f, COO_f, new CsrCooFunctor<ID_t,NNZ_t,VAL_t>());
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    SparseConverter<ID_t, NNZ_t, VAL_t>::~SparseConverter()
    {
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    void SparseConverter<ID_t, NNZ_t, VAL_t>::register_conversion_function(Format from_format, Format to_format, ConversionFunctor<ID_t, NNZ_t, VAL_t>* conv_func)
    {
        if(conversion_map.count(from_format) == 0){
            conversion_map.emplace(from_format,unordered_map<Format,ConversionFunctor<ID_t,NNZ_t,VAL_t>*, format_hash>());
        }

        if(conversion_map[from_format].count(to_format) == 0){
            conversion_map[from_format].emplace(to_format,conv_func);
        } else {
            conversion_map[from_format][to_format] = conv_func;
        }
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    SparseFormat<ID_t, NNZ_t> *SparseConverter<ID_t, NNZ_t, VAL_t>::convert(SparseFormat<ID_t, NNZ_t> *source, Format to_format)
    {
        try{
            ConversionFunctor<ID_t,NNZ_t,VAL_t>* conv_func = get_conversion_function(source->get_format(),to_format);
            return (*conv_func)(source);
        } catch(...) {
            throw "Unsupported conversion error"; // TODO: Add decent exception mechanism
        }
    }

    template <typename ID_t, typename NNZ_t, typename VAL_t>
    ConversionFunctor<ID_t, NNZ_t,VAL_t>* SparseConverter<ID_t, NNZ_t, VAL_t>::get_conversion_function(Format from_format, Format to_format)
    {
        try {
            return conversion_map[from_format][to_format];
        } catch(...) {
            throw "Unsupported conversion error"; // TODO: Add decent exception mechanism
        }
    }


    template class SparseConverter<int, int, int>;
    template class CooCsrFunctor<int,int,int>;
    template class CsrCooFunctor<int,int,int>;

}
