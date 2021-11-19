#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseConverter.hpp"

using namespace std;

namespace sparsebase
{

    template <typename ID_t, typename NNZ_t>
    SparseFormat<ID_t, NNZ_t> *csr_to_coo(SparseFormat<ID_t, NNZ_t> *source){
        CSR<ID_t,NNZ_t,NNZ_t> *csr = dynamic_cast<CSR<ID_t,NNZ_t,NNZ_t> *>(source);
        COO<ID_t, NNZ_t, NNZ_t> *coo = new COO<ID_t,NNZ_t,NNZ_t>();

        vector<ID_t> dimensions = csr->get_dimensions();
        ID_t n = dimensions[0];
        ID_t m = dimensions[1];
        NNZ_t nnz = coo->get_num_nnz();

        coo->adj = new ID_t[nnz];
        coo->is = new ID_t[nnz];
        coo->vals = new NNZ_t[nnz];

        ID_t count = 0;        
        for(ID_t i=0; i<n; i++){
            ID_t start = csr->xadj[i];
            ID_t end = csr->xadj[i+1];

            for(ID_t j=start; j<end; j++){
                coo->is[count] = i;
                coo->adj[count] = j;
                coo->vals[count] = csr->vals[csr->adj[j]];
                count++;
            }
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
    template <typename ID_t, typename NNZ_t>
    SparseFormat<ID_t, NNZ_t> *coo_to_csr(SparseFormat<ID_t, NNZ_t> *source)
    {

        COO<ID_t,NNZ_t,NNZ_t> *coo = dynamic_cast<COO<ID_t,NNZ_t,NNZ_t> *>(source);
        CSR<ID_t,NNZ_t,NNZ_t> *csr = new CSR<ID_t,NNZ_t,NNZ_t>(); // TODO: Fix this

        vector<ID_t> dimensions = coo->get_dimensions();
        ID_t n = dimensions[0];
        ID_t m = dimensions[1];
        NNZ_t nnz = coo->get_num_nnz();

        csr->xadj = new ID_t[n+1];
        csr->adj = new ID_t[m];
        csr->vals = new NNZ_t[nnz];

        fill(csr->xadj, csr->xadj+n+1, 0);
        fill(csr->adj, csr->adj+m, 0);
        fill(csr->vals, csr->vals+nnz, 0);

        ID_t mt = 0;
        for(NNZ_t i=0; i<nnz; i++){
            ID_t u = coo->adj[i];
            ID_t v = coo->is[i];
            NNZ_t w = coo->vals[i];

            csr->xadj[u]++;
            csr->adj[mt++] = v;
        }

        for(ID_t i=0; i<n+1; i++){
            csr->xadj[i] += csr->xadj[i-1];
        }

        for(ID_t i=n; i>0; i--){
            csr->xadj[i] = csr->xadj[i-1];
        }
        csr->xadj[0] = 0;

        for(NNZ_t i=0; i<nnz; i++){
            csr->vals[i] = coo->vals[i];
        }

        vector<ID_t> dims{n,m};
        csr->dimension = dims;
        csr->nnz = nnz;

        return csr;
    }

    template <typename ID_t, typename NNZ_t>
    SparseConverter<ID_t, NNZ_t>::SparseConverter()
    {
        this->register_conversion_function(COO_f, CSR_f, coo_to_csr);
        this->register_conversion_function(CSR_f, COO_f, csr_to_coo);
    }

    template <typename ID_t, typename NNZ_t>
    SparseConverter<ID_t, NNZ_t>::~SparseConverter()
    {
    }

    template <typename ID_t, typename NNZ_t>
    void SparseConverter<ID_t, NNZ_t>::register_conversion_function(Format from_format, Format to_format, ConversionFunction<ID_t, NNZ_t> conv_func)
    {
        if(conversion_map.count(from_format) == 0){
            conversion_map.emplace(from_format,unordered_map<Format,ConversionFunction<ID_t,NNZ_t>>());
        }

        if(conversion_map[from_format].count(to_format) == 0){
            conversion_map[from_format].emplace(to_format,conv_func);
        } else {
            conversion_map[from_format][to_format] = conv_func;
        }
    }

    template <typename ID_t, typename NNZ_t>
    SparseFormat<ID_t, NNZ_t> *SparseConverter<ID_t, NNZ_t>::convert(SparseFormat<ID_t, NNZ_t> *source, Format to_format)
    {
        try{
            auto conv_func = get_conversion_function(source->get_format(),to_format);
            return conv_func(source);
        } catch(...) {
            throw "Unsupported conversion error"; // TODO: Add decent exception mechanism
        }
    }

    template <typename ID_t, typename NNZ_t>
    ConversionFunction<ID_t, NNZ_t> SparseConverter<ID_t, NNZ_t>::get_conversion_function(Format from_format, Format to_format)
    {
        try {
            return conversion_map[from_format][to_format];
        } catch(...) {
            throw "Unsupported conversion error"; // TODO: Add decent exception mechanism
        }
    }


    template class SparseConverter<int, int>;

}