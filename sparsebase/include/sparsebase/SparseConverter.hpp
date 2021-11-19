#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "SparseFormat.hpp"
#include <unordered_map>

using namespace std;

namespace sparsebase{

    template<typename ID_t, typename NNZ_t>
    using ConversionFunction =  SparseFormat<ID_t, NNZ_t>* (*)(SparseFormat<ID_t, NNZ_t>*);

    template<typename ID_t, typename NNZ_t>
    class SparseConverter
    {
    private:
        unordered_map<Format,unordered_map<Format, ConversionFunction<ID_t,NNZ_t>>> conversion_map;
    public:
        SparseConverter();
        ~SparseConverter();
        void register_conversion_function(Format from_format, Format to_format, ConversionFunction<ID_t,NNZ_t> conv_func);
        ConversionFunction<ID_t,NNZ_t> get_conversion_function(Format from_format, Format to_format);
        SparseFormat<ID_t,NNZ_t>* convert(SparseFormat<ID_t,NNZ_t>* source, Format to_format);
    };

}

#endif