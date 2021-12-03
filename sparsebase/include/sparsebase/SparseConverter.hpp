#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "SparseFormat.hpp"
#include <unordered_map>

using namespace std;

namespace sparsebase{

    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class ConversionFunctor {
        public:
            virtual SparseFormat<ID_t,NNZ_t>* operator() (SparseFormat<ID_t, NNZ_t>* source){
                return nullptr;
            }
    };

    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class CsrCooFunctor : public ConversionFunctor<ID_t,NNZ_t, VAL_t> {
        public:
            SparseFormat<ID_t,NNZ_t>* operator() (SparseFormat<ID_t, NNZ_t>* source);
    };
    
    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class CooCsrFunctor : public ConversionFunctor<ID_t,NNZ_t, VAL_t> {
        public:
            SparseFormat<ID_t,NNZ_t>* operator() (SparseFormat<ID_t, NNZ_t>* source);
    };



    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class SparseConverter
    {
    private:
        unordered_map<Format,unordered_map<Format, ConversionFunctor<ID_t,NNZ_t,VAL_t>*>> conversion_map;
    public:
        SparseConverter();
        ~SparseConverter();
        void register_conversion_function(Format from_format, Format to_format, ConversionFunctor<ID_t,NNZ_t,VAL_t>* conv_func);
        ConversionFunctor<ID_t,NNZ_t,VAL_t>* get_conversion_function(Format from_format, Format to_format);
        SparseFormat<ID_t,NNZ_t>* convert(SparseFormat<ID_t,NNZ_t>* source, Format to_format);
    };

}

#endif