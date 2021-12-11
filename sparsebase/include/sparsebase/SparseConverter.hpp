#ifndef _SPARSECONVERTER_HPP
#define _SPARSECONVERTER_HPP

#include "SparseFormat.hpp"
#include <unordered_map>

using namespace std;

namespace sparsebase{

    typedef std::vector<std::tuple<bool, Format> > conversion_schema;
    struct format_hash{
        size_t operator()(Format f) const;
    };

    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class ConversionFunctor {
        public:
            virtual SparseFormat<ID_t,NNZ_t,VAL_t>* operator() (SparseFormat<ID_t, NNZ_t, VAL_t>* source){
                return nullptr;
            }
    };

    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class CsrCooFunctor : public ConversionFunctor<ID_t,NNZ_t, VAL_t> {
        public:
            SparseFormat<ID_t,NNZ_t,VAL_t>* operator() (SparseFormat<ID_t, NNZ_t, VAL_t>* source);
    };
    
    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class CooCsrFunctor : public ConversionFunctor<ID_t,NNZ_t, VAL_t> {
        public:
            SparseFormat<ID_t,NNZ_t,VAL_t>* operator() (SparseFormat<ID_t, NNZ_t,VAL_t>* source);
    };

    template<typename ID_t, typename NNZ_t, typename VAL_t>
    class SparseConverter
    {
    private:
        unordered_map<Format,std::unordered_map<Format, ConversionFunctor<ID_t,NNZ_t,VAL_t>*, format_hash>,format_hash>conversion_map;
    public:
        SparseConverter();
        ~SparseConverter();
        void register_conversion_function(Format from_format, Format to_format, ConversionFunctor<ID_t,NNZ_t,VAL_t>* conv_func);
        ConversionFunctor<ID_t,NNZ_t,VAL_t>* get_conversion_function(Format from_format, Format to_format);
        SparseFormat<ID_t,NNZ_t, VAL_t>* convert(SparseFormat<ID_t,NNZ_t,VAL_t>* source, Format to_format);
        bool can_convert(Format from_format, Format to_format);
        std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> apply_conversion_schema(conversion_schema cs, std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*>packed_sfs);
        
    };

}

#endif