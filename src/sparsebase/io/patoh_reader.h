#ifndef SPARSEBASE_PROJECT_PATOH_READER_H
#define SPARSEBASE_PROJECT_PATOH_READER_H

#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
class PatohReader : public Reader,
                    public ReadsCSR<IDType, NNZType, ValueType>{
                        
public :

explicit PatohReader(std::string filename,bool convert_to_zero_index = true);
void ReadCSR();
format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
~PatohReader() override;

private:
struct HyperGraphOptions {
    int base_type;
    int cell_num;
    int net_num;
    int pin_num;
    int weight_scheme = NULL; // Optional, If it equals 1 cells are weighted, If it equals 2 nets are weighted, if equals 3, both are weighted
    int constraint_num = 1; // Optional
};
std::string filename_;
bool convert_to_zero_index_;
HyperGraphOptions options_;
HyperGraphOptions ParseHeader(std::string header_line) const;
};
}
#ifdef _HEADER_ONLY              
#include "patoh_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_PATOH_READER_H