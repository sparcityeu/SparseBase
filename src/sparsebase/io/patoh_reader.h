#ifndef SPARSEBASE_PROJECT_PATOH_READER_H
#define SPARSEBASE_PROJECT_PATOH_READER_H
#include <string>
#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"
#include "sparsebase/object/object.h"

template <typename IDType, typename NNZType, typename ValueType>
class ReadsHyperGraph {
 public:
  //! Reads the file to a Graph instance and returns a pointer to it
  virtual sparsebase::object::HyperGraph<IDType, NNZType, ValueType> *ReadHyperGraph() const = 0;
};

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
class PatohReader : public Reader,
                    public ReadsHyperGraph<IDType, NNZType, ValueType>{
                        
public :

explicit PatohReader(std::string filename);
sparsebase::object::HyperGraph<IDType, NNZType, ValueType> *ReadHyperGraph() const override;
~PatohReader() override;

private:
struct HyperGraphOptions {
    IDType base_type;
    IDType cell_num;
    IDType net_num;
    IDType pin_num;
    IDType weighted_scheme = 0; // Optional, If it equals 1 cells are weighted, If it equals 2 nets are weighted, if equals 3, both are weighted
    IDType constraint_num = 1; // Optional
};
std::string filename_;
HyperGraphOptions options_;
HyperGraphOptions ParseHeader(std::string header_line) const;
};
}
#ifdef _HEADER_ONLY              
#include "patoh_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_PATOH_READER_H