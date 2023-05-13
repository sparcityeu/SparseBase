#ifndef SPARSEBASE_PROJECT_PATOH_WRITER_H
#define SPARSEBASE_PROJECT_PATOH_WRITER_H
#include <string>
#include "sparsebase/config.h"
#include "sparsebase/io/writer.h"
#include "sparsebase/object/object.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
class PatohWriter: public Writer,
                    public WritesHyperGraph<IDType, NNZType, ValueType> {
    
public:

explicit PatohWriter(std::string filename, bool is_zero_indexed = false , bool is_edge_weighted = false,
                                                                          bool is_vertex_weighted = false, int constraint_num = 1);

void WriteHyperGraph(object::HyperGraph<IDType, NNZType, ValueType> *hyperGraph) const override;
~PatohWriter() override = default;

private: 
std::string filename_;
bool is_zero_indexed_;
bool is_edge_weighted_;
bool is_vertex_weighted_;
};

} // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "patoh_writer.cc"
#endif
#endif  // SPARSEBASE_PROJECT_PATOH_WRITER_H