#ifndef SPARSEBASE_PROJECT_METIS_GRAPH_READER_H
#define SPARSEBASE_PROJECT_METIS_GRAPH_READER_H
#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"
#include "sparsebase/object/object.h"

//! Interface for readers that can return a Graph instance
template <typename IDType, typename NNZType, typename ValueType>
class ReadsGraph {
 public:
  //! Reads the file to a Graph instance and returns a pointer to it
  virtual sparsebase::object::Graph<IDType, NNZType, ValueType> *ReadGraph() const = 0;
};

namespace sparsebase::io {
//! Reader for the METIS GRAPH file format
/*!
 * File format: https://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html
 */
template <typename IDType, typename NNZType, typename ValueType>
class MetisGraphReader : public Reader,
                       public ReadsGraph<IDType, NNZType, ValueType> {
 public:
  /*!
   * Constructor for the MetisGraphReader class
   * @param filename path to the file to be read
   * @param convert_to_zero_index if set to true the indices will be converted
   */
  explicit MetisGraphReader(std::string filename, bool convert_to_zero_index = false);
  sparsebase::object::Graph<IDType, NNZType, ValueType> *ReadGraph() const override;
  ~MetisGraphReader() override;

 private:
  std::string filename_;
  bool convert_to_zero_index_;
};

}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "metis_graph_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_METIS_GRAPH_READER_H
