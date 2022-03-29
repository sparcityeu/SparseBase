#ifndef _SPARSEWRITER_HPP
#define _SPARSEWRITER_HPP

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_reader.h"

namespace sparsebase::utils {

template <typename IDType, typename NNZType, typename ValueType> class Writer {
public:
  virtual ~Writer();
};

template <typename IDType, typename NNZType, typename ValueType>
class WritesCSR {
  virtual void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const = 0;
};

template <typename IDType, typename NNZType, typename ValueType>
class WritesCOO {
  virtual void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const = 0;
};

template <typename IDType, typename NNZType, typename ValueType>
class BinaryWriter : public Writer<IDType, NNZType, ValueType>,
                     public WritesCOO<IDType, NNZType, ValueType>,
                     public WritesCSR<IDType, NNZType, ValueType> {
public:
  BinaryWriter(std::string filename);
  ~BinaryWriter() = default;
  void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const;
  void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const;

private:
  std::string filename_;
};


}



#endif