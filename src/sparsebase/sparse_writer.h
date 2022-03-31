#ifndef _SPARSEWRITER_HPP
#define _SPARSEWRITER_HPP

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_reader.h"

namespace sparsebase::utils {

class Writer {
public:
  virtual ~Writer() = default;
};

template <typename IDType, typename NNZType, typename ValueType>
class WritesCSR {
  virtual void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const = 0;
};

template <typename IDType, typename NNZType, typename ValueType>
class WritesCOO {
  virtual void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const = 0;
};

template <typename T> class WritesArray {
  virtual void WriteArray(format::Array<T> arr) const = 0;
};

template <typename IDType, typename NNZType, typename ValueType>
class BinaryWriterOrderTwo : public Writer,
                             public WritesCOO<IDType, NNZType, ValueType>,
                             public WritesCSR<IDType, NNZType, ValueType> {
public:
  explicit BinaryWriterOrderTwo(std::string filename);
  ~BinaryWriterOrderTwo() override = default;
  void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const;
  void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const;

private:
  std::string filename_;
};

template <typename T>
class BinaryWriterOrderOne : public Writer, public WritesArray<T> {
public:
  explicit BinaryWriterOrderOne(std::string filename);
  ~BinaryWriterOrderOne() override = default;
  void WriteArray(format::Array<T> *arr) const;

private:
  std::string filename_;
};

}



#endif