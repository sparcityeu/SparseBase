#include <exception>
#include <iostream>

namespace sparsebase {
class InvalidDataMember : public std::exception {
  std::string msg_;

public:
  InvalidDataMember(const std::string &f, const std::string &dm)
      : msg_(std::string("Format ") + f + std::string(" does not have ") + dm +
            std::string(" as a data member.")) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class SparseReaderException : public std::exception {
  std::string msg_;

public:
  SparseReaderException(const std::string &msg) : msg_(msg) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};
} // namespace sparsebase
