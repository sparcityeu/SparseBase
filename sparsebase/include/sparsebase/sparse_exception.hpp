#include <exception>
#include <iostream>

using namespace std;

namespace sparsebase {
class InvalidDataMember : public exception {
  string msg;

public:
  InvalidDataMember(const string &f, const string &dm)
      : msg(string("Format ") + f + string(" does not have ") + dm +
            string(" as a data member.")) {}
  virtual const char *what() const throw() { return msg.c_str(); }
};

class SparseReaderException : public exception {
  string msg;

public:
  SparseReaderException(const string &msg) : msg(msg) {}
  virtual const char *what() const throw() { return msg.c_str(); }
};
} // namespace sparsebase
