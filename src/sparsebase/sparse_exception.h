#ifndef _SPARSEEXC_HPP
#define _SPARSEEXC_HPP

#include "config.h"
#include <exception>
#include <iostream>

namespace sparsebase {

namespace utils {

class Exception : public std::exception {};

class InvalidDataMember : public Exception {
  std::string msg_;

public:
  InvalidDataMember(const std::string &f, const std::string &dm)
      : msg_(std::string("Format ") + f + std::string(" does not have ") + dm +
             std::string(" as a data member.")) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class ReaderException : public Exception {
  std::string msg_;

public:
  ReaderException(const std::string &msg) : msg_(msg) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class WriterException : public Exception {
  std::string msg_;

public:
  WriterException(const std::string &msg) : msg_(msg) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class TypeException : public Exception {
  std::string msg_;

public:
  TypeException(const std::string type1, const std::string type2)
      : msg_("Object is of type " + type1 + " not " + type2) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class ConversionException : public Exception {
  std::string msg_;

public:
  ConversionException(const std::string type1, const std::string type2)
      : msg_("Can not convert type " + type1 + " to " + type2) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class CUDADeviceException : public Exception {
  std::string msg_;

  public:
  CUDADeviceException(const int available_devices, const int requested_device) {
    msg_ = "Attempting to use CUDA device " + std::to_string(requested_device) +
           " when only " + std::to_string(available_devices) +
           " CUDA devices are available\n";
  }
  virtual const char *what() const throw() { return msg_.c_str(); }
};
} // namespace utils

} // namespace sparsebase

#endif