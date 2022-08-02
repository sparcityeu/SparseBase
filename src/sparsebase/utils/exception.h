/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_EXCEPTION_H_
#define SPARSEBASE_SPARSEBASE_UTILS_EXCEPTION_H_

#include "sparsebase/config.h"
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

class FunctionNotFoundException : public Exception {
  std::string msg_;

public:
  FunctionNotFoundException(std::string message) : msg_(message) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class NoConverterException : public Exception {
  std::string msg_;

public:
  NoConverterException()
      : msg_("Attempting to convert a format in a preprocessing object that "
             "does not have a Converter") {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class FeatureException : public Exception {
  std::string msg_;

public:
  FeatureException(const std::string feature, const std::string extractor)
      : msg_("ERROR! " + feature + " is not registered in " + extractor + "!") {
  }
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class FeatureParamsException : public Exception {
  std::string msg_;

public:
  FeatureParamsException(const std::string feature, const std::string type)
      : msg_("ERROR! " + feature + " do not store params for " + type + "!") {}
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

#endif // SPARSEBASE_SPARSEBASE_UTILS_EXCEPTION_H_