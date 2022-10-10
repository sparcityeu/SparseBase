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
#include <vector>
#include <typeindex>

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

class DemangleException : public Exception {
  int status_;

public:
  DemangleException(int status) : status_(status) {}
  virtual const char *what() const throw() {
    if(status_ == -1){
      return "A memory allocation failiure occurred.";
    } else if(status_ == -2){
      return "mangled_name is not a valid name under the C++ ABI mangling rules.";
    } else {
      return "Unknown failure in demangling.";
    }
  }
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
  TypeException(const std::string& msg) : msg_(msg) {}
  TypeException(const std::string& type1, const std::string& type2)
      : msg_("Object is of type " + type1 + " not " + type2) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

class ConversionException : public Exception {
  std::string msg_;

public:
  ConversionException(const std::string& type1, const std::string& type2)
      : msg_("Can not convert type " + type1 + " to " + type2) {}
  virtual const char *what() const throw() { return msg_.c_str(); }
};

template <typename KeyType>
std::string ListOfKeysToString(KeyType vec){
  static_assert(std::is_same<KeyType, std::vector<std::type_index>>::value, "Cannot make a string of keys of other types than vector<type_index>");
  std::string output="[";
  for (auto ti : vec){
    output+=std::string(ti.name())+", ";
  }
  output = output.substr(0, output.size()-2);
  output+="]";
  return output;
}

template <typename KeyType>
class DirectExecutionNotAvailableException : public Exception {
public:
  KeyType used_format_;
  std::vector<KeyType> available_formats_;
  std::string msg_;
  DirectExecutionNotAvailableException(const KeyType & used_format, const std::vector<KeyType> & available_formats): used_format_(used_format), available_formats_(available_formats){
    msg_="Preprocessing could not be used directly using input formats:\n "+ ListOfKeysToString(used_format_)+"\nThis class can only be used with the following formats:\n ";
    for (auto format_ti : available_formats_){
      msg_+= ListOfKeysToString(format_ti)+"\n ";
    }
  }
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