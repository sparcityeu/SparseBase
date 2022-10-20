/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_H_
#define SPARSEBASE_SPARSEBASE_UTILS_H_

#include <string>
#include <typeindex>
#include <typeinfo>
#include <limits>
#include <cstdint>
#include "exception.h"
#include <fstream>
#include <algorithm>
#include <sstream>
#include <ctime>

namespace sparsebase::utils {

// Thanks to artificial mind blog: https://artificial-mind.net/blog/2020/10/03/always-false
template <typename ... T>
constexpr bool always_false = false;

using std::numeric_limits;

// Cross float-integral type conversion is not currently available
template <typename T, typename U>
bool CanTypeFitValue(const U value) {
  if constexpr (std::is_integral_v<T> != std::is_integral_v<U>) return false;
  if constexpr (std::is_integral_v<T> && std::is_integral_v<U> ) {
    const intmax_t botT = []() {
      if constexpr (std::is_floating_point_v<T>)
        return intmax_t(-(numeric_limits<T>::max()));
      else
        return intmax_t(numeric_limits<T>::min());
    }();
    const intmax_t botU = []() {
      if constexpr (std::is_floating_point_v<U>)
        return intmax_t(-(numeric_limits<U>::max()));
      else
        return intmax_t(numeric_limits<U>::min());
    }();
    const uintmax_t topT = uintmax_t(numeric_limits<T>::max());
    const uintmax_t topU = uintmax_t(numeric_limits<U>::max());
    return !((botT > botU && value < (U)(botT)) ||
             (topT < topU && value > (U)(topT)));
  } else if constexpr(!std::is_integral_v<T> && !std::is_integral_v<U> ) {
    const double botT = []() {
      if constexpr (std::is_floating_point_v<T>)
        return T(-(numeric_limits<T>::max()));
      else
        return T(numeric_limits<T>::min());
    }();
    const double botU = []() {
      if constexpr (std::is_floating_point_v<U>)
        return U(-(numeric_limits<U>::max()));
      else
        return U(numeric_limits<U>::min());
    }();
    const double topT = numeric_limits<T>::max();
    const double topU = numeric_limits<U>::max();
    return !((botT > botU && value < (U)(botT)) ||
             (topT < topU && value > (U)(topT)));
  }
  //} else if constexpr (std::is_integral_v<T> && !std::is_integral_v<U> ){
  //  const double topT = double(numeric_limits<T>::max());
  //  const uintmax_t topU = uintmax_t(numeric_limits<U>::max());
  //  const double botT = []() {
  //    if constexpr (std::is_floating_point_v<T>)
  //      return double(-(numeric_limits<T>::max()));
  //    else
  //      return double(numeric_limits<T>::min());
  //  }();
  //  const intmax_t botU = []() {
  //    if constexpr (std::is_floating_point_v<U>)
  //      return intmax_t(-(numeric_limits<U>::max()));
  //    else
  //      return intmax_t(numeric_limits<U>::min());
  //  }();
  //  return !(double(topU) > topT && double(value) > topT) || !(double(botU) < botT && double(value) > topT);
  //} else {
  //  const uintmax_t topT = uintmax_t(numeric_limits<T>::max());
  //  const double topU = double(numeric_limits<U>::max());
  //  const intmax_t botT = []() {
  //    if constexpr (std::is_floating_point_v<T>)
  //      return intmax_t(-(numeric_limits<T>::max()));
  //    else
  //      return intmax_t(numeric_limits<T>::min());
  //  }();
  //  const double botU = []() {
  //    if constexpr (std::is_floating_point_v<U>)
  //      return double(-(numeric_limits<U>::max()));
  //    else
  //      return double(numeric_limits<U>::min());
  //  }();
  //  return !(topU > double(topT) && value > double(topT)) || !(botU < double(botT) && value > double(topT));
  //}
}

template <typename FromType, typename ToType>
inline bool isTypeConversionSafe(FromType from_val, ToType to_val) {
  return from_val == to_val && CanTypeFitValue<ToType>(from_val);
}

template <typename ToType, typename FromType, typename SizeType>
ToType *ConvertArrayType(FromType *from_ptr, SizeType size) {
  if (from_ptr == nullptr) return nullptr;
  auto to_ptr = new ToType[size];
  for (SizeType i = 0; i < size; i++){
    to_ptr[i] = from_ptr[i];
    if (!isTypeConversionSafe(from_ptr[i], to_ptr[i])) {
      throw utils::TypeException(
          "Could not convert array from type " +
          std::string(std::type_index(typeid(FromType)).name()) + " to type " +
          std::string(std::type_index(typeid(ToType)).name()) + ". Overflow detected");
    }
  }
  return to_ptr;
}

template <typename T>
class OnceSettable {
public:
  OnceSettable(): is_set_(false){}
  operator T() const {
    return data_;
  }
  OnceSettable(const OnceSettable&) = delete;
  OnceSettable(OnceSettable&&) = delete;
  OnceSettable& operator=(T&& data){
    if (!is_set_) {
      data_ = std::move(data);
      is_set_ = true;
      return *this;
    }
    throw utils::AttemptToReset<T>();
  }
  const T& get() const{
    return data_;
  }
private:
  T data_;
  bool is_set_;
};
std::string demangle(const std::string& name);

std::string demangle(std::type_index type);


enum LogLevel {
  LOG_LVL_INFO,
  LOG_LVL_WARNING,
  LOG_LVL_NONE,
};

class Logger {

private:

  std::string root;
  static LogLevel level;
  static bool use_stdout;
  static bool use_stderr;
  static std::string filename;
  std::ofstream file;



public:
  Logger() {
    if(!Logger::filename.empty()) file.open(filename);
  }

  Logger(std::type_index root_type) {
    root = demangle(root_type);
    root.erase(std::remove(root.begin(), root.end(), '*'), root.end());
    if(!Logger::filename.empty()) file.open(filename);
  }

  ~Logger(){
    if(file.is_open()) file.close();
  }

  static void SetLogLevel(LogLevel new_level){
    Logger::level = new_level;
  }

  static void SetStdOut(bool use){
    Logger::use_stdout = use;
  }

  static void SetStdErr(bool use){
    Logger::use_stderr = use;
  }

  static void SetFile(const std::string& new_filename){
    Logger::filename = new_filename;
  }

  void Log(const std::string& message, LogLevel msg_level = LOG_LVL_INFO){

    if(msg_level < Logger::level){
      return;
    }

    std::time_t current = std::time(0);
    auto now_tm = std::localtime(&current);
    char buffer[30];
    size_t size = strftime(buffer, 30, "%x %X", now_tm);
    std::string now_str(buffer, buffer + size);

    std::stringstream ss;
    ss << "[" << now_str << "]" << " ";

    std::string level_str = "INFO";
    if(msg_level == LOG_LVL_WARNING){
      level_str = "WARNING";
    }
    ss << "[" << level_str << "]" << " ";

    ss << "[" << root << "]" << " ";

    ss << message;

    std::string log = ss.str();

    if(file.is_open()) file << log << std::endl;
    if(Logger::use_stdout) std::cout << log << std::endl;
    if(Logger::use_stderr) std::cerr << log << std::endl;
  }
};

}

#ifdef _HEADER_ONLY
#include "sparsebase/utils/utils.cc"
#endif

#endif