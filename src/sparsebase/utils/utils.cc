#include "utils.h"

#include <cxxabi.h>

#include <algorithm>
#include <ctime>
#include <sstream>

#include "exception.h"

namespace sparsebase::utils {
std::string demangle(const std::string& name) {
  int status;
  char* res = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);
  if (status != 0) {
    throw utils::DemangleException(status);
  }

  std::string res_str = res;
  free(res);
  return res_str;
}

std::string demangle(std::type_index type) { return demangle(type.name()); }

Logger::Logger(std::type_index root_type) {
  root_ = demangle(root_type);
  root_.erase(std::remove(root_.begin(), root_.end(), '*'), root_.end());
  if (!Logger::filename_.empty()) file_.open(filename_);
}

Logger::Logger() {
  if (!Logger::filename_.empty()) file_.open(filename_);
}

Logger::~Logger() {
  if (file_.is_open()) file_.close();
}

void Logger::Log(const std::string& message, LogLevel msg_level) {
  if (msg_level < Logger::level_) {
    return;
  }

  std::time_t current = std::time(0);
  auto now_tm = std::localtime(&current);
  char buffer[30];
  size_t size = strftime(buffer, 30, "%x %X", now_tm);
  std::string now_str(buffer, buffer + size);

  std::stringstream ss;
  ss << "[" << now_str << "]"
     << " ";

  std::string level_str = "INFO";
  if (msg_level == LOG_LVL_WARNING) {
    level_str = "WARNING";
  }
  ss << "[" << level_str << "]"
     << " ";

  ss << "[" << root_ << "]"
     << " ";

  ss << message;

  std::string log = ss.str();

  if (file_.is_open()) file_ << log << std::endl;
  if (Logger::use_stdout_) std::cout << log << std::endl;
  if (Logger::use_stderr_) std::cerr << log << std::endl;
}

LogLevel Logger::level_ = LOG_LVL_WARNING;
std::string Logger::filename_;
bool Logger::use_stdout_ = true;
bool Logger::use_stderr_ = false;
}  // namespace sparsebase::utils
