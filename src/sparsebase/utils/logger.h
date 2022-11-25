#ifndef SPARSEBASE_PROJECT_LOGGER_H
#define SPARSEBASE_PROJECT_LOGGER_H

#include <fstream>
#include <string>
#include <typeindex>

namespace sparsebase::utils {

enum LogLevel {
  LOG_LVL_INFO,
  LOG_LVL_WARNING,
  LOG_LVL_NONE,
};

class Logger {
 private:
  std::string root_;
  static LogLevel level_;
  static bool use_stdout_;
  static bool use_stderr_;
  static std::string filename_;
  std::ofstream file_;

 public:
  Logger();

  Logger(std::type_index root_type);

  ~Logger();

  static void set_level(LogLevel level) { Logger::level_ = level; }

  static void set_stdout(bool use) { Logger::use_stdout_ = use; }

  static void set_stderr(bool use) { Logger::use_stderr_ = use; }

  static void set_file(const std::string& filename) {
    Logger::filename_ = filename;
  }

  void Log(const std::string& message, LogLevel msg_level = LOG_LVL_INFO);
};
}  // namespace sparsebase::utils
#ifdef _HEADER_ONLY
#include "logger.cc"
#endif

#endif  // SPARSEBASE_PROJECT_LOGGER_H
