#include "utils.h"

#include <cxxabi.h>

#include <algorithm>
#include <sstream>
#include <string>

#include "exception.h"

#define MMX_PREFIX "%%MatrixMarket"

namespace sparsebase::utils {
std::size_t TypeIndexVectorHash::operator()(
    const std::vector<std::type_index>& vf) const {
  size_t hash = 0;
  for (auto f : vf) hash += f.hash_code();
  return hash;
}
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

namespace MatrixMarket {

MTXOptions ParseHeader(std::string header_line) {
  std::stringstream line_ss(header_line);
  MTXOptions options;
  std::string prefix, object, format, field, symmetry;
  line_ss >> prefix >> object >> format >> field >> symmetry;
  if (prefix != MMX_PREFIX)
    // throw utils::ReaderException("Wrong prefix in a matrix market file");
    // parsing Object option
    if (object == "matrix") {
      options.object = MTXObjectOptions::matrix;
    } else if (object == "vector") {
      options.object = MTXObjectOptions::matrix;
      throw utils::ReaderException(
          "Matrix market reader does not currently support reading vectors.");
    } else {
      throw utils::ReaderException(
          "Illegal value for the 'object' option in matrix market header");
    }
  // parsing format option
  if (format == "array") {
    options.format = MTXFormatOptions::array;
  } else if (format == "coordinate") {
    options.format = MTXFormatOptions::coordinate;
  } else {
    throw utils::ReaderException(
        "Illegal value for the 'format' option in matrix market header");
  }
  // parsing field option
  if (field == "real") {
    options.field = MTXFieldOptions::real;
  } else if (field == "double") {
    options.field = MTXFieldOptions::double_field;
  } else if (field == "complex") {
    options.field = MTXFieldOptions::complex;
  } else if (field == "integer") {
    options.field = MTXFieldOptions::integer;
  } else if (field == "pattern") {
    options.field = MTXFieldOptions::pattern;
  } else {
    throw utils::ReaderException(
        "Illegal value for the 'field' option in matrix market header");
  }
  // parsing symmetry
  if (symmetry == "general") {
    options.symmetry = MTXSymmetryOptions::general;
  } else if (symmetry == "symmetric") {
    options.symmetry = MTXSymmetryOptions::symmetric;
  } else if (symmetry == "skew-symmetric") {
    options.symmetry = MTXSymmetryOptions::skew_symmetric;
  } else if (symmetry == "hermitian") {
    options.symmetry = MTXSymmetryOptions::hermitian;
    throw utils::ReaderException(
        "Matrix market reader does not currently support hermitian symmetry.");
  } else {
    throw utils::ReaderException(
        "Illegal value for the 'symmetry' option in matrix market header");
  }
  return options;
}
}  // namespace MatrixMarket

}  // namespace sparsebase::utils
