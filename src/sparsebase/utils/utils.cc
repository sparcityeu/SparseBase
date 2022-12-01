#include "utils.h"

#include <cxxabi.h>

#include <algorithm>
#include <sstream>

#include "exception.h"

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

}  // namespace sparsebase::utils
