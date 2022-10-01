#include "utils.h"
#include "exception.h"
#include <cxxabi.h>

namespace sparsebase::utils {
    std::string demangle(const std::string& name) {
      int status;
      char *res = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);
      if (status != 0) {
        throw utils::DemangleException(status);
      }

      std::string res_str = res;
      free(res);
      return res_str;
    }

    std::string demangle(std::type_index type){
      return demangle(type.name());
    }
}

