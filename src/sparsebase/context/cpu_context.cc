//
// Created by Amro on 3/31/2022.
//

#include "sparsebase/context/cpu_context.h"

#include "sparsebase/context/context.h"

namespace sparsebase::context {

bool CPUContext::IsEquivalent(Context *rhs) const {
  if (dynamic_cast<CPUContext *>(rhs) != nullptr) {
    return true;
  } else {
    return false;
  }
}
}  // namespace sparsebase::context
