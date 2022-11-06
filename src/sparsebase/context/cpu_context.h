#include <typeindex>

#include "sparsebase/config.h"
#include "sparsebase/utils/utils.h"
#include "sparsebase/context/context.h"

#ifndef SPARSEBASE_PROJECT_CPU_CONTEXT_H
#define SPARSEBASE_PROJECT_CPU_CONTEXT_H

namespace sparsebase::context {

struct CPUContext : utils::IdentifiableImplementation<CPUContext,Context> {
  virtual bool IsEquivalent(Context *) const;
};

}

#ifdef _HEADER_ONLY
#include "cpu_context.cc"
#endif

#endif  // SPARSEBASE_PROJECT_CPU_CONTEXT_H
