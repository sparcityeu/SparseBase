/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_CONTEXT_CONTEXT_H_
#define SPARSEBASE_SPARSEBASE_CONTEXT_CONTEXT_H_

#include <typeindex>

namespace sparsebase {

namespace context {
struct Context {
  virtual bool IsEquivalent(Context *) const = 0;
  virtual std::type_index get_context_type_member() const = 0;
  virtual ~Context();
};

template <typename ContextType> struct ContextImplementation : public Context {
  virtual std::type_index get_context_type_member() const {
    return typeid(ContextType);
  }
  static std::type_index get_context_type() { return typeid(ContextType); }
};

struct CPUContext : ContextImplementation<CPUContext> {
  virtual bool IsEquivalent(Context *) const;
};
}; // namespace context
}; // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparsebase/context/context.cc"
#endif
#endif // SPARSEBASE_SPARSEBASE_CONTEXT_CONTEXT_H_
