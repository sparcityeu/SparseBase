//
// Created by Amro on 3/31/2022.
//

#ifndef SPARSEBASE_PROJECT_CONTEXT_H
#define SPARSEBASE_PROJECT_CONTEXT_H

#include <typeindex>

namespace sparsebase {

namespace context {
struct Context{
  virtual bool IsEquivalent(Context*) const = 0;
  virtual std::type_index get_context_type_member() const = 0;
  virtual ~Context();
};

template <typename ContextType>
struct ContextImplementation : public Context {
  virtual std::type_index get_context_type_member() const {
    return typeid(ContextType);
  }
  static std::type_index get_context_type() {
    return typeid(ContextType);
  }
};

struct CPUContext : ContextImplementation<CPUContext>{
  virtual bool IsEquivalent(Context *) const;
};

};
};
#endif // SPARSEBASE_PROJECT_CONTEXT_H
