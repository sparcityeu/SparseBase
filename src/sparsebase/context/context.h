/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_CONTEXT_CONTEXT_H_
#define SPARSEBASE_SPARSEBASE_CONTEXT_CONTEXT_H_

#include <typeindex>

#include "sparsebase/config.h"
#include "sparsebase/utils/utils.h"

namespace sparsebase::context {
struct Context : public utils::Identifiable{
  virtual bool IsEquivalent(Context *) const = 0;
  virtual ~Context(){}
};

}  // namespace sparsebase
#endif  // SPARSEBASE_SPARSEBASE_CONTEXT_CONTEXT_H_
