/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#include "sparsebase/config.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

#ifndef SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CUDA_CONVERTER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CUDA_CONVERTER_H_

namespace sparsebase::converter {
bool CUDAPeerToPeer(sparsebase::context::Context *from,
                    sparsebase::context::Context *to);

}  // namespace sparsebase::converter

#endif  // SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_H_