/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_H_
#define SPARSEBASE_SPARSEBASE_UTILS_H_

#include <string>
#include <typeindex>
#include <typeinfo>

namespace sparsebase::utils {

std::string demangle(const std::string& name);

std::string demangle(std::type_index type);

}

#ifdef _HEADER_ONLY
#include "sparsebase/utils/utils.cc"
#endif

#endif