/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_
#define SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_

#include <any>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

#include "sparsebase/utils/class_matcher_mixin.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/feature/extractor.h"
#include "sparsebase/utils/utils.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class FeatureExtractor : public Extractor {
 public:
  FeatureExtractor();
};

}  // namespace sparsebase::feature

#ifdef _HEADER_ONLY
#include "sparsebase/feature/feature_extractor.cc"
#endif

#endif  // SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_