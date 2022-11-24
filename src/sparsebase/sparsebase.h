/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_H_
#define SPARSEBASE_SPARSEBASE_H_
#include "sparsebase/config.h"
#include "sparsebase/bases/iobase.h"
#include "sparsebase/bases/graph_feature_base.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/converter/converter_order_two.h"
#include "sparsebase/converter/converter_order_one.h"
#include "sparsebase/converter/converter_store.h"
#include "sparsebase/experiment/experiment_type.h"
#include "sparsebase/experiment/concrete_experiment.h"
#include "sparsebase/experiment/experiment_helper.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_implementation.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/array.h"
#include "sparsebase/object/object.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/logger.h"
#include "sparsebase/utils/utils.h"
#ifdef USE_CUDA
#include "sparsebase/context/cuda_context_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/utils/utils_cuda.cuh"
#endif
#endif