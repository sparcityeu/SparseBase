/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_H_
#define SPARSEBASE_SPARSEBASE_H_
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/format/format.h"
#include "sparsebase/object/object.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/converter/converter.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/utils/io/iobase.h"
#include "sparsebase/utils/io/writer.h"
#ifdef USE_CUDA
#include "sparsebase/context/cuda/context.cuh"
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#include "sparsebase/utils/converter/cuda/converter.cuh"
#endif
#endif