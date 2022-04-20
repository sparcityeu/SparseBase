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
#include "sparsebase/utils/io/writer.h"
#ifdef CUDA
#include "sparsebase/context/cuda/context.cuh"
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#include "sparsebase/utils/converter/cuda/converter.cuh"
#endif
#endif