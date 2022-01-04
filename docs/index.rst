.. SparseBase documentation master file, created by
   sphinx-quickstart on Sat Jan  1 23:46:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SparseBase's documentation!
======================================

Our vision for
SparseBase is a library built in C++ that encapsulates, preprocesses,
and performs I/O operations on sparse data structures seamlessly and
optimally to provide a backbone for algorithms that use these structures.

It is designed with HPC (High Performance Computing) usage at the forefront. It is meant as a container of sparse objects such as tensors,
graphs, multi-graphs, and hypergraphs. It mainly focuses on re-ordering, partitioning, and
coarsening sparse objects. Also, it supports many different formats (representations) of sparse data and enables
smooth conversion between formats.

SparseBase is not meant as a library for carrying out downstream tasks with data it represents. It won’t do graph
embedding, tensor-tensor mathematical operations, or factorization, for example. Instead, it is a container and
preprocessor that will make carrying out different downstream tasks easier and more standardized.
It will give developers quick utilities in terms of converting between formats, preprocessing, and reading/writing.

The library is currently in an early stage of development. As a result the API is not stable and
likely to change in the near future.

If you are new to the project, we suggest starting here: :doc:`pages/getting_started`

.. toctree::
   :glob:
   :maxdepth: 2


   pages/getting_started
   pages/explanations/index
   pages/how_to_guides/index
   api/library_root
   pages/contributing/index

