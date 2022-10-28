(Available_Components)=
#  Available components
## Formats
The following are the formats currently provided by the library classified based on the order.
### FormatOrderOne
Single dimensional formats.
- [Array](exhale_class_classsparsebase_1_1format_1_1_array)
- [CUDAArray](exhale_class_classsparsebase_1_1format_1_1_cuda_array) (requires compilation with CMake option`USE_CUDA=ON`)
### FormatOrderTwo
Two-dimensional formats.
- [CSR](exhale_class_classsparsebase_1_1format_1_1_c_s_r)
- [COO](exhale_class_classsparsebase_1_1format_1_1_c_o_o)
- [CSC](exhale_class_classsparsebase_1_1format_1_1_c_s_c)
- [CUDACSR](exhale_class_classsparsebase_1_1format_1_1_c_u_d_a_c_s_r) (requires compilation with CMake option`USE_CUDA=ON`)
## Functionalities
Below are the functionalities available in SparseBase. After each functionality, the formats for which that functionality is implemented are shown.
### Reordering algorithms
- [RCMReorder](exhale_class_classsparsebase_1_1preprocess_1_1_r_c_m_reorder)
  - CSR
- [GrayReorder](exhale_class_classsparsebase_1_1preprocess_1_1_gray_reorder)
  - CSR
- [DegreeReorder](exhale_class_classsparsebase_1_1preprocess_1_1_degree_reorder)
  - CSR
- [MetisReorder](exhale_class_classsparsebase_1_1preprocess_1_1_metis_reorder) (requires compilation with CMake option`USE_METIS=ON`)
  - CSR
- [RabbitReorder](exhale_class_classsparsebase_1_1preprocess_1_1_rabbit_reorder) (requires compilation with CMake option`USE_RABBIT_REORDER=ON`)
### Partitioning algorithms
- [MetisPartiton](exhale_class_classsparsebase_1_1preprocess_1_1_metis_partition) (requires compilation with CMake option`USE_METIS=ON`)
  - CSR
### Permutation
- [PermuteOrderOne](exhale_class_classsparsebase_1_1preprocess_1_1_permute_order_one)
  - CSR
- [PermuteOrderTwo](exhale_class_classsparsebase_1_1preprocess_1_1_permute_order_two)
  - CSR
## Feature extraction
- [Degrees](exhale_class_classsparsebase_1_1preprocess_1_1_degrees)
  - CSR
- [DegreeDistribution](exhale_class_classsparsebase_1_1preprocess_1_1_degree_distribution)
  - CSR
- [Degrees_DegreeDistribution](exhale_class_classsparsebase_1_1preprocess_1_1_degrees___degree_distribution)
  - CSR
- [JaccardWeights](exhale_class_classsparsebase_1_1preprocess_1_1_jaccard_weights)
  - CUDACSR (requires compilation with `CUDA`)
