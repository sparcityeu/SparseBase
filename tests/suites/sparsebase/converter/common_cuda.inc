#define TYPE int, int, int
using namespace sparsebase;
using namespace converter;
using namespace format;
using namespace context;
class CUDAFormatsFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    csr = new CSR<int, int, int>(n, m, csr_row_ptr, csr_col, csr_vals,
                                 sparsebase::format::kNotOwned);
    array = new Array<int>(nnz, csr_vals, sparsebase::format::kNotOwned);
    cudaMalloc(&cuda_col, nnz * sizeof(int));
    cudaMemcpy(cuda_col, csr_col, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_vals, nnz * sizeof(int));
    cudaMemcpy(cuda_vals, csr_vals, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_row_ptr, (n + 1) * sizeof(int));
    cudaMemcpy(cuda_row_ptr, csr_row_ptr, (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_arr_vals, (nnz) * sizeof(int));
    cudaMemcpy(cuda_arr_vals, csr_vals, (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
  }
  void TearDown() override {
    delete csr;
    delete array;
    cudaFree(cuda_col);
    cudaFree(cuda_row_ptr);
    cudaFree(cuda_vals);
    cudaFree(cuda_arr_vals);
  }
  template <typename T>
  void compare_arrays_cuda_cpu(T* cuda_ptr, T* cpu_ptr, int size) {
    T* cuda_copy_on_cpu = new T[size];
    cudaMemcpy(cuda_copy_on_cpu, cuda_ptr, size * sizeof(T),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(cuda_copy_on_cpu[i], cpu_ptr[i]);
    }
    delete[] cuda_copy_on_cpu;
  }
  template <typename ID, typename NNZ, typename Value>
  void compare_cuda_cpu_csr(CUDACSR<ID, NNZ, Value>* cudacsr,
                            CSR<ID, NNZ, Value>* cpucsr) {
    compare_arrays_cuda_cpu(cudacsr->get_row_ptr(), cpucsr->get_row_ptr(),
                            cpucsr->get_dimensions()[0] + 1);
    compare_arrays_cuda_cpu(cudacsr->get_col(), cpucsr->get_col(),
                            cpucsr->get_num_nnz());
    compare_arrays_cuda_cpu(cudacsr->get_vals(), cpucsr->get_vals(),
                            cpucsr->get_num_nnz());
  }
  int* cuda_col;
  int* cuda_row_ptr;
  int* cuda_vals;
  int* cuda_arr_vals;
  CSR<TYPE>* csr;
  Array<int>* array;
  CUDAContext gpu0{0};
  CPUContext cpu;
  ConverterOrderTwo<TYPE> c2;
  ConverterOrderOne<int> c1;
};
