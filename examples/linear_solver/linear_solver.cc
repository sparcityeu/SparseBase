#include <fstream>
#include <iostream>
#include <string>
#include <sparsebase/utils/io/reader.h>
#include <sparsebase/utils/converter/converter.h>
#include <sparsebase/context/context.h>
#include <sparsebase/preprocess/preprocess.h>
#include <omp.h>
#include <stdio.h>
#include <tuple>

#define CHECK

using namespace sparsebase;
using namespace format;
using namespace preprocess;

typedef int ull;
typedef int val;

//template<typename T, typename C>
//std::tuple<val, val, ull> compare(T* j1, T* j2, C count, T threshold, bool verbose = false, bool print_samples = false);

// matrices to try
//  offshor
//  Pres_poisson

std::string A_filename = "TSOPF_FS_b9_c6/TSOPF_FS_b9_c6.mtx";
std::string B_filename = "TSOPF_FS_b9_c6/TSOPF_FS_b9_c6_b.mtx";

using namespace sparsebase;
int main(){
  // Hamrjki thinkVle1/Hamrle1.mtx
  context::CPUContext cpu_context;

  utils::io::MTXReader<ull, ull, val> A_reader(A_filename, true);
  COO<ull, ull, val> * A =  A_reader.ReadCOO();

  ull *perm = ReorderBase::Reorder<RCMReorder>({}, A, {&cpu_context}, true);

  auto * A_reordered = ReorderBase::Permute2D<CSC>(perm, A, {&cpu_context}, true);

  auto *A_csc = A_reordered->Convert<CSC>();

  //utils::io::MTXReader<ull, ull, val> b_reader(B_filename);
  //format::Array<val> * b = b_reader.ReadArray();

  Array<val> *b = new Array<val>(3, nullptr);

  Array<val> * b_reordered = ReorderBase::Permute1D<Array>(perm, b, {&cpu_context}, true);

  // solving for x
  Array<val> *inv_x = new Array<val>(3, nullptr);

  ull *inv_perm = ReorderBase::InversePermutation(perm, A->get_dimensions()[0]);
  format::Array<val> *x = ReorderBase::Permute1D<Array>(inv_perm, inv_x, {&cpu_context}, true);

  float * deg_dist = preprocess::GraphFeatureBase ::GetDegreeDistribution<float>({}, A, {&cpu_context}, true);
  int * deg = preprocess::GraphFeatureBase ::GetDegrees({}, A, {&cpu_context}, true);
  return 0;
}

//// compares two vectors of numbers element wise and returns (mean, std, num_different)
//template<typename T, typename C>
//std::tuple<double, double, ull> compare_vectors(T* v1, T* v2, C count, T threshold, std::vector<C>& indexes){
//  ull  total_different = 0;
//  double sum = 0;
//  for (C i = 0; i<count ; i++){
//    double diff =abs(v1[i]-v2[i]);
//    if (diff>threshold){
//      indexes.push_back(i);
//      total_different++;
//      sum+=diff;
//    }
//  }
//  double mean = sum/ double(count);
//  double sum_sqrd=0;
//  for (C i = 0; i<count ; i++){
//    double diff =abs(v1[i]-v2[i]);
//    if (diff>threshold){
//      sum_sqrd+=pow(diff-mean,2);
//    }
//  }
//  double std_div = sqrt(sum_sqrd/double(count));
//  return std::make_tuple(mean, std_div, total_different);
//}
//
//
//template<typename T, typename C>
//std::tuple<double, double, ull> compare(T* j1, T* j2, C count, T threshold, bool verbose, bool print_samples){
//  std::vector<C> indexes;
//  std::tuple<double, double, ull> comparison_tuple= compare_vectors(j1, j2, count,  threshold, indexes);
//  if (indexes.size() > 0){
//    std::cout << "Errors:\n";
//    for (int i =0; i<indexes.size() && i < 10; i++){
//      std::cout << indexes[i] << " " << j1[indexes[i]] << " " << j2[indexes[i]] << std::endl;
//    }
//  }
//  if (verbose){
//    std::cout << "Mean = " << std::get<0>(comparison_tuple) << " STD = " << std::get<1>(comparison_tuple)  << "Tot. diff. = " << std::get<2>(comparison_tuple) << std::endl;
//    if (print_samples){
//      std::cout << "Samples:\n";
//      std::cout << j1[0] << " " << j2[0] << std::endl;
//      std::cout << j1[10] << " " << j2[10] << std::endl;
//      std::cout << j1[100] << " " << j2[100] << std::endl;
//      std::cout << j1[1000] << " " << j2[1000] << std::endl;
//      std::cout << j1[10000] << " " << j2[10000] << std::endl;
//    }
//  }
//  return comparison_tuple;
//}

// Solving
/*
std::cout << "A properties:" << std::endl;
std::cout << "  Rows: " << A_csc->get_dimensions()[0] << ". Columns: " << A_csc->get_dimensions()[1] << std::endl;
std::cout << "  Number of non-zeros: " << A_csc->get_num_nnz() << std::endl;
std::cout << "b properties:" << std::endl;
std::cout << "  Elements: " << b->get_dimensions()[0] << std::endl;
std::cout << "  Number of non-zeros: " << b->get_num_nnz() << std::endl;


auto A_csc_rows = A_csc->get_rows();
arma::ucolvec row(A_csc->get_row(), A_csc->get_num_nnz());
arma::ucolvec col_ptr(A_csc->get_col_ptr(), A_csc->get_dimensions()[0]+1);
arma::dvec vals(A_csc->get_vals(), A_csc->get_num_nnz());
arma::SpMat<double> a_mat(row, col_ptr, vals, A->get_dimensions()[0], A->get_dimensions()[1]);
arma::dvec b_vals(b_reordered->get_vals(), A->get_dimensions()[0]);

double start = omp_get_wtime();
arma::dvec x = spsolve(a_mat, b_vals, "superlu");
double end = omp_get_wtime();

std::cout << "Solving time: " << end - start << " seconds\n";
*/
