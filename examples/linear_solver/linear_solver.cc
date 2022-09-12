#include <armadillo>
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

typedef int ull;
typedef int val;

template<typename T, typename C>
std::tuple<val, val, ull> compare(T* j1, T* j2, C count, T threshold, bool verbose = false, bool print_samples = false);

// matrices to try
//  offshor
//  Pres_poisson

std::string A_filename = "TSOPF_FS_b9_c6/TSOPF_FS_b9_c6.mtx";
std::string B_filename = "TSOPF_FS_b9_c6/TSOPF_FS_b9_c6_b.mtx";

using namespace sparsebase;
int main(){
  // Hamrjki thinkVle1/Hamrle1.mtx
  context::CPUContext cpu_context;

  utils::io::MTXReader<ull, ull, val> A_reader(A_filename);
  format::COO<ull, ull, val> * A =  A_reader.ReadCOO();

  //preprocess::ReorderingSuite<ull, ull, val> reorder;

  //ull *perm = reorder.Reorder<preprocess::RCMReorder>({}, A, {&cpu_context});
  ull *perm = preprocess::ReorderingSuite::Reorder<preprocess::RCMReorder>({}, A, {&cpu_context});

  //auto * A_reordered = reorder.Permute2D(perm, A, {&cpu_context})->Convert<format::COO>();
  auto * A_reordered = preprocess::ReorderingSuite::Permute2D(perm, A, {&cpu_context})->Convert<format::COO>();

  auto *A_csc = A_reordered->Convert<format::CSC>();

  //utils::io::MTXReader<ull, ull, val> b_reader(B_filename);
  //format::Array<val> * b = b_reader.ReadArray();

  format::Array<val> *b = new format::Array<val>(3, nullptr);

  //format::Array<val> * b_reordered = reorder.Permute1D(perm, b, {&cpu_context})->Convert<format::Array>();
  format::Array<val> * b_reordered = preprocess::ReorderingSuite::Permute1D<int, int>(perm, b, {&cpu_context})->Convert<format::Array>();

  // solving for x
  format::Array<val> *inv_x = new format::Array<val>(3, nullptr);
  //ull *inv_perm = reorder.InversePermutation(perm, A);
  //format::Array<val> *x = reorder.Permute1D(inv_perm, inv_x, {&cpu_context})->Convert<format::Array>();
  ull *inv_perm = preprocess::ReorderingSuite::InversePermutation(perm, A);
  format::Array<val> *x = preprocess::ReorderingSuite::Permute1D(inv_perm, inv_x, {&cpu_context})->Convert<format::Array>();

  float * deg_dist = preprocess::GraphFeatureSuite::GetDegreeDistribution<float>({}, A, {&cpu_context});
  float * deg = preprocess::GraphFeatureSuite::GetDegrees({}, A, {&cpu_context});
  cout << deg_dist << deg <<
  return 0;
}

// compares two vectors of numbers element wise and returns (mean, std, num_different)
template<typename T, typename C>
std::tuple<double, double, ull> compare_vectors(T* v1, T* v2, C count, T threshold, std::vector<C>& indexes){
  ull  total_different = 0;
  double sum = 0;
  for (C i = 0; i<count ; i++){
    double diff =abs(v1[i]-v2[i]);
    if (diff>threshold){
      indexes.push_back(i);
      total_different++;
      sum+=diff;
    }
  }
  double mean = sum/ double(count);
  double sum_sqrd=0;
  for (C i = 0; i<count ; i++){
    double diff =abs(v1[i]-v2[i]);
    if (diff>threshold){
      sum_sqrd+=pow(diff-mean,2);
    }
  }
  double std_div = sqrt(sum_sqrd/double(count));
  return std::make_tuple(mean, std_div, total_different);
}


template<typename T, typename C>
std::tuple<double, double, ull> compare(T* j1, T* j2, C count, T threshold, bool verbose, bool print_samples){
  std::vector<C> indexes;
  std::tuple<double, double, ull> comparison_tuple= compare_vectors(j1, j2, count,  threshold, indexes);
  if (indexes.size() > 0){
    std::cout << "Errors:\n";
    for (int i =0; i<indexes.size() && i < 10; i++){
      std::cout << indexes[i] << " " << j1[indexes[i]] << " " << j2[indexes[i]] << std::endl;
    }
  }
  if (verbose){
    std::cout << "Mean = " << std::get<0>(comparison_tuple) << " STD = " << std::get<1>(comparison_tuple)  << "Tot. diff. = " << std::get<2>(comparison_tuple) << std::endl;
    if (print_samples){
      std::cout << "Samples:\n";
      std::cout << j1[0] << " " << j2[0] << std::endl;
      std::cout << j1[10] << " " << j2[10] << std::endl;
      std::cout << j1[100] << " " << j2[100] << std::endl;
      std::cout << j1[1000] << " " << j2[1000] << std::endl;
      std::cout << j1[10000] << " " << j2[10000] << std::endl;
    }
  }
  return comparison_tuple;
}

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
