#include <iostream>

#include "sparsebase/format/format.h"
#include "sparsebase/utils/converter/converter.h"

using namespace std;
using namespace sparsebase;

int main() {

  int row[6] = {0, 0, 1, 1, 2, 2};
  int col[6] = {0, 1, 1, 2, 3, 3};
  int vals[6] = {10, 20, 30, 40, 50, 60};

  // Conversion Syntax 1
  context::CPUContext cpu_context;
  format::COO<int, int, int> *coo =
      new format::COO<int, int, int>(6, 6, 6, row, col, vals);
  auto csr = coo->Convert<format::CSR>(&cpu_context);
  auto csr2 = csr->Convert<format::CSR>(&cpu_context);

  auto dims = csr2->get_dimensions();
  int n = dims[0];
  int m = dims[1];
  int nnz = csr->get_num_nnz();

  cout << "CSR" << endl;

  for (int i = 0; i < nnz; i++)
    cout << csr2->get_vals()[i] << ",";
  cout << endl;

  for (int i = 0; i < nnz; i++)
    cout << csr2->get_col()[i] << ",";
  cout << endl;

  for (int i = 0; i < n + 1; i++)
    cout << csr2->get_row_ptr()[i] << ",";
  cout << endl;

  cout << endl;

  // Conversion Syntax 2
  auto coo2 = csr2->Convert<format::COO>(&cpu_context);
  cout << "COO" << endl;

  for (int i = 0; i < nnz; i++)
    cout << coo2->get_vals()[i] << ",";
  cout << endl;

  for (int i = 0; i < nnz; i++)
    cout << coo2->get_row()[i] << ",";
  cout << endl;

  for (int i = 0; i < nnz; i++)
    cout << coo2->get_col()[i] << ",";
  cout << endl;

  delete coo;
  delete csr2;
}
