#include "sparsebase/reorder/gray_reorder.h"

#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/logger.h"

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
GrayReorder<IDType, NNZType, ValueType>::GrayReorder(
    BitMapSize resolution, int nnz_threshold, int sparse_density_group_size) {
  auto params_struct = new GrayReorderParams;
  params_struct->resolution = resolution;
  params_struct->nnz_threshold = nnz_threshold;
  params_struct->sparse_density_group_size = sparse_density_group_size;
  this->params_ = std::unique_ptr<GrayReorderParams>(params_struct);

  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GrayReorderingCSR);
}
template <typename IDType, typename NNZType, typename ValueType>
GrayReorder<IDType, NNZType, ValueType>::GrayReorder(GrayReorderParams p)
    : GrayReorder(p.resolution, p.nnz_threshold, p.sparse_density_group_size) {}

template <typename IDType, typename NNZType, typename ValueType>
bool GrayReorder<IDType, NNZType, ValueType>::desc_comparator(
    const row_grey_pair &l, const row_grey_pair &r) {
  return l.second > r.second;
}

template <typename IDType, typename NNZType, typename ValueType>
bool GrayReorder<IDType, NNZType, ValueType>::asc_comparator(
    const row_grey_pair &l, const row_grey_pair &r) {
  return l.second < r.second;
}

template <typename IDType, typename NNZType, typename ValueType>
// not sure if all IDTypes work for this
unsigned long long GrayReorder<IDType, NNZType, ValueType>::grey_bin_to_dec(
    unsigned long long n) {
  unsigned long long inv = 0;

  for (; n; n = n >> 1) {
    inv ^= n;
  }
  return inv;
}

template <typename IDType, typename NNZType, typename ValueType>
void GrayReorder<IDType, NNZType, ValueType>::print_dec_in_bin(unsigned long n,
                                                               int size) {
  // array to store binary number
  int binaryNum[size];

  // counter for binary array
  int i = 0;
  while (n > 0) {
    // storing remainder in binary array
    binaryNum[i] = n % 2;
    n = n / 2;
    i++;
  }

  // printing binary array in reverse order
  std::string bin_nums = "";
  for (int j = i - 1; j >= 0; j--)
    bin_nums = bin_nums + std::to_string(binaryNum[j]);
  utils::Logger l(typeid(GrayReorder));
  l.Log(bin_nums, utils::LogLevel::LOG_LVL_INFO);
}

// not sure if all IDTypes work for this
template <typename IDType, typename NNZType, typename ValueType>
unsigned long GrayReorder<IDType, NNZType, ValueType>::bin_to_grey(
    unsigned long n) {
  /* Right Shift the number by 1
  taking xor with original number */
  return n ^ (n >> 1);
}
template <typename IDType, typename NNZType, typename ValueType>
bool GrayReorder<IDType, NNZType, ValueType>::is_banded(
    int nnz, int n_cols, NNZType *row_ptr, IDType *cols,
    std::vector<IDType> order, int band_size) {
  if (band_size == -1) band_size = n_cols / 64;
  int band_count = 0;
  int nnz_ = 0;
  bool banded = false;

  for (int r = 0; r < order.size(); r++) {
    for (int i = row_ptr[order[r]]; i < row_ptr[order[r] + 1]; i++) {
      int col = cols[i];
      nnz_++;
      if ((col >= r) ? (col - r <= band_size) : (r - col <= band_size)) band_count++;
    }
  }
  
  if (double(band_count) / nnz_ >= 0.3) {
    banded = true;
  }
  utils::Logger logger(typeid(GrayReorder));
  logger.Log("NNZ % in band: " + std::to_string(double(band_count) / nnz),
             utils::LogLevel::LOG_LVL_INFO);
  return banded;
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *GrayReorder<IDType, NNZType, ValueType>::GrayReorderingCSR(
    std::vector<format::Format *> input_sf, utils::Parameters *poly_params) {
  auto csr = input_sf[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  context::CPUContext *cpu_context =
      static_cast<context::CPUContext *>(csr->get_context());

  IDType n_rows = csr->get_dimensions()[0];
  IDType n_cols = csr->get_dimensions()[1];
  
  /*This array stores the permutation vector such as order[0] = 243 means that
   * row 243 is the first row of the reordered matrix*/
  IDType *order = new IDType[n_rows]();

  GrayReorderParams *params = static_cast<GrayReorderParams *>(poly_params);
  int group_size = params->sparse_density_group_size;
  int bit_resolution = params->resolution;

  int raise_to = 0;
  unsigned long long adder = 0;
  int start_split_reorder, end_split_reorder;

  int last_row_nnz_count = 0;
  int threshold = 0;  // threshold used to set a bit in bitmap to 1
  bool decresc_grey_order = false;

  int group_count = 0;
 
  /*added*/
  int sparse_diagonal = 0;
  int dense_diagonal = 0;
  int nnz_sparse = 0;
  int nnz_dense = 0;
  int band_size = csr->get_dimensions()[1] / 128;

  // Initializing row order
  std::vector<IDType> v_order;
  std::vector<IDType> sparse_v_order;
  std::vector<IDType> dense_v_order;

  sparse_v_order.reserve(n_rows);
  dense_v_order.reserve(n_rows);

  // Splitting original matrix's rows in two submatrices
  IDType sparse_dense_split = 0;
  for (IDType i = 0; i < n_rows; i++) {
    if ((csr->get_row_ptr()[i + 1] - csr->get_row_ptr()[i]) <=
        params->nnz_threshold) {
      sparse_v_order.push_back(i);
      sparse_dense_split++;
      /* Counts how many non-zeros are in the sparse sub-matrix diagonal */
      for (int j = csr->get_row_ptr()[i]; j < csr->get_row_ptr()[i + 1]; j++) {
        int col = csr->get_col()[j];
        nnz_sparse++;
        if ((col >= i) ? (col - i <= band_size) : (i - col <= band_size)) sparse_diagonal++;
      }
    } else {
      dense_v_order.push_back(i);
      /* Counts how many non-zeros are in the dense sub-matrix diagonal */
      for (int j = csr->get_row_ptr()[i]; j < csr->get_row_ptr()[i + 1]; j++) {
        int col = csr->get_col()[j];
        nnz_dense++;
        if ((col >= i) ? (col - i <= band_size) : (i - col <= band_size)) dense_diagonal++;
      }
    }
  }
  v_order.reserve(sparse_v_order.size() +
                  dense_v_order.size());  // preallocate memory

  utils::Logger logger(typeid(GrayReorder));

  /* ADDED */
  /* Checks of sub-matrices are highly banded or not */
  bool is_sparse_banded;
  bool is_dense_banded;

  if (double(sparse_diagonal) / nnz_sparse > 0.3) {
    is_sparse_banded = true;
  } else {
    is_sparse_banded = false;
  }
  if (double(dense_diagonal) / nnz_dense > 0.2) {
    is_dense_banded = true;
  } else {
    is_dense_banded = false;
  }

  if (is_sparse_banded)
    logger.Log("Sparse Sub-Matrix highly banded - Performing just density reordering",
        utils::LogLevel::LOG_LVL_INFO);
  if (is_dense_banded)
    logger.Log("Dense Sub-Matrix highly banded - Maintaining structure",
               utils::LogLevel::LOG_LVL_INFO);

  std::sort(sparse_v_order.begin(), sparse_v_order.end(),
            [&](int i, int j) -> bool {
              return (csr->get_row_ptr()[i + 1] - csr->get_row_ptr()[i]) <
                     (csr->get_row_ptr()[j + 1] - csr->get_row_ptr()[j]);
            });  // reorder sparse matrix into nnz amount

  // the bit resolution determines the width of the bitmap of each row
  if (n_cols < bit_resolution) {
    bit_resolution = n_cols;
  }

  int row_split = n_cols / bit_resolution;

  auto nnz_per_row_split = new IDType[bit_resolution];
  auto nnz_per_row_split_bin = new IDType[bit_resolution];

  unsigned long long decimal_bit_map = 0;
  unsigned long long dec_begin = 0;
  int dec_begin_ind = 0;

  std::vector<row_grey_pair>
      reorder_section;  // vector that contains a section to be reordered

  reorder_section.reserve(n_rows);
  if (!is_sparse_banded) {  // if banded just row ordering by nnz count is
    // enough, else do bitmap reordering in groups
    for (int i = 0; i < sparse_v_order.size();
         i++) {  // sparse sub matrix if not highly banded
      if (i == 0) {
        last_row_nnz_count =
            csr->get_row_ptr()[sparse_v_order[i] + 1] -
            csr->get_row_ptr()[sparse_v_order[i]];  // get nnz count in first
        // row
        start_split_reorder = 0;
      }  // check if nnz amount changes from last row

      if ((csr->get_row_ptr()[sparse_v_order[i] + 1] -
           csr->get_row_ptr()[sparse_v_order[i]]) ==
          0) {  // for cases where rows are empty
        start_split_reorder = i + 1;
        last_row_nnz_count = csr->get_row_ptr()[sparse_v_order[i + 1] + 1] -
                             csr->get_row_ptr()[sparse_v_order[i + 1]];
        continue;
      }

      // reset bitmap for this row
      for (int j = 0; j < bit_resolution; j++) nnz_per_row_split[j] = 0;
      for (int j = 0; j < bit_resolution; j++) nnz_per_row_split_bin[j] = 0;

      // get number of nnz in each bitmap section
      for (int k = csr->get_row_ptr()[sparse_v_order[i]];
           k < csr->get_row_ptr()[sparse_v_order[i] + 1]; k++) {
        nnz_per_row_split[csr->get_col()[k] / row_split]++;
      }

      // get bitmap of the row in decimal value (first rows are less significant
      // bits)

      decimal_bit_map = 0;
      for (int j = 0; j < bit_resolution; j++) {
        adder = 0;
        if (nnz_per_row_split[j] > threshold) {
          nnz_per_row_split_bin[j] = 1;
          raise_to = j;
          adder = pow(2, raise_to);

          decimal_bit_map = decimal_bit_map + adder;
        }
      }

      // if number of nnz changed from last row, increment group count, which
      // might trigger a reorder of the group
      if ((i != 0) &&
          (last_row_nnz_count != (csr->get_row_ptr()[sparse_v_order[i] + 1] -
                                  csr->get_row_ptr()[sparse_v_order[i]]))) {
        group_count = group_count + 1;
        logger.Log("Rows[" + std::to_string(start_split_reorder) + " -> " +
                       std::to_string(i - 1) +
                       "] NNZ Count: " + std::to_string(last_row_nnz_count),
                   utils::LogLevel::LOG_LVL_INFO);
        // update nnz count for current row
        last_row_nnz_count = csr->get_row_ptr()[sparse_v_order[i] + 1] -
                             csr->get_row_ptr()[sparse_v_order[i]];

        // if group size achieved, start reordering section until this row
        if (group_count == group_size) {
          end_split_reorder = i;
          logger.Log("Reorder Group[" + std::to_string(start_split_reorder) +
                         " -> " + std::to_string(end_split_reorder - 1) + "]",
                     utils::LogLevel::LOG_LVL_INFO);
          // start next split the split for processing

          // process and reorder the reordered_matrix array till this point
          // (ascending or descending alternately)
          if (!decresc_grey_order) {
            sort(reorder_section.begin(), reorder_section.end(),
                 asc_comparator);
            decresc_grey_order = !decresc_grey_order;
          } else {
            sort(reorder_section.begin(), reorder_section.end(),
                 desc_comparator);
            decresc_grey_order = !decresc_grey_order;
          }

          dec_begin = reorder_section[0].second;
          dec_begin_ind = start_split_reorder;

          // apply reordered
          for (int a = start_split_reorder; a < end_split_reorder; a++) {
            if ((dec_begin !=
                 reorder_section[a - start_split_reorder].second) &&
                (a < 100000)) {
              logger.Log("Rows[" + std::to_string(dec_begin_ind) + " -> " +
                             std::to_string(a) + "] Grey Order: " +
                             std::to_string(dec_begin) + "// Binary:",
                         utils::LogLevel::LOG_LVL_INFO);
              // print_dec_in_bin(bin_to_grey(dec_begin));

              dec_begin = reorder_section[a - start_split_reorder].second;
              dec_begin_ind = a;
            }

            sparse_v_order[a] = reorder_section[a - start_split_reorder].first;
          }

          start_split_reorder = i;

          reorder_section.clear();
          group_count = 0;
        }
      }
      
      // if(decimal_bit_map != 0){
      //   for(int i = 0; i < bit_resolution; i++){
      //     std::cout << "[" << nnz_per_row_split_bin[(bit_resolution-1)-i] <<
      //     "]";
      //   }
      //     std::cout << "\nRow "<< i << "[" << v_order[i] << "] grey value: "
      //     << decimal_bit_map << " translates to: "<<
      //     grey_bin_to_dec(decimal_bit_map) <<"\n";
      // }

      //
      
      reorder_section.push_back(
          row_grey_pair(sparse_v_order[i], grey_bin_to_dec(decimal_bit_map)));


      // when reaching end of sparse submatrix, reorder section
      if (i == sparse_v_order.size() - 1) {
        end_split_reorder = sparse_v_order.size();
        logger.Log("Rows[" + std::to_string(start_split_reorder) + " -> " +
                       std::to_string(end_split_reorder - 1) +
                       "] NNZ Count: " + std::to_string(last_row_nnz_count),
                   utils::LogLevel::LOG_LVL_INFO);
        if (!decresc_grey_order) {
          sort(reorder_section.begin(), reorder_section.end(), asc_comparator);
          decresc_grey_order = !decresc_grey_order;
        } else {
          sort(reorder_section.begin(), reorder_section.end(), desc_comparator);
          decresc_grey_order = !decresc_grey_order;
        }
        for (int a = start_split_reorder; a < end_split_reorder; a++) {
          sparse_v_order[a] = reorder_section[a - start_split_reorder].first;
        }
      }
    }
    
    reorder_section.clear();
  }
  if (!is_dense_banded) {
    logger.Log("Rows [" + std::to_string(sparse_dense_split) + "-" +
                   std::to_string(n_rows) +
                   "] Starting Dense Sorting through NNZ and Grey code..",
               utils::LogLevel::LOG_LVL_INFO);

    for (int i = 0; i < dense_v_order.size(); i++) {
      // if first row, establish the nnz amount, and starting index
      for (int j = 0; j < bit_resolution; j++) nnz_per_row_split[j] = 0;
      for (int j = 0; j < bit_resolution; j++) nnz_per_row_split_bin[j] = 0;

      for (int k = csr->get_row_ptr()[dense_v_order[i]];
           k < csr->get_row_ptr()[dense_v_order[i] + 1]; k++) {
        nnz_per_row_split[csr->get_col()[k] / row_split]++;
      }
      threshold = (csr->get_row_ptr()[dense_v_order[i] + 1] -
                   csr->get_row_ptr()[dense_v_order[i]]) /
                  bit_resolution;  // floor
      decimal_bit_map = 0;
      for (int j = 0; j < bit_resolution; j++) {
        adder = 0;
        if (nnz_per_row_split[j] > threshold) {
          nnz_per_row_split_bin[j] = 1;
          raise_to = j;  // row 0 = lowest significant bit
          adder = pow(2, raise_to);

          decimal_bit_map = decimal_bit_map + adder;
        }
      }

      reorder_section.push_back(
          row_grey_pair(dense_v_order[i], grey_bin_to_dec(decimal_bit_map)));
    }
    logger.Log("Reordering Rows based on grey values...",
               utils::LogLevel::LOG_LVL_INFO);
    std::sort(reorder_section.begin(), reorder_section.end(), asc_comparator);

    for (int a = 0; a < dense_v_order.size(); a++) {
      dense_v_order[a] = reorder_section[a].first;
    }

    reorder_section.clear();
  }
  v_order.insert(v_order.end(), sparse_v_order.begin(), sparse_v_order.end());
  v_order.insert(v_order.end(), dense_v_order.begin(), dense_v_order.end());

  /*This order array stores the inverse permutation vector such as order[0] =
   * 243 means that row 0 is placed at the row 243 of the reordered matrix*/
  // std::vector<IDType> v_order_inv(n_rows);
  for (int i = 0; i < n_rows; i++) {
    order[v_order[i]] = i;
  }

  // std::copy(v_order_inv.begin(), v_order_inv.end(), order);
  return order;
}

#if !defined(_HEADER_ONLY)
#include "init/gray_reorder.inc"
#endif
}  // namespace sparsebase::reorder
