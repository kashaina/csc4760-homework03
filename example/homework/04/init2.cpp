#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

vector<int> linearDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, int a);
vector<int> computeYSub(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, vector<int> &x_sub, int x_size);
vector<int> scatterDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, int a);

int main(int argc, char *argv[]) {
  // PROBLEM 5 FROM HOMEWORK 2
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if(size != 15){
    if (rank == 0){ 
      cerr << "Please try again with 15 processes. Exiting." << endl << endl;
    } 
    MPI_Finalize();
    return 0;
  }

  int Q = 5;
  int P = 3;
  int M = 32;

  if (rank == 0){
    cout << "\nWorld size: " << size;
    cout << "\nQ: 5";
    cout << "\nP: 3";
  }

  // split communicator based on ranks divided by Q
  int color_row = rank / Q;
  MPI_Comm comm_row;
  MPI_Comm_split(MPI_COMM_WORLD, color_row, rank, &comm_row);
  
  // split communicator based on ranks mod Q
  int color_col = rank % Q;
  MPI_Comm comm_col;
  MPI_Comm_split(MPI_COMM_WORLD, color_col, rank, &comm_col);
  
  // get and print ranks and sizes of new communicators
  int rank_row, rank_col, size_row, size_col;
  MPI_Comm_rank(comm_row, &rank_row);
  MPI_Comm_rank(comm_col, &rank_col);
  MPI_Comm_size(comm_row, &size_row);
  MPI_Comm_size(comm_col, &size_col);

  //cout << "World rank: " << setw(2) << rank << "  |  Cell: (" <<  rank_col << ", " << rank_row << ")" << endl;


  //PROBLEM 1 FROM HOMEWORK 3

  //**do Problem 1 with vertical and horizontal loads, and print result**
  
  //**call functions to create vector of size M (numbers 0-63 (even for vector1 and odd for vector2) and distribute. Results in a sub-vector of the original vector that is sliced in each row. Both final results are the same
  // create vector using only linear-loaded distribution
  vector<int> vector1 = linearDistribute(comm_row, comm_col, rank_row, rank_col, size_row, size_col, M, 0);  
  MPI_Barrier(MPI_COMM_WORLD);
  cout << "Vector 1 - Linear | (" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < vector1.size(); i++) {
    cout << setw(2) << vector1[i] << " ";
  }
  cout << endl;
  MPI_Barrier(MPI_COMM_WORLD);
 

  // create vector initially in scatter distribution, then transposes it twice to be linear-loaded to be compatible with vector1
  vector<int> vector2 = scatterDistribute(comm_row, comm_col, rank_row, rank_col, size_row, size_col, M, 1);
  cout << "Vector 2 - Linear (Transposed) | (" << rank_col << ", " << rank_row << ") | : ";
  for (int i = 0; i < vector2.size(); i++) {
    cout << setw(2) << vector2[i] << " ";
  }
  cout << endl;
  MPI_Barrier(MPI_COMM_WORLD);

   //**compute dot product among each row**
  int local_dot_product = 0;
  for (size_t i = 0; i < vector1.size(); ++i) {
     local_dot_product += vector1[i] * vector2[i];
  }

  int global_dot_product;
  MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, comm_row); 
  MPI_Barrier(MPI_COMM_WORLD);
  cout << "(" << rank_col << ", " << rank_row << ") | Dot Product: " << global_dot_product << endl;


  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);

  MPI_Finalize();
  return 0;
}

// create vector using only linear-loaded distribution
vector<int> linearDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, int a){
  int x_size = M/size_col + ((rank_col < (M % size_col)) ? 1 : 0);
  vector<int> x_sub(x_size);
  int x_displs;

  // work with column 0 to develop one vector of size M and distribute it among this column then vertically
  if (rank_row == 0){
    std::vector<int> x_sizes_array(size_col);
    std::vector<int> x_displs_array(size_col);

    // calculate size of each sub-array
    for (int i = 0; i < size_col; ++i) {
      x_sizes_array[i] = M / size_col + ((i < (M % size_col)) ? 1 : 0);
    }

    // calculate displacement (what index the sub-array starts)
    int x_count = x_displs_array[0] = 0;
    for(int i = 1; i < size_col; ++i)
    {
      x_count += x_sizes_array[i-1];
      x_displs_array[i] = x_count;
    }

    // define x and populate from 0, 2, ..., 64 in (0, 0)
    std::vector<int> x(M);
    if (rank_col == 0) {
      cout << "\nOriginal array of vector " << a + 1 << ": ";
      for (int i = 0; i < M; ++i) {
        x[i] = i * 2;
	//x[i] = i + 1;
	cout << x[i] << " ";
      }
      cout << endl;
    }

    // scatter x and displacements across column 0
    MPI_Scatterv(&x[0], &x_sizes_array[0], &x_displs_array[0], MPI_INT, &x_sub[0], x_size, MPI_INT, 0, comm_col);
  }

  // broadcast each x_sub from column 0 horizontally in each process row**
  MPI_Bcast(&x_sub[0], x_size, MPI_INT, 0, comm_row);

  // print the received portion of vector x_sub
  /*cout << "(" << rank_col << ", " << rank_row << ") received: ";
  for (int i = 0; i < x_size; i++) {
    cout << x_sub[i] << " ";
  }
  cout << endl;
  */

  return computeYSub(comm_row, comm_col, rank_row, rank_col, size_row, size_col, M, x_sub, x_size);
}

// the only purpose of the function is so it is not repeated for both linearDistribute and scatterDistribute
// starts with x_sub vectors to produce a linear-loaded distribution
vector<int> computeYSub(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, vector<int> &x_sub, int x_size){
 // work with row 0 to develop one vector of size M to hold results and distribute it among this row then horizontally
  int y_size, y_displs;
  if(rank_col == 0){
    int *y_sizes_array = new int[size_row];
    int *y_displs_array = new int[size_row];

    // work with column 0 to develop y_sub arrays
    if (rank_row == 0){
      
      // calculate size of each sub-array
      for(int i = 0; i < size_row; ++i){
        y_sizes_array[i] = M / size_row + ((i < (M % size_row)) ? 1 : 0);
      }
      
      // calculate displacement (what index the sub-array starts)
      int count = y_displs_array[0] = 0;
      for(int i = 1; i < size_row; ++i){
        count += y_sizes_array[i-1];
        y_displs_array[i] = count;
      }
    }

    // scatter sizes and displacements across row 0
    MPI_Scatter(y_sizes_array, 1, MPI_INT, &y_size, 1, MPI_INT, 0, comm_row);
    MPI_Scatter(y_displs_array, 1, MPI_INT, &y_displs, 1, MPI_INT, 0, comm_row);
  }

  // broadcast each y_sub from row 0 vertically in each process row
  MPI_Bcast(&y_size, 1, MPI_INT, 0, comm_col);
  

  //**conduct forward linear load-balanced distribution**
  vector<int> y_sub(y_size, 0);
  int extra1 = M %size_col;
  int nominal1 = M/size_col;
  int extra2 = M %size_row;
  int nominal2 = M/size_row;
 
  for(int i = 0; i < x_size; i++){
    // compute global index I
    int I = i + ((rank_col < extra1) ? (nominal1 + 1) * rank_col : (extra1 * (nominal1 + 1) + (rank_col - extra1) * nominal1));

    // compute (qhat,jhat) of the element in the global array
    int qhat = (I < extra2 * (nominal2 + 1)) ? I / (nominal2 + 1) : extra2 + (I - extra2 * (nominal2 + 1)) / nominal2;
    int jhat = I - ((qhat < extra2) ? (nominal2 + 1) * qhat : (extra2 * (nominal2 + 1) + (qhat - extra2) * nominal2));

    if(qhat == rank_row){
      y_sub[jhat] = x_sub[i];
    }
  }

  // perform allreduce among each col to fill up each index on each result vecot
  vector<int> result(y_size, 0);
  MPI_Allreduce(&y_sub[0], &result[0], y_size, MPI_INT, MPI_SUM, comm_col);

  // print final results
  /*cout << "(" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < y_size; i++) {
    cout << setw(2) << result[i] << " ";
  }
  cout << endl;*/
 
  return result;
}


// create vector initially in scatter distribution, then transposes it twice to be linear-loaded to be compatible with vector1
vector<int> scatterDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, int a){
  int x_size = M/size_col + ((rank_col < (M % size_col)) ? 1 : 0);
  vector<int> x_sub(x_size);
  int x_displs;

  // work with column 0 to develop one vector of size M and distribute it among this column then vertically
  if (rank_row == 0){
    std::vector<int> x_sizes_array(size_col);
    std::vector<int> x_displs_array(size_col);

    // calculate size of each sub-array
    for (int i = 0; i < size_col; ++i) {
      x_sizes_array[i] = M / size_col + ((i < (M % size_col)) ? 1 : 0);
    }

    // calculate displacement (what index the sub-array starts)
    int x_count = x_displs_array[0] = 0;
    for(int i = 1; i < size_col; ++i)
    {
      x_count += x_sizes_array[i-1];
      x_displs_array[i] = x_count;
    }

    // define x and populate from 0, 2, ..., 64 in (0, 0)
    std::vector<int> x(M);
    if (rank_col == 0) {
      cout << "\nOriginal array of vector: " << a + 1 << ": ";
      for (int i = 0; i < M; ++i) {
        //x[i] = i + 1;
	x[i] = i * 2 + 1;
        cout << x[i] << " ";
      }
      cout << endl;
    }

    // scatter x and displacements across column 0
    x_sub.resize(x_sizes_array[rank_col]);
    MPI_Scatterv(&x[0], &x_sizes_array[0], &x_displs_array[0], MPI_INT, &x_sub[0], x_size, MPI_INT, 0, comm_col);
  }

  // broadcast each x_sub from column 0 horizontally in each process row**
  MPI_Bcast(&x_sub[0], x_size, MPI_INT, 0, comm_row);

  // print the received portion of vector x_sub
  /*cout << "(" << rank_col << ", " << rank_row << ") received: ";
  for (int i = 0; i < x_size; i++) {
    cout << x_sub[i] << " ";
  }
  cout << endl;*/

  // work with row 0 to develop one vector of size M to hold results and distribute it among this row then horizontally
  int y_size, y_displs;
  if(rank_col == 0){
    int *y_sizes_array = new int[size_row];
    int *y_displs_array = new int[size_row];

    // work with column 0 to develop y_sub arrays
    if (rank_row == 0){
      
      // calculate size of each sub-array
      for(int i = 0; i < size_row; ++i){
        y_sizes_array[i] = M / size_row + ((i < (M % size_row)) ? 1 : 0);
      }
      
      // calculate displacement (what index the sub-array starts)
      int count = y_displs_array[0] = 0;
      for(int i = 1; i < size_row; ++i){
        count += y_sizes_array[i-1];
        y_displs_array[i] = count;
      }
    }

    // scatter sizes and displacements across row 0
    MPI_Scatter(y_sizes_array, 1, MPI_INT, &y_size, 1, MPI_INT, 0, comm_row);
    MPI_Scatter(y_displs_array, 1, MPI_INT, &y_displs, 1, MPI_INT, 0, comm_row);
  }

  // broadcast each y_sub from row 0 vertically in each process row
  MPI_Bcast(&y_size, 1, MPI_INT, 0, comm_col);

  //**conduct two transpositions from scatter to linear**
  // transposition 1: convert scatter back to x_sub arrays  
  vector<int> y_sub(y_size, 0);
  int extra1 = M % size_col;
  int nominal1 = M / size_col;
  int extra2 = M % size_row;
  int nominal2 = M / size_row;

  for(int i = 0; i < x_size; i++){
    // compute global index I
    int I = i + ((rank_col < extra1) ? (nominal1 + 1)* rank_col : (extra1*(nominal1 + 1)+(rank_col - extra1) * nominal1));

    // compute (qhat, jhat) index of element in global vector
    int qhat = I % size_row;
    int jhat = I / size_row;

    if(qhat == rank_row){ 
      y_sub[jhat] = x_sub[i];
    }
  }

  // perform allreduce among each col to fill up each index on each result vector
  vector<int> result_scatter(y_size, 0);
  MPI_Allreduce(&y_sub[0], &result_scatter[0], y_size, MPI_INT, MPI_SUM, comm_col);

  // print normal  distribution
  /*cout << "Vector 2 - Normal | " << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < y_size; i++) {
    cout << setw(2) << y_sub[i] << " ";
  }
  cout << endl;
  MPI_Barrier(MPI_COMM_WORLD);*/

  //**transposition 1: convert scatter to x_sub arrays**
  vector<int> new_x_sub(x_size, 0);
  for (int i = 0; i < x_size; i++) {
    int I = i + ((rank_col < extra1) ? (nominal1 + 1) * rank_col : (extra1 * (nominal1 + 1) + (rank_col - extra1) * nominal1));
    int qhat = I % size_row;
    int jhat = I / size_row;
    if (qhat == rank_row) {
        new_x_sub[i] = y_sub[jhat];
    }
  }

  vector<int> result_transpose1(x_size, 0);
  MPI_Allreduce(&new_x_sub[0], &result_transpose1[0], x_size, MPI_INT, MPI_SUM, comm_row);
  MPI_Barrier(MPI_COMM_WORLD);

  // print new_x_sub
  cout << "Vector 2 - Scattered (Original) | (" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < x_size; i++){ 
    cout << setw(2) << result_transpose1[i] << " ";
  }
  cout << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  //**transposition 2: convert x_sub arrays to linear-loaded**
  return computeYSub(comm_row, comm_col, rank_row, rank_col, size_row, size_col, M, result_transpose1, x_size);
}
