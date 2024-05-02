#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

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


  //**store vector x of length M, distributed in a linear load-balanced fashion**
  int M = 32;
  int x_size = M/size_col + ((rank_col < (M % size_col)) ? 1 : 0);
  vector<int> x_sub(x_size);

  // work with column 0 to develop one vector of size M and distribute it among this column then broadcast to the others
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

    // define x and populate from 0, 2, ... 62 in (0, 0)
    std::vector<int> x(M);
    if (rank_col == 0) {
      cout << "\nM: ";
      for (int i = 0; i < M; ++i) {
        x[i] = i * 2;
	cout << x[i] << " ";
      }
      cout << "\n\nFinal results:\n";
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


  // work with row 0 to develop one vector of size M to hold results and distribute it among this row
  int y_size;
  if(rank_col == 0){
    int *y_sizes_array = new int[size_row];

    // work with column 0 to develop y_sub arrays
    if (rank_row == 0){
      // calculate size of each sub-array
      for(int i = 0; i < size_row; ++i){
        y_sizes_array[i] = M / size_row + ((i < (M % size_row)) ? 1 : 0);
      }
    }

    // scatter sizes across row 0
    MPI_Scatter(y_sizes_array, 1, MPI_INT, &y_size, 1, MPI_INT, 0, comm_row);
  }

  // broadcast each y_sub from row 0 vertically in each process row
  MPI_Bcast(&y_size, 1, MPI_INT, 0, comm_col);
  

  //NEW!! PROBLEM 3 OF HOMEWORK 3
  //**conduct linear scatter distribution**
  vector<int> y_sub(y_size, 0);
  int extra1 = M % size_col;
  int nominal1 = M / size_col;
  int extra2 = M % size_row;
  int nominal2 = M / size_row;


  for(int i = 0; i < x_size; i++){
    // compute global index I
    int I = i + ((rank_col < extra1) ? (nominal1 + 1)* rank_col : (extra1*(nominal1 + 1)+(rank_col - extra1) * nominal1));

    // compute (qhat, jhat) index of element in global vector
    int qhat = I % Q;
    int jhat = I / Q;

    if(qhat == rank_row){ 
      y_sub[jhat] = x_sub[i];
    }
  }

  // print y_sub
  /*(cout << "(" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < y_size; i++){
    cout << y_sub[i] << " ";
  }
  cout << endl;*/

  // perform allreduce among each col to fill up each index on each result vector
  vector<int> result(y_size, 0);
  MPI_Allreduce(&y_sub[0], &result[0], y_size, MPI_INT, MPI_SUM, comm_col);

  // print final results
  cout << "(" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < y_size; i++) {
    cout << setw(2) << result[i] << " ";
  }
  cout << endl;



  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);

  MPI_Finalize();
  return 0;
}
