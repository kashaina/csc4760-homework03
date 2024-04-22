#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  // problem 5 from homework 2
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int Q = 5;
  int P = 3;
  if (rank == 0){
    cout << "\nWorld size: " << size;
    cout << "\nQ: 5\n";
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

  //PROBLEM 1


  int M = 14;
  int x_size = M/size_col + ((rank_col < (M % size_col)) ? 1 : 0);
  vector<int> x_sub(x_size);
  
  //scatter x across col 0
  if (rank_row == 0){
    std::vector<int> x_sendcounts(size_col);
    std::vector<int> x_displs(size_col);

    //calculate size of each sub-array
    for (int i = 0; i < size_col; ++i) {
      x_sendcounts[i] = M / size_col + ((i < (M % size_col)) ? 1 : 0);
    }

    //calculate displacement (what index the sub-array starts)
    int x_count = x_displs[0] = 0;
    for(int i = 1; i < size_col; ++i)
    {
      x_count += x_sendcounts[i-1];
      x_displs[i] = x_count;
    }

    //define x, and populate from 1-14 in (0, 0)
    std::vector<int> x(M);
    if (rank_col == 0) {
        for (int i = 0; i < M; ++i) {
            x[i] = i + 1;
        }
    }

    //scatter x across col 0
    x_sub.resize(x_sendcounts[rank_col]);
    MPI_Scatterv(&x[0], &x_sendcounts[0], &x_displs[0], MPI_INT, &x_sub[0], x_size, MPI_INT, 0, comm_col);
  }

  MPI_Bcast(&x_sub[0], x_size, MPI_INT, 0, comm_row);

  //print the received portion of vector x_sub on each process in the row communicator
  /*std::cout << "(" << rank_col << ", " << rank_row << ") received: ";
  for (int i = 0; i < x_size; i++) {
    std::cout << x_sub[i] << " ";
  }
  std::cout << std::endl;*/
  

  int y_size = M/size_row + ((rank_row < (M % size_row)) ? 1 : 0);
  vector<int> y_sub(y_size);
  //std::cout << "(" << rank_col << ", " << rank_row << ") row size: " << y_size << endl;
  for (int i = 0; i < y_size; ++i){
    for (int j = 1; j <= x_size; ++1_{
      //iif (size_row * rank_col + j
  }


  //calculate size of each sub-array
    for (int i = 0; i < size_col; ++i) {
      x_sendcounts[i] = M / size_col + ((i < (M % size_col)) ? 1 : 0);
    }

    //calculate displacement (what index the sub-array starts)
    int x_count = x_displs[0] = 0;
    for(int i = 1; i < size_col; ++i)
    {
      x_count += x_sendcounts[i-1];
      x_displs[i] = x_count;
    }





  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);

  MPI_Finalize();
  return 0;
}
