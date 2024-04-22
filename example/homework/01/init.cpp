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

  //actual homework problem
  int M = 14;
  int x_size = M/P + ((rank_col < (M % P)) ? 1 : 0);
  cout << "rank_col: " << rank_col << " | x_size: " << x_size << endl;
  vector<int> x_sub(x_size);
  
  if (rank_row == 0 && rank_col == 0){
    vector<int> x(M);

    // Fill the vector with numbers 1 through 14
    for (int i = 0; i < M; ++i) {
        x[i] = i + 1;
    }

    // Print the vector
   // std::cout << "Vector content:\n";
    for (int num : x) {
        //cout << num << " ";
    } 


int *recvarray = new int[recvcount];

cout << rank << ": Before Scatterv" << endl;

// Assuming you have defined and initialized array, sendcounts, and displs correctly
MPI_Scatterv(array, sendcounts, displs, MPI_INT, recvarray, recvcount,
             MPI_INT, 0, MPI_COMM_WORLD);

  }
cout << rank << ": After Scatterv; recvcount = " << recvcount << endl;

for(int i = 0; i < recvcount; ++i)
    cout << rank << ": recvarray[" << i << "] = " << recvarray[i] << endl;




  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);

  MPI_Finalize();
  return 0;
}
