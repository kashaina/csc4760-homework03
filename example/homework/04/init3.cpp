
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

vector<int> vectorDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M);
vector<vector<int>> matrixDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, int N);

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
    cout << "\nQ: " << Q;
    cout << "\nP: " << P;
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


  int M = 7;
  int N = 6;

  //**create vector and matrix**

  // create a vector (numbers 0-63 even) and distribute it in horizontal linear load balance
  vector<int> vector1 = vectorDistribute(comm_row, comm_col, rank_row, rank_col, size_row, size_col, M);  
  MPI_Barrier(MPI_COMM_WORLD);

  // create a 7 x 6 matrix (numbers 0-41) and slice it into m x n portions  
  vector<vector<int>> matrix = matrixDistribute(comm_row, comm_col, rank_row, rank_col, size_row, size_col, M, N);
  MPI_Barrier(MPI_COMM_WORLD);

  //**calculate global y displacements**
  /*int y_size = N/size_col + ((rank_col < (N % size_col)) ? 1 : 0);//only edited col
  int y_displs;
  int y_displs_temp;
  vector<int> y(5, 0);
  vector<int> y_displs_array(size_col);

  if (rank_row == 0){
    vector<int> y_sizes_array(size_col);

    // calculate size of each sub-array
    for (int i = 0; i < size_col; ++i) {
      y_sizes_array[i] = N / size_col + ((i < (N % size_col)) ? 1 : 0);
    }

    // calculate displacement (what index the sub-array starts)
    int y_count = y_displs_array[0] = 0;
    for(int i = 1; i < size_col; ++i){
      y_count += y_sizes_array[i-1];
      y_displs_array[i] = y_count;
    }
  }
  MPI_Scatter(&y_displs_array[0], 1, MPI_INT, &y_displs, 1, MPI_INT, 0, comm_col);
  MPI_Bcast(&y_displs, 1, MPI_INT, 0, comm_row);
  //cout << "Matrix | (" << rank_col << ", " << rank_row << ") | " << y_displs << endl;

  for (int i = 0; i < matrix.size(); i++){
    for (int j = 0; j < matrix[i].size(); j++){
      y[i + y_displs] += matrix[i][j] * vector1[j];
      cout << "(" << rank_col << ", " << rank_row << ") | " << i + y_displs << " = " << matrix[i][j] << " * " << vector1[j] << " = " << (matrix[i][j] * vector1[j]) << endl;
    }
  }*/
  
  //**compute y:= Ax**
  // compute the local dot product between a local row and and local vector. repeat for each row in a process
  vector<int> y_sub(matrix[0].size(), 0);
  for (int i = 0; i < matrix.size(); i++){
    for (int j = 0; j < matrix[i].size(); j++){
      y_sub[i] += matrix[i][j] * vector1[j];
      //cout << "(" << rank_col << ", " << rank_row << ") | " << i + y_displs << " = " << matrix[i][j] << " * " << vector1[j] << " = " << (matrix[i][j] * vector1[j]) << endl;
    }
  }

  // do allreduce for each column to add local dot products for each y index
  vector<int> y(matrix.size());
  MPI_Allreduce(&y_sub[0], &y[0], N, MPI_INT, MPI_SUM, comm_row);
  
  // print dot product
  cout << "Y - Dot Product | (" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < y.size(); ++i) {
        cout << y[i] << " ";
    }
  cout << endl;
  MPI_Barrier(MPI_COMM_WORLD);
  

  //**compute z:= Ay**
  // compute the local dot product between a local row and and local vector. repeat for each row in a process
  vector<int> z_sub(matrix[0].size(), 0);
  for (int i = 0; i < matrix.size(); i++){
    for (int j = 0; j < matrix[i].size(); j++){
      z_sub[i] += matrix[i][j] * y[j];
      //cout << "(" << rank_col << ", " << rank_row << ") | " << i + y_displs << " = " << matrix[i][j] << " * " << vector1[j] << " = " << (matrix[i][j] * vector1[j]) << endl;
    }
  }

  // do allreduce for each column to add local dot products for each y index
  vector<int> z(matrix.size());
  MPI_Allreduce(&z_sub[0], &z[0], N, MPI_INT, MPI_SUM, comm_row);

  // print dot product
  cout << "Z (Extra Credit) | (" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < z.size(); ++i) {
        cout << z[i] << " ";
    }
  cout << endl;






  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);

  MPI_Finalize();
  return 0;
}


// create a vector (numbers 0-63 even) and distribute it in horizontal linear load balance
vector<int> vectorDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M){
  int y_size = M/size_row + ((rank_row < (M % size_row)) ? 1 : 0);
  vector<int> y_sub(y_size);
  int y_displs;

  // work with column 0 to develop one vector of size M and distribute it among this column then horizontal
  if (rank_col == 0){
    std::vector<int> y_sizes_array(size_row);
    std::vector<int> y_displs_array(size_row);

    // calculate size of each sub-array
    for (int i = 0; i < size_row; ++i) {
      y_sizes_array[i] = M / size_row + ((i < (M % size_row)) ? 1 : 0);
    }

    // calculate displacement (what index the sub-array starts)
    int y_count = y_displs_array[0] = 0;
    for(int i = 1; i < size_row; ++i)
    {
      y_count += y_sizes_array[i-1];
      y_displs_array[i] = y_count;
    }

    // define x and populate from 0, 2, ..., 64 in (0, 0)
    std::vector<int> y(M);
    if (rank_row == 0) {
      cout << "\nOriginal array " << ": ";
      for (int i = 0; i < M; ++i) {
        y[i] = i * 2;
	cout << y[i] << " ";
      }
      cout << endl;
    }

    // scatter x and displacements across column 0
    y_sub.resize(y_sizes_array[rank_row]);
    MPI_Scatterv(&y[0], &y_sizes_array[0], &y_displs_array[0], MPI_INT, &y_sub[0], y_size, MPI_INT, 0, comm_row);
  }

  // broadcast each x_sub from column 0 horizontally in each process row**
  MPI_Bcast(&y_sub[0], y_size, MPI_INT, 0, comm_col);

  // print the received portion of vector y_sub
  cout << "Vector Division | (" << rank_col << ", " << rank_row << ") | ";
  for (int i = 0; i < y_size; i++) {
    cout << y_sub[i] << " ";
  }
  cout << endl;

  return y_sub;
}


vector<vector<int>> matrixDistribute(MPI_Comm comm_row, MPI_Comm comm_col, int rank_row, int rank_col, int size_row, int size_col, int M, int N){
  //**create matrix and distribute in linear-load balanced among rows and columns**
  int x_size = M / size_row + ((rank_row < (M % size_row)) ? 1 : 0);
  vector<vector<int>> x_sub(N, vector<int>(x_size, 0));

  // work with column 0 to develop one vector of size M and distribute it among this column then vertically
  if (rank_col == 0){
    vector<int> x_sizes_array(size_row);
    vector<int> x_displs_array(size_row);
    vector<vector<int>> matrix(N, vector<int>(M, 0));

    // calculate size of each sub-array
    for (int i = 0; i < size_row; ++i) {
      x_sizes_array[i] = M / size_row + ((i < (M % size_row)) ? 1 : 0);
    }

    // calculate displacement (what index the sub-array starts)
    int x_count = x_displs_array[0] = 0;
    for(int i = 1; i < size_row; ++i){
      x_count += x_sizes_array[i-1];
      x_displs_array[i] = x_count;
    }

    // create and initialize matrix
    if (rank_row == 0){
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          matrix[i][j] = i * M + j;
        }
      }

      // print the elements of the matrix
      cout << "Original Matrix | " << endl;
      for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
          cout << matrix[i][j] << " ";
        }
        cout << endl;
      }
    }

    // scatter x and displacements across column 0
    for (int i = 0; i < N; i++){
      MPI_Scatterv(&matrix[i][0], &x_sizes_array[0], &x_displs_array[0], MPI_INT, &x_sub[i][0], x_size, MPI_INT, 0, comm_row);
    }
  }

  for (int i = 0; i < N; i++){
    MPI_Bcast(&x_sub[i][0], x_size, MPI_INT, 0, comm_col);
  }

  /*cout << "Matrix - Row | (" << rank_col << ", " << rank_row << ") | " << endl;
    for (int i = 0; i < x_sub.size(); ++i) {
      for (int j = 0; j < x_sub[i].size(); ++j) {
        cout << setw(2) << x_sub[i][j] << " ";
      }
     cout << endl;
  }*/
  
  //**slice x_sub by rows and keep needed row
  int y_size = N/size_col + ((rank_col < (N % size_col)) ? 1 : 0);//only edited col
  int y_displs;
  vector<int> y_displs_array(size_col);

  if (rank_row == 0){
    vector<int> y_sizes_array(size_col);

    // calculate size of each sub-array
    for (int i = 0; i < size_col; ++i) {
      y_sizes_array[i] = N / size_col + ((i < (N % size_col)) ? 1 : 0);
    }

    // calculate displacement (what index the sub-array starts)
    int y_count = y_displs_array[0] = 0;
    for(int i = 1; i < size_col; ++i){
      y_count += y_sizes_array[i-1];
      y_displs_array[i] = y_count;
    }
  }
  MPI_Scatter(&y_displs_array[0], 1, MPI_INT, &y_displs, 1, MPI_INT, 0, comm_col);
  MPI_Bcast(&y_displs, 1, MPI_INT, 0, comm_row);

  //cout << "Matrix | (" << rank_col << ", " << rank_row << ") | " << y_displs << endl;

  vector<vector<int>> y_sub(y_size, vector<int>(x_size, 0));
  for (int i = 0; i < y_sub.size(); ++i) {
    for (int j = 0; j < y_sub[i].size(); ++j) {
      y_sub[i][j] = x_sub[i + y_displs][j];
    }
  }

  cout << "Matrix Division | (" << rank_col << ", " << rank_row << ") | " << endl;
    for (int i = 0; i < y_sub.size(); ++i) {
      for (int j = 0; j < y_sub[i].size(); ++j) {
        cout << y_sub[i][j] << " ";
      }
     cout << endl;
  }
 
 
return y_sub;
}
