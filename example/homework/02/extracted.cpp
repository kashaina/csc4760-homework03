//partial number 2 using extracted.cpp, as instructed


//modified exterior to be a View and interior to be a subview
class Domain // this will be integrated with a Kokkos view and subview in HW#3.
{            // this is a row-major arrangement of data in 2D.
public:
  Domain(int _M, int _N, int _halo_depth, const char *_name="") :
        exterior("exterior", (_M + 2 * _halo_depth) * (_N + 2 * _halo_depth)),
        interior(Kokkos::subview(exterior, Kokkos::make_pair(_halo_depth, _halo_depth + _M),
                                  Kokkos::make_pair(_halo_depth, _halo_depth + _N))),
        M(_M), N(_N), halo_depth(_halo_depth), name(_name) {}

  virtual ~Domain() {delete[] exterior;}

  char &operator()(int i, int j)       {return interior[i*(N+2*halo_depth)+j];}
  char operator() (int i, int j) const {return interior[i*(N+2*halo_depth)+j];}

  int rows() const {return M;}
  int cols() const {return N;} // actual domain size.

  const string &myname() const {return name;}

  char *rawptr()    {return exterior;}
  char *cookedptr() {return interior;}

protected:
  char *exterior, *interior;
  int M;
  int N;
  int halo_depth;

  string name;
};

//modified update_domain to use parallel_for instead of serial for-loops and if-else statements
{
  int m = old_domain.rows();
  int n = old_domain.cols();

  Kokkos::parallel_for("update_domain", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, n}),KOKKOS_LAMBDA(int i, int j) {
    int neighbor_count =
        old_domain.exterior[(i - 1) * (n + 2 * old_domain.halo_depth) + (j - 1)] +
        old_domain.exterior[(i - 1) * (n + 2 * old_domain.halo_depth) + j] +
        old_domain.exterior[(i - 1) * (n + 2 * old_domain.halo_depth) + (j + 1)] +
        old_domain.exterior[i * (n + 2 * old_domain.halo_depth) + (j - 1)] + 0 +
        old_domain.exterior[i * (n + 2 * old_domain.halo_depth) + (j + 1)] +
        old_domain.exterior[(i + 1) * (n + 2 * old_domain.halo_depth) + (j - 1)] +
        old_domain.exterior[(i + 1) * (n + 2 * old_domain.halo_depth) + j] +
        old_domain.exterior[(i + 1) * (n + 2 * old_domain.halo_depth) + (j + 1)];

    new_domain.interior(i, j) = update_the_cell(old_domain.interior(i, j), neighbor_count);
  });
  Kokkos::fence();
}

