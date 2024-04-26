class LinearDistribution
{
public:

  LinearDistribution(int _P, int _M) : the_P(_P), the_M(_M)
                                       {nominal = the_M/the_P;
					extra   = the_M%the_P;
					factor1 = extra*(nominal+1);}
  virtual ~LinearDistribution() {}
  
  void global_to_local(int I, int &p, int &i) const
              {p = (I < factor1) ? I/(nominal+1) : extra+((I-factor1)/nominal);
		i = I - ((p < extra) ? p*(nominal+1) : (factor1+(p-extra)*nominal));}
  int local_to_global(int p, int i) const
  {return i + ((p < extra) ? p*(nominal+1) : (factor1+(p-extra)*nominal));}
					      
  int m(int p) const {return (p < extra) ? (nominal+1) : nominal;}
  
  int M() const {return the_M;}
  int P() const {return the_P;}

protected:
  int the_M, the_P, nominal, extra, factor1;
};


class ScatterDistribution
{
public:

  ScatterDistribution(int _P, int _M) : the_P(_P), the_M(_M)
  {nominal = the_M/the_P; extra = the_M%the_P;}

  virtual ~ScatterDistribution() {}

  void global_to_local(int I, int &p, int &i) const {p = I%the_P; i = I/the_P;}
  int local_to_global(int p, int i) const {return i*the_P+p;}

  int m(int p) const {return (p < extra) ? (nominal+1) : nominal;}
  
  int M() const {return the_M;}
  int P() const {return the_P;}

protected:
  int the_M, the_P, nominal, extra;
};

