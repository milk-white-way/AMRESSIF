#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Gpu.H>

using namespace amrex;

int main (int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   {
      Print() << "Hello World! This is AMReX version " << amrex::Version() << "\n";
      Vector<Real> rk(4);
      {
	 rk[0] = Real(1.0)/Real(4.0);
	 rk[1] = Real(1.0)/Real(3.0);
	 rk[2] = Real(1.0)/Real(2.0);
	 rk[3] = 1.0;
      }
      for ( int n = 0; n < 4; ++n ){
	 // rk[n] = rk[n]*Real(40.0);
	 Print() << "Print vector " << rk[n] << "\n";
      }
   }
   amrex::Finalize();

   return 0;
}
