/** 
 * Incompressible Flow Solver
 * Testing .... 

 * 
 * Trung Edited
 * Jul/31/2023 - Trung edited
 */

// =================== LISTING KERNEL HEADERS ==============================
#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>

#include "main.H"

// Modulization library
#include "utilities.H"
#include "fn_init.H"
#include "fn_enforce_wall_bcs.H"
#include "fn_flux_calc.H"

// Default library
#include "myfunc.H"
#include "momentum.H"
#include "poisson.H"

using namespace amrex;

/*
 * This is the solver context, which stores all the information about 
 * the solver and the associated grid
 */
struct amress_solver
    {
      // Grid information
      int n_cell;
      int max_grid_size;

      int IterNum;


      // Plotting results variables
      int plot_int;
      int nsteps;

      Real cfl;
      Real ren;
      Real vis;

      //      int phy_bc_lo[AMREX_SPACEDIM];
      //      int phy_bc_hi[AMREX_SPACEDIM];

      // // Declaring params for boundary conditon type
      int bc_lo[AMREX_SPACEDIM];
      int bc_hi[AMREX_SPACEDIM];

      int is_periodic[AMREX_SPACEDIM];

      // The declaration for the box domain
      BoxArray ba;
      Geometry geom;
      Box domain;

      // Ghost points and computational components
      // Nghost = number of ghost cells for each array
      int Nghost = 2; // 2nd order accuracy scheme is used for convective terms

      // Ncomp = number of components for each array
      // The userCtx has 2 components: pressure and phi
      int Ncomp  = 2;

      
    // How Boxes are distrubuted among MPI processes
    // Distribution mapping between the processors
    // This is for volume-center variables
    DistributionMapping dm;


    // User Contex MultiFab contains 2 components, pressure and Phi, at the cell center
    MultiFab userCtx;
    MultiFab userCtxPrev;


    };

// ============================== MAIN SECTION ==============================//
/**
 * This is the code using AMReX for solving Navier-Stokes equation using
 * hybrid staggerred/non-staggered method
 * Note that the Contravariant variables stay at the face center
 * The pressure and Cartesian velocities are in the volume center
 */
int main (int argc, char* argv[])
{
    // Initialization
    //    amrex::Print() << "Code initialization .... " <<  "\n";

    amrex::Initialize(argc,argv);
    // Solver
    main_main();
    // Finalization
    amrex::Finalize();
    // Error handeling
    return 0;
}

/* 
 *------------------------------------------------
 *------ This function reads the inputs file -----
 *----and set the parameters for the simulation---
 *------------------------------------------------
 */

void Input_Parameters(amress_solver *SolverCtx)
{
      
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file -
	// this is the number of cells on each side of
        //   a square (or cubic) domain.
	// AMReX only allows the logical domain to have a square or cubic shape
	// 
        pp.get("n_cell", SolverCtx->n_cell);

        pp.get("IterNum", SolverCtx->IterNum);

        // // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size", SolverCtx->max_grid_size);

        // // Default plot_int to -1, allow us to set it to something else in the inputs file
        // //  If plot_int < 0 then no plot files will be written
        SolverCtx->plot_int = -1;
        pp.query("plot_int", SolverCtx->plot_int);

        // // Default nsteps to 10, allow us to set it to something else in the inputs file
        SolverCtx->nsteps = 10;
        pp.query("nsteps", SolverCtx->nsteps);

        SolverCtx->cfl = 0.9;
        pp.query("cfl", SolverCtx->cfl);

        // // Parsing the Reynolds number and viscosity from input file also
        pp.get("ren", SolverCtx->ren);
        pp.get("vis", SolverCtx->vis);


	// Physical boundary condition mapping
	/* These are the types of boundary conditions 
	 * supported by the codes
	 */
	// 0 is periodic
	// -1 is non-slip
	// 1 is slip

	//Vector<int> phy_bc_lo(AMREX_SPACEDIM, 0);
	//Vector<int> phy_bc_hi(AMREX_SPACEDIM, 0);

	// Declaring params for boundary conditon type
	Vector<int> bc_lo(AMREX_SPACEDIM, 0);
	Vector<int> bc_hi(AMREX_SPACEDIM, 0);


        // // Parsing boundary condition from input file
        //pp.queryarr("phy_bc_lo", phy_bc_lo);
        //pp.queryarr("phy_bc_hi", phy_bc_hi);

        pp.queryarr("bc_lo", bc_lo);
        pp.queryarr("bc_hi", bc_hi);

	// Loop all around the dimensions and assign the values of the BCs
       for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
	 {
	   //SolverCtx->phy_bc_lo[dir] = phy_bc_lo[dir];
	   //SolverCtx->phy_bc_hi[dir] = phy_bc_hi[dir];
	  SolverCtx->bc_lo[dir] = bc_lo[dir];
	  SolverCtx->bc_hi[dir] = bc_hi[dir];
       }// End of loop around all dimensions

       // Testing periodicity in all directions
    for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
      {
        if (SolverCtx->bc_lo[dir] == 0 && SolverCtx->bc_hi[dir] == 0)
	  {
            SolverCtx->is_periodic[dir] = 1;
	  }
      }// End of all directions

}// End of function
/*
 * Define the geometrical characteristics of the domain
 * Define the MPI processes among all compute nodes 
*/
void Define_Domain(amress_solver *SolverCtx)
{
    BoxArray ba;
    Geometry geom;
    
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(SolverCtx->n_cell-1, SolverCtx->n_cell-1, SolverCtx->n_cell-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(SolverCtx->max_grid_size);

        // Here, the real domain is a rectangular box defined by (0,0); (0,1); (1,0); (1,1)
        // This defines the physical box, [0,1] in each direction.
        RealBox real_box({AMREX_D_DECL( Real(0.0), Real(0.0), Real(0.0))},
                         {AMREX_D_DECL( Real(1.0), Real(1.0), Real(1.0))});

        // This defines a Geometry object
        //   NOTE: the coordinate system is Cartesian
	std::vector<int> is_periodic(std::begin(SolverCtx->is_periodic), std::end(SolverCtx->is_periodic));

        geom.define(domain, &real_box, CoordSys::cartesian, is_periodic.data());
    

	// How Boxes are distrubuted among MPI processes
	// Distribution mapping between the processors
	// This is for volume-center variables
	DistributionMapping dm(ba);

    SolverCtx->ba = ba;
    SolverCtx->geom = geom;
    SolverCtx->domain = domain;
    SolverCtx->dm     = dm;

    
    // User Contex MultiFab contains 2 components, pressure and Phi, at the cell center
    SolverCtx->userCtx.define(ba, dm, SolverCtx->Ncomp, SolverCtx->Nghost);
    SolverCtx->userCtxPrev.define(ba, dm, SolverCtx->Ncomp, SolverCtx->Nghost);

    
}
//-------------------------------------------------
//+++++++++++++++++++++++++++++++++++++++++++++++++
//+++  EXPORT VELOCITY FIELS ++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++
void Export_Flow_Field (
			       amrex::BoxArray& ba,
			       amrex::DistributionMapping& dm,
			       amrex::Geometry& geom,
			       amrex::MultiFab& userCtx,
                               amrex::MultiFab& velCart,
                               int const& timestep,
			       amrex::Real const& time)
{

  // Depending on the dimensions the MultiFab needs to store enough
  // components 4 : (u,v,w, p) for flow fields in 3D
  // components = 3 (u,v,p) for flow fields in 2D
#if (AMREX_SPACEDIM > 2)
        MultiFab plt(ba, dm, 4, 0);
#else
        MultiFab plt(ba, dm, 3, 0);
#endif

        // Copy the pressure and velocity fields to the 'plt' Multifab
	// Note the component sequence
	// userCtx [0] --> pressure
	// velCart [1] --> u
	// velCart [2] --> v
	// velCart [3] --> w
        MultiFab::Copy(plt, userCtx, 0, 0, 1, 0);
        MultiFab::Copy(plt, velCart, 0, 1, 1, 0);
        MultiFab::Copy(plt, velCart, 1, 2, 1, 0);
#if (AMREX_SPACEDIM > 2)
        MultiFab::Copy(plt, velCart, 2, 3, 1, 0);
#endif

        
        const std::string& pltfile = amrex::Concatenate("Fields", timestep, 5); //5 spaces
#if (AMREX_SPACEDIM > 2)
        WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V", "W"}, geom, timestep, 0);
#else
        WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V"}, geom, timestep, 0);
#endif

  
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//------- EXPORT FLUXES -------------------------------------------
//-----------------------------------------------------------------
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Export_Fluxes(
		   amrex::MultiFab &fluxConvect,
		   amrex::MultiFab &fluxViscous,
		   amrex::MultiFab &fluxPrsGrad,
		   amrex::BoxArray &ba,
		   amrex::DistributionMapping &dm,
		   amrex::Geometry &geom,
		   int const       &timestep,
		   amrex::Real const &time)
{

            MultiFab plt(ba, dm, 3*AMREX_SPACEDIM, 0);

            MultiFab::Copy(plt, fluxConvect, 0, 0, 1, 0);
            MultiFab::Copy(plt, fluxConvect, 1, 1, 1, 0);
            MultiFab::Copy(plt, fluxViscous, 0, 2, 1, 0);
            MultiFab::Copy(plt, fluxViscous, 1, 3, 1, 0);
            MultiFab::Copy(plt, fluxPrsGrad, 0, 4, 1, 0);
            MultiFab::Copy(plt, fluxPrsGrad, 1, 5, 1, 0);

            const std::string& plt_flux = amrex::Concatenate("pltFlux", timestep, 5);
            WriteSingleLevelPlotfile(plt_flux, plt, {"conv_fluxx", "conv_fluxy", "visc_fluxx", "visc_fluxy", "press_gradx", "press_grady"}, geom, time, timestep);

  
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++ Error Norm computation ++++++++++++++++++++++++++++++
//-----------------------------------------------------------------
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
amrex::Real Error_Computation( amrex::Array<MultiFab, AMREX_SPACEDIM>  &velImRK,
			       amrex::Array<MultiFab, AMREX_SPACEDIM>  &velImPrev,
			       amrex::Array<MultiFab, AMREX_SPACEDIM>  &velImDiff,
			       amrex::Geometry const &geom)
{
  amrex::Real normError;
  
 	
           //  // MOMENTUM |3| UPDATE ERROR
            for ( MFIter mfi(velImRK[0]); mfi.isValid(); ++mfi )
              {
		
                  const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
                  const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));

  #if (AMREX_SPACEDIM > 2)
                  const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
  #endif
                  auto const& ximrk = velImRK[0].array(mfi);
                  auto const& yimrk = velImRK[1].array(mfi);

  #if (AMREX_SPACEDIM > 2)
                  auto const& zimrk = velImRK[2].array(mfi);
  #endif
                  auto const& xprev = velImPrev[0].array(mfi);
                  auto const& yprev = velImPrev[1].array(mfi);
  #if (AMREX_SPACEDIM > 2)
                  auto const& zprev = velImPrev[2].array(mfi);
  #endif
                  auto const& xdiff = velImDiff[0].array(mfi);
                  auto const& ydiff = velImDiff[1].array(mfi);

  #if (AMREX_SPACEDIM > 2)
                  auto const& zdiff = velImDiff[2].array(mfi);
  #endif
                  amrex::ParallelFor(xbx,
                  [=] AMREX_GPU_DEVICE (int i, int j, int k){
                      xdiff(i, j, k) = xprev(i, j, k) - ximrk(i, j, k);
                  });

                  amrex::ParallelFor(ybx,
                  [=] AMREX_GPU_DEVICE(int i, int j, int k){
                      ydiff(i, j, k) = yprev(i, j, k) - yimrk(i, j, k);
                  });

  #if (AMREX_SPACEDIM > 2)
                  amrex::ParallelFor(zbx,
                  [=] AMREX_GPU_DEVICE(int i, int j, int k){
                      zdiff(i, j, k) = zprev(i, j, k) - zimrk(i, j, k);
                  });
  #endif
	      }// End of all loops for Multi-Fabs
	   
            Real xerror = velImDiff[0].norm2(0, geom.periodicity());
            Real yerror = velImDiff[1].norm2(0, geom.periodicity());
            // Real xerror = velImDiff[0].norminf(0, 0);
            // Real yerror = velImDiff[1].norminf(0, 0);

            normError = std::max(xerror, yerror);
 #if (AMREX_SPACEDIM > 2)
             Real zerror = velContDiff[2].norminf(0, geom.periodicity());
             // Real zerror = velImDiff[2].norminf(0, 0);
             normError = std::max(normError, zerror);
 #endif

   
	    return normError;
}


//-----------------------------------------------------------------
// ============================== SOLVER SECTION ==================
//-----------------------------------------------------------------
void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions
    // These are stock params for AMReX
    int n_cell, max_grid_size, nsteps, plot_int;
    int IterNum;

    // Porting extra params from Julian code
    Real ren, vis, cfl;
    
    // Declaring params for boundary conditon type
    Vector<int> bc_lo(AMREX_SPACEDIM, 0);
    Vector<int> bc_hi(AMREX_SPACEDIM, 0);

    //-------------------------------------------------------------
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Parsing Inputs =-=-=-=-=-=-=-=
    //-------------------------------------------------------------
    amress_solver SolverCtx;
    Input_Parameters(&SolverCtx);
	
    // Temporary here!
       	n_cell = SolverCtx.n_cell;
	IterNum = SolverCtx.IterNum;
       	max_grid_size  = SolverCtx.max_grid_size;
	plot_int = SolverCtx.plot_int;
        nsteps = SolverCtx.nsteps;
        cfl = SolverCtx.cfl;
	ren = SolverCtx.ren;
	vis = SolverCtx.vis;

        // Parsing boundary condition from the Context
	for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
	  {

	    bc_lo[dir] == SolverCtx.bc_lo[dir]; 
	    bc_hi[dir] == SolverCtx.bc_hi[dir]; 

	  }

	    
        amrex::Print() << "INFO| number of cells in each side of the domain: " << n_cell << "\n";
	

    for (int dir=0; dir < AMREX_SPACEDIM; ++dir)        
	amrex::Print() << "INFO| periodicity in " << dir << "th dim " << SolverCtx.is_periodic[dir] << "\n";

    //------------------------------------------------------------------
    //------------------------------------------------------------------
    // ==-=-=-=-=-=-=-=-=-=-=-= Defining System's Variables =-=-=-==-=-=
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    Define_Domain(&SolverCtx);


    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    Box domain;
    
    ba = SolverCtx.ba;
    geom = SolverCtx.geom;
    domain = SolverCtx.domain;
    
      // Nghost = number of ghost cells for each array
    int Nghost = SolverCtx.Nghost; // 2nd order accuracy scheme is used for convective terms


    // Ncomp = number of components for each array
    // The userCtx has 2 components: phi and pressure
    int Ncomp  = SolverCtx.Ncomp;

    // How Boxes are distrubuted among MPI processes
    // Distribution mapping between the processors
    DistributionMapping dm = SolverCtx.dm;

    /*
     * ----------------------- 
     *   Volume center 
     *  ----------------------
     *  |                    |
     *  |                    |
     *  |          0         |
     *  |                    |
     *  |                    |
     *  ---------------------- 
     */

    // User Contex MultiFab contains 2 components, pressure and Phi, at the cell center
    MultiFab userCtx(ba, dm, Ncomp, Nghost);
    MultiFab userCtxPrev(ba, dm, Ncomp, Nghost);

    
    // Define the RHS for the Poisson equation - Just one component
    MultiFab Poisson_RHS_Vector(ba, dm, 1, Nghost);
    MultiFab phi_solution(ba, dm, 1, Nghost);
   

    // Cartesian velocities have SPACEDIM as number of components, live in the cell center
    MultiFab velCart(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab velCartDiff(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab velCartPrev(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab velCartPrevPrev(ba, dm, AMREX_SPACEDIM, Nghost);

    // Three type of fluxes contributing the the total flux live in the cell center
    MultiFab fluxConvect(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab fluxViscous(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab fluxPrsGrad(ba, dm, AMREX_SPACEDIM, Nghost);

    MultiFab fluxTotal(ba, dm, AMREX_SPACEDIM, Nghost);

    /* --------------------------------------
     * Face center variables - FLUXES -------
     * and Variables ------------------------
     *---------------------------------------
     *              _____________
     *             |             |
     *             |             |---> velCont
     *             |    0 velCart|
     *             |             |
     *              _____________      
     *
     */            

    //------------------------------------------------------
    // 1 of 4 options: 'pressure', 'velocity', 'flux', 'velocity'
    //------------------------------------------------------
    const std::string& type1 = "pressure";
    const std::string& type2 = "velocity";
    const std::string& type3 = "flux";
    const std::string& type4 = "velocity";

    // Contravariant velocities live in the face center
    Array<MultiFab, AMREX_SPACEDIM> velCont;
    Array<MultiFab, AMREX_SPACEDIM> velContDiff;
    // Right-Hand-Side terms of the Momentum equation have SPACEDIM as number of components, live in the face center
    Array<MultiFab, AMREX_SPACEDIM> rhs;
    // Half-node fluxes contribute to implementation of QUICK scheme in calculating the convective flux
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN1;
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN2;
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN3;

    // Intermediate velocity fields
    Array<MultiFab, AMREX_SPACEDIM> velImRK;
    Array<MultiFab, AMREX_SPACEDIM> velImPrev;
    Array<MultiFab, AMREX_SPACEDIM> velImDiff;
    // gradient of phi
    Array<MultiFab, AMREX_SPACEDIM> grad_phi;
	
    // Due to the mismatch between the volume-center and face-center variables
    // The physical quantities living at the face center need to be blowed out one once in the respective direction
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        BoxArray edge_ba = ba;
        edge_ba.surroundingNodes(dir);

        velCont[dir].define(edge_ba, dm, 1, 0);
        velContDiff[dir].define(edge_ba, dm, 1, 0);
        rhs[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN1[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN2[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN3[dir].define(edge_ba, dm, 1, 0);

        velImRK[dir].define(edge_ba, dm, 1, 0);
        velImPrev[dir].define(edge_ba, dm, 1, 0);
        velImDiff[dir].define(edge_ba, dm, 1, 0);

	grad_phi[dir].define(edge_ba, dm, 1, 0);
    }

    //---------------------------------------------------------------
    // Defining the boundary conditions for each face of the domain
    // --------------------------------------------------------------
    Vector<BCRec> bc(userCtxPrev.nComp());
    for (int n = 0; n < userCtxPrev.nComp(); ++n)
    {
        for(int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            //Internal Dirichlet Periodic Boundary conditions, or bc_lo = bc_hi = 0
            if (bc_lo[idim] == BCType::int_dir) {
                bc[n].setLo(idim, BCType::int_dir);
            }
            //First Order Extrapolation for Neumann boundary conditions or bc_lo, bc_hi = 2
            else if (bc_lo[idim] == BCType::foextrap) {
                bc[n].setLo(idim, BCType::foextrap);
            }
            //External Dirichlet Boundary Condition, or bc_lo, bc_hi = 3
            else if(bc_lo[idim] == BCType::ext_dir) {
                bc[n].setLo(idim, BCType::ext_dir);
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            //Internal Dirichlet Periodic Boundary conditions, or bc_lo = bc_hi = 0
            if (bc_hi[idim] == BCType::int_dir) {
                bc[n].setHi(idim, BCType::int_dir);
            }
            //First Order Extrapolation for Neumann boundary conditions or bc_lo, bc_hi = 2
            else if (bc_hi[idim] == BCType::foextrap) {
                bc[n].setHi(idim, BCType::foextrap);
            }
            //External Dirichlet Boundary Condition, or bc_lo, bc_hi = 3
            else if(bc_hi[idim] == BCType::ext_dir) {
                bc[n].setHi(idim, BCType::ext_dir);
            }
            else {
                amrex::Abort("Invalid bc_hi");
            }
        }
    }




    
    // Print desired variables for debugging
    amrex::Print() << "INFO| number of dimensions: " << AMREX_SPACEDIM << "\n";
    amrex::Print() << "INFO| geometry: " << geom << "\n";
    amrex::Print() << "PARAMS| number of ghost cells for each array: " << Nghost << "\n";
    amrex::Print() << "PARAMS| number of components for each array: " << Ncomp << "\n";

   //--------------------------------------------------------------------------------------// 
   // =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Initialization =-=-=-=-=-=-=-=-=-=-=-=-=-=-------------//
   //--------------------------------------------------------------------------------------//
    amrex::Print() << "==== INITIALIZATION STEP =========== \n";
    // Current: Taylor-Green Vortex initial conditions
    // How partial periodic boundary conditions can be deployed?
    init(userCtx, velCart, velCartDiff, velContDiff, geom);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    Real coeff = AMREX_D_TERM(   1./(dx[0]*dx[0]),
                               + 1./(dx[1]*dx[1]),
                               + 1./(dx[2]*dx[2]) );
    Real dt = cfl/(2.0*coeff);
    
    // time = starting time in the simulation
    Real time = 0.0;

    amrex::Print() << "PARAMS| cfl value: " << cfl << "\n";
    amrex::Print() << "PARAMS| dt value from above cfl: " << dt << "\n";

    //------------------------------------------------------------------//    
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Plotting =-=-=-=-=-=-=-=-=-=-=-=-=-//
    // -----------------------    Initial state ----------------------- //
    //------------------------------------------------------------------//

    // Write a plotfile of the initial data if plot_int > 0
    // (plot_int was defined in the inputs file)
    if (plot_int > 0)      
      Export_Flow_Field( ba, dm, geom, userCtx, velCart, 0, time); // Export the initial flow field
      
    //====== MODULE | ADVANCE =====================================
    // ++++++++++ KIM AND MOINE'S RUNGE-KUTTA +++++++++++++++++++++
    amrex::Print() << "======= ADVANCING STEP  ==================== \n";
    // Setup stopping criteria
    Real Tol = 1.0e-8;
    // int IterNum = 10;

    // Setup Runge-Kutta scheme coefficients
    int RungeKuttaOrder = 4;
    GpuArray<Real,MAX_RK_ORDER> rk;
    {
        rk[0] = Real(0.25);
        rk[1] = Real(1.0)/Real(3.0);
        rk[2] = Real(0.5);
        rk[3] = Real(1.0);
    }

    //+++++++++++++++++++++++++++++++++++++++++
    //+++++   Begin time loop +++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++++++
    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(userCtxPrev, userCtx, 0, 0, Ncomp, Nghost);
        MultiFab::Copy(velCartPrev, velCart, 0, 0, AMREX_SPACEDIM, Nghost);
        // Forming boundary conditions
        userCtx.FillBoundary(geom.periodicity());

	// Enforce the physical boundary conditions
        enforce_boundary_conditions(userCtx, type1, Nghost, bc_lo, bc_hi, n_cell);

	// Doing the HALO exchange
	// This is important
	// If the physical boundary are not periodic
	// Then the update will not touch those grid points
        velCart.FillBoundary(geom.periodicity());
        
	// Enforce the boundary conditions again
        enforce_boundary_conditions(velCart, type2, Nghost, bc_lo, bc_hi, n_cell);

        // Convert cartesian velocity to contravariant velocity
	// after boundary conditions are enfoced
        // velCont is updated first via Momentum solver
        cart2cont(velCart, velCont);

	// Copy the intermediate values to the next sub-iteration
        for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
        {
            MultiFab::Copy(  velImRK[comp],     velCont[comp], 0, 0, 1, 0);
            MultiFab::Copy(velImDiff[comp], velContDiff[comp], 0, 0, 1, 0);
        }

        // Momentum solver
        // After debugging, all the code below will be modulized to the MOMENTUM module i.e.,:
        // momentum_km_runge_kutta();

        // MOMENTUM |1| Setup counter
        int countIter = 0;
        Real normError = 1.0e6;

	//-----------------------------------------------
        // This is the sub-iteration of the implicit RK4
	//-----------------------------------------------
        while ( countIter < IterNum && normError > Tol )
        {
            countIter++;
            amrex::Print() << "SOLVING| Momentum | performing Runge-Kutta at iteration: " << countIter << "\n";

	    // Assign the initial guess as the previous flow field
            for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
            {
                MultiFab::Copy(velImPrev[comp], velImRK[comp], 0, 0, 1, 0);
            }

	    // 4 sub-iterations of one RK4 iteration
            for (int sub = 0; sub < RungeKuttaOrder; ++sub )
            { 
                // RUNGE-KUTTA | Calculate Cell-centered Convective terms
                convective_flux_calc(fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCart, velImRK, bc_lo, bc_hi, geom, n_cell);

                // RUNGE-KUTTA | Calculate Cell-centered Viscous terms
                viscous_flux_calc(fluxViscous, velCart, geom, ren);

                // RUNGE-KUTTA | Calculate Cell-centered Pressure Gradient terms
                pressure_gradient_calc(fluxPrsGrad, userCtx, geom);

                // RUNGE-KUTTA | Calculate Cell-centered Total Flux = -fluxConvect + fluxViscous - fluxPrsGrad
                total_flux_calc(fluxTotal, fluxConvect, fluxViscous, fluxPrsGrad);

                fluxTotal.FillBoundary(geom.periodicity());
                enforce_boundary_conditions(fluxTotal, type3, Nghost, bc_lo, bc_hi, n_cell);

                // RUNGE-KUTTA | Calculate the Face-centered Right-Hand-Side terms by averaging the Cell-centered fluxes
                righthand_side_calc(rhs, fluxTotal);
		
                // RUNGE-KUTTA | Advance; increment rhs and use it to update velImRK
                km_runge_kutta_advance(rk, sub, rhs, velImRK, velCont, velContDiff, dt, bc_lo, bc_hi, n_cell);
                // After advance through 4 sub-step we obtain guessed velCont at next time step

                // RUNGE-KUTTA | Update velCart from velImRK
                cont2cart(velCart, velImRK, geom);
                // This updated velCart will be used again next sub-iteration
                // So, we need to re-enforce the boundary conditions
                velCart.FillBoundary(geom.periodicity());
                enforce_boundary_conditions(velCart, type4, Nghost, bc_lo, bc_hi, n_cell);

            } // RUNGE-KUTTA | END

	    normError = Error_Computation(velImRK, velImPrev, velImDiff, geom);
	    amrex::Print() << "error norm2 = " << normError << "\n";
 
        }// End of the RK-4 LOOP Iteration!
        amrex::Print() << "SOLVING| Momentum | ending Runge-Kutta after " << countIter << " iteration(s) with convergence: " << normError << "\n";

	//---------------------------------------
        // MOMENTUM |4| PLOTTING
	// This is just for debugging only !
	//---------------------------------------
        if (plot_int > 0 )	  
	    Export_Fluxes( fluxConvect, fluxViscous, fluxPrsGrad, ba, dm, geom, n, time);

        amrex::Print() << "SOLVING| Momentum | finished time step: " << n << "\n";

	// Setup the RHS
	Poisson_RHS(geom, velImRK, Poisson_RHS_Vector, dt);
	Set_Phi_To_Zero(phi_solution);
	
        //POISSON SOLVER
	Poisson_Solver (phi_solution, Poisson_RHS_Vector, geom, ba, dm, bc);

	// Update the solution
	// U^{n+1} = v* + grad (\phi)
	// p^{n+1} = p  + \phi
        Poisson_Update_Solution (phi_solution, grad_phi, userCtx, velCont, velImRK, geom, ba, dm, bc, dt);

	// Update velCart from the velCont solutions
        // SHOULD THIS BE velCont instead of velImRK?
        cont2cart(velCart, velImRK, geom);

	// This updated velCart will be used again next sub-iteration
	// So, we need to re-enforce the boundary conditions
	// Update the halo exchange points!
	velCart.FillBoundary(geom.periodicity());
	enforce_boundary_conditions(velCart, type4, Nghost, bc_lo, bc_hi, n_cell);

	// advance will do all above steps
        time = time + dt;

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
	  Export_Flow_Field(ba, dm, geom, userCtx, velCart, n, time);
	
    }//end of time loop - this is the (n) loop!

    // Call the timer again and compute the maximum difference
    // between the start time and stop time
    // over all processors
    auto stop_time = ParallelDescriptor::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}
