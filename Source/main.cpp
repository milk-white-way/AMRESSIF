/** 
 * Incompressible Flow Solver
 * Testing .... 

 * 
 * Trung Edited
 * Jul/31/2023 - Trung edited
 */

// ============================== LISTING KERNEL HEADERS ==============================
#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>

#include "main_main.H"
#include "main.H"
// #include "./meTh_Fractional_Time/mdl_initialization/fn_init.H"
// #include "./meTh_Fractional_Time/mdl_momentum/fn_momentum.H"
// #include "./meTh_Fractional_Time/mdl_poisson/fn_poisson.H"
// #include "./meTh_Fractional_Time/mdl_advance/fn_advance.H"

// #include "./meTh_Fractional_Time/utl_boundary_conditions/fn_fill_ghostcells.H"
// #include "./meTh_Fractional_Time/utl_conversion/fn_cart2cont.H"
// #include "./meTh_Fractional_Time/utl_conversion/fn_cont2cart.H"

// Modulization library
#include "fn_cart2cont.H"
#include "fn_cont2cart.H"
#include "fn_init.H"
#include "fn_enforce_wall_bcs.H"
#include "fn_flux_calc.H"
#include "fn_rhs.H"

// Default library
#include "myfunc.H"
//#include "momentum.H"

    class amress_solver
    {
    public:
      //  amress_solver ();
      //~amress_solver ();
      //  amress_solver (amress_solver const&) = delete;
      //  amress_solver (amress_solver &&) = delete;
      //  amress_solver& operator= (amress_solver const&) = delete;
      //  amress_solver& operator= (amress_solver &&) = delete;
      int test_value;

        
    
    };


using namespace amrex;

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

void Input_Parameters(amress_solver *UserCtx)
{
  UserCtx->test_value = 1;

}

// ============================== SOLVER SECTION ==============================
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

    // Physical boundary condition mapping
    /* These are the types of boundary conditions 
     * supported by the codes
     */
    // 0 is periodic
    // -1 is non-slip
    // 1 is slip
    Vector<int> phy_bc_lo(AMREX_SPACEDIM, 0);
    Vector<int> phy_bc_hi(AMREX_SPACEDIM, 0);

    // Declaring params for boundary conditon type
    Vector<int> bc_lo(AMREX_SPACEDIM, 0);
    Vector<int> bc_hi(AMREX_SPACEDIM, 0);
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Parsing Inputs =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
	// AMReX only allows the logical domain to have a square or cubic shape
	// 
        pp.get("n_cell", n_cell);

        amrex::Print() << "INFO| number of cells in each side of the domain: " << n_cell << "\n";

        pp.get("IterNum", IterNum);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size", max_grid_size);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int", plot_int);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps", nsteps);

        cfl = 0.9;
        pp.query("cfl", cfl);

        // Parsing the Reynolds number and viscosity from input file also
        pp.get("ren", ren);
        pp.get("vis", vis);

        // Parsing boundary condition from input file
        pp.queryarr("phy_bc_lo", phy_bc_lo);
        pp.queryarr("phy_bc_hi", phy_bc_hi);

        pp.queryarr("bc_lo", bc_lo);
        pp.queryarr("bc_hi", bc_hi);

	
	amress_solver test_solver;
	test_solver.test_value = 1;
	
	amrex::Print() << "test_value: TRUNG"  << test_solver.test_value << "\n";


    }

    
    // Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    Vector<int> is_periodic(AMREX_SPACEDIM, 0);
    // BCType::int_dir = 0
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        if (phy_bc_lo[idim] == 0 && phy_bc_hi[idim] == 0) {
            is_periodic[idim] = 1;
        }

        amrex::Print() << "INFO| periodicity in " << idim << "th dimension: " << is_periodic[idim] << "\n";

    }

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Defining System's Variables =-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

        // Here, the real domain is a rectangular box defined by (0,0); (0,1); (1,0); (1,1)
        // This defines the physical box, [0,1] in each direction.
        RealBox real_box({AMREX_D_DECL( Real(0.0), Real(0.0), Real(0.0))},
                         {AMREX_D_DECL( Real(1.0), Real(1.0), Real(1.0))});

        // This defines a Geometry object
        //   NOTE: the coordinate system is Cartesian
        geom.define(domain, &real_box, CoordSys::cartesian, is_periodic.data());
    }

    // Nghost = number of ghost cells for each array
    int Nghost = 2; // 2nd order accuracy scheme is used for convective terms


    // Ncomp = number of components for each array
    // The userCtx has 2 components: phi and pressure
    int Ncomp  = 2;

    // How Boxes are distrubuted among MPI processes
    // Distribution mapping between the processors
    DistributionMapping dm(ba);

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

    // Contravariant velocities live in the face center
    Array<MultiFab, AMREX_SPACEDIM> velCont;
    Array<MultiFab, AMREX_SPACEDIM> velContDiff;
    // Right-Hand-Side terms of the Momentum equation have SPACEDIM as number of components, live in the face center
    Array<MultiFab, AMREX_SPACEDIM> rhs;
    // Half-node fluxes contribute to implementation of QUICK scheme in calculating the convective flux
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN1;
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN2;
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN3;

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

    amrex::Print() << "INFO| box array: " << ba << "\n";
    amrex::Print() << "INFO| geometry: " << geom << "\n";

    amrex::Print() << "PARAMS| number of ghost cells for each array: " << Nghost << "\n";
    amrex::Print() << "PARAMS| number of components for each array: " << Ncomp << "\n";

   //--------------------------------------------------------------------------------------// 
   // =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Initialization =-=-=-=-=-=-=-=-=-=-=-=-=-=-------------//
   //--------------------------------------------------------------------------------------//
    amrex::Print() << "===================== INITIALIZATION STEP ===================== \n";
    // Current: Taylor-Green Vortex initial conditions
    // How partial periodic boundary conditions can be deployed?
    init(userCtx, velCart, velCartDiff, velContDiff, geom);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    Real coeff = AMREX_D_TERM(   1./(dx[0]*dx[0]),
                               + 1./(dx[1]*dx[1]),
                               + 1./(dx[2]*dx[2]) );
    Real dt = cfl/(2.0*coeff);
    // Real dt = 1.0e-4;

    // time = starting time in the simulation
    Real time = 0.0;

    amrex::Print() << "PARAMS| cfl value: " << cfl << "\n";
    amrex::Print() << "PARAMS| dt value from above cfl: " << dt << "\n";

    //------------------------------------------------------------------//    
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Plotting =-=-=-=-=-=-=-=-=-=-=-=-=-//
    // -----------------------    Initial state ----------------------- //
    //------------------------------------------------------------------//

    // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
#if (AMREX_SPACEDIM > 2)
        MultiFab plt(ba, dm, 4, 0);
#else
        MultiFab plt(ba, dm, 3, 0);
#endif

        // Copy the pressure and velocity fields to the 'plt' Multifab
        MultiFab::Copy(plt, userCtx, 0, 0, 1, 0);
        MultiFab::Copy(plt, velCart, 0, 1, 1, 0);
        MultiFab::Copy(plt, velCart, 1, 2, 1, 0);
#if (AMREX_SPACEDIM > 2)
        MultiFab::Copy(plt, velCart, 2, 3, 1, 0);
#endif

        int n = 0;
        const std::string& pltfile = amrex::Concatenate("pltInitialization", n, 5);
#if (AMREX_SPACEDIM > 2)
        WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V", "W"}, geom, time, 0);
#else
        WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V"}, geom, time, 0);
#endif
    }

// ==================================== MODULE | ADVANCE =====================================
    // ++++++++++ KIM AND MOINE'S RUNGE-KUTTA ++++++++++
    amrex::Print() << "======================= ADVANCING STEP  ======================= \n";
    // Setup stopping criteria
    Real Tol = 1.0e-8;
    // int IterNum = 10;

    // Setup Runge-Kutta scheme coefficients
int RungeKuttaOrder = 4;
Vector<Real> rk(RungeKuttaOrder, 0);
    {
        rk[0] = Real(0.25);
        rk[1] = Real(1.0)/Real(3.0);
        rk[2] = Real(0.5);
        rk[3] = Real(1.0);
    }

    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(userCtxPrev, userCtx, 0, 0, Ncomp, Nghost);
        MultiFab::Copy(velCartPrev, velCart, 0, 0, AMREX_SPACEDIM, Nghost);
        // Forming boundary conditions
        userCtx.FillBoundary(geom.periodicity());
        const std::string& type1 = "pressure"; // 1 of 4 options: 'pressure', 'phi', 'velocity', 'flux'
        enforce_boundary_conditions(velCart, type1, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

        velCart.FillBoundary(geom.periodicity());
        const std::string& type2 = "velocity";
        enforce_boundary_conditions(velCart, type2, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

        // Convert cartesian velocity to contravariant velocity after boundary conditions are enfoced
        // velCont is updated first via Momentum solver
        cart2cont(velCart, velCont);

        Array<MultiFab, AMREX_SPACEDIM> velImRK;
        Array<MultiFab, AMREX_SPACEDIM> velImPrev;
        Array<MultiFab, AMREX_SPACEDIM> velImDiff;
        for ( int dir=0; dir < AMREX_SPACEDIM; dir++ )
        {
            BoxArray edge_ba = ba;
            edge_ba.surroundingNodes(dir);

            velImRK[dir].define(edge_ba, dm, 1, 0);
            velImPrev[dir].define(edge_ba, dm, 1, 0);
            velImDiff[dir].define(edge_ba, dm, 1, 0);
        }
        for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
        {
            MultiFab::Copy(velImPrev[comp], velCont[comp], 0, 0, 1, 0);
            MultiFab::Copy(velImDiff[comp], velContDiff[comp], 0, 0, 1, 0);
        }

        // Momentum solver
        // After debugging, all the code below will be modulized to the MOMENTUM module i.e.,:
        // momentum_km_runge_kutta();

        // MOMENTUM |1| Setup counter
        int countIter = 0;
        Real normError = 1.0e1;

        // MOMENTUM |2| KIM AND MOINE'S FTS START
        while ( countIter < IterNum && normError > Tol )
        {
            countIter++;
            amrex::Print() << "SOLVING| Momentum | performing Runge-Kutta at iteration: " << countIter << "\n";
            /*
            if (countIter == 1) {
                amrex::Print() << "RUNGE-KUTTA |1| Create new face-centered variable to house the implicit velocity solutions... \n";
            } else {
                amrex::Print() << "RUNGE-KUTTA |1| Stopping conditions are not met; use the previous solution for iteration... \n";
            }
            */
            for ( int comp=0; comp < AMREX_SPACEDIM; ++comp )
            {
                MultiFab::Copy(velImRK[comp], velImPrev[comp], 0, 0, 1, 0);
            }

            // amrex::Print() << "RUNGE-KUTTA |2| Performing 4th-Order Runge-Kutta... \n";
            for (int sub = 0; sub < RungeKuttaOrder; ++sub )
            { // RUNGE-KUTTA | START
                // RUNGE-KUTTA | Calculate Cell-centered Convective terms
                convective_flux_calc(fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCart, velImRK, phy_bc_lo, phy_bc_hi, geom, n_cell);

                // RUNGE-KUTTA | Calculate Cell-centered Viscous terms
                viscous_flux_calc(fluxViscous, velCart, geom, ren);

                // RUNGE-KUTTA | Calculate Cell-centered Pressure Gradient terms
                pressure_gradient_calc(fluxPrsGrad, userCtx, geom);

                // RUNGE-KUTTA | Calculate Cell-centered Total Flux
                total_flux_calc(fluxTotal, fluxConvect, fluxViscous, fluxPrsGrad);

                fluxTotal.FillBoundary(geom.periodicity());
                const std::string& type3 = "flux";
                enforce_boundary_conditions(fluxTotal, type3, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

                // RUNGE-KUTTA | Calculate the Face-centered Righ-Hand-Side terms
                righthand_side_calc(rhs, fluxTotal);

                // RUNGE-KUTTA | Advance
                km_runge_kutta_advance(rk, sub, rhs, velImRK, velCont, velContDiff, dt, phy_bc_lo, phy_bc_hi, n_cell);
                // After advance through 4 sub-step we obtain guessed velCont at next time step

                // RUNGE-KUTTA | Update velCart from the velCont solutions
                cont2cart(velCart, velImRK, geom);
                // This updated velCart will be used again next sub-iteration
                // So, we need to re-enforce the boundary conditions
                velCart.FillBoundary(geom.periodicity());
                const std::string& type4 = "velocity";
                enforce_boundary_conditions(velCart, type4, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

            } // RUNGE-KUTTA | END
            // MOMENTUM |3| UPDATE ERROR
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
            }
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
            // amrex::Print() << "DEBUGGING| intermediate convergence: " << normError << "\n";
            // Update contravariant velocity
            for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
            {
                MultiFab::Copy(velImPrev[comp], velImRK[comp], 0, 0, 1, 0);
            }
        }
        amrex::Print() << "SOLVING| Momentum | ending Runge-Kutta after " << countIter << " iteration(s) with convergence: " << normError << "\n";

        // MOMENTUM |4| PLOTTING
        if (plot_int > 0 ) {
            MultiFab plt(ba, dm, 3*AMREX_SPACEDIM, 0);

            MultiFab::Copy(plt, fluxConvect, 0, 0, 1, 0);
            MultiFab::Copy(plt, fluxConvect, 1, 1, 1, 0);
            MultiFab::Copy(plt, fluxViscous, 0, 2, 1, 0);
            MultiFab::Copy(plt, fluxViscous, 1, 3, 1, 0);
            MultiFab::Copy(plt, fluxPrsGrad, 0, 4, 1, 0);
            MultiFab::Copy(plt, fluxPrsGrad, 1, 5, 1, 0);

            const std::string& plt_flux = amrex::Concatenate("pltFlux", n, 5);
            WriteSingleLevelPlotfile(plt_flux, plt, {"conv_fluxx", "conv_fluxy", "visc_fluxx", "visc_fluxy", "press_gradx", "press_grady"}, geom, time, n);
        }

        amrex::Print() << "SOLVING| Momentum | finished time step: " << n << "\n";
        // MOMENTUM |5| KIM AND MOINE'S FTS END

        // PSEUDO-CODE: POISSON SOLVER HERE

        // advance will do all above steps
        //advance(userCtxOld, userCtx, flux, dt, geom);
        time = time + dt;

        // Tell the I/O Processor to write out which step we're doing
        // amrex::Print() << "INFO| End Advanced Step " << n << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            const std::string& plt_pressure_file = amrex::Concatenate("pltPressue", n, 5);
            const std::string& plt_velfield_file = amrex::Concatenate("pltVelocity", n, 5);
            WriteSingleLevelPlotfile(plt_pressure_file, userCtx, {"pressure", "phi"}, geom, time, n);
            WriteSingleLevelPlotfile(plt_velfield_file, velCart, {"U", "V"}, geom, time, n);
	}//End of plotting function
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    auto stop_time = ParallelDescriptor::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}
