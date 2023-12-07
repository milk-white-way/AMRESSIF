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
#include "fn_init.H"
#include "fn_enforce_wall_bcs.H"
#include "fn_flux_calc.H"
#include "fn_rhs.H"
#include "momentum.H"
#include "poisson.H"
#include "utilities.H"

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
    amrex::Initialize(argc,argv);
    main_main();
    amrex::Finalize();
    return 0;
}

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

    // Physical boundary condition mapping
    // 0 is periodic
    // -1 is non-slip
    // 1 is slip
    Vector<int> phy_bc_lo(AMREX_SPACEDIM, 0);
    Vector<int> phy_bc_hi(AMREX_SPACEDIM, 0);

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Parsing Inputs =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
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
        nsteps = 1;
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
    }

    Vector<int> is_periodic(AMREX_SPACEDIM, 0);
    // BCType::int_dir = 0
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        if (phy_bc_lo[idim] == 0 && phy_bc_hi[idim] == 0) {
            is_periodic[idim] = 1;
        }
        amrex::Print() << "INFO| periodicity in " << idim+1 << "th dimension: " << is_periodic[idim] << "\n";
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

    // Ncomp = number of components for userCtx
    // The userCtx has 03 components:
    // userCtx(0) = Pressure
    // userCtx(1) = Phi
    int Ncomp = 2;

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

    // Cartesian velocities have SPACEDIM as number of components, live in the cell center
    MultiFab velCart(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab velCartDiff(ba, dm, AMREX_SPACEDIM, Nghost);

    // Three type of fluxes contributing the the total flux live in the cell center
    MultiFab fluxConvect(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab fluxViscous(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab fluxPrsGrad(ba, dm, AMREX_SPACEDIM, Nghost);

    MultiFab fluxTotal(ba, dm, AMREX_SPACEDIM, Nghost);

    MultiFab poisson_rhs(ba, dm, 1, 1);
    MultiFab poisson_sol(ba, dm, 1, 1);

    MultiFab analyticSol(ba, dm, 3, 1);

    MultiFab l2norm(ba, dm, 3, 0);
    // Comp 0 is velocity field along x-axis
    // Comp 1 is velocity field along y-axis
    // Comp 2 is pressure field

    //---------------------------------------------------------------
    // Defining the boundary conditions for each face of the domain
    // --------------------------------------------------------------
    Vector<BCRec> bc(poisson_sol.nComp());
    for (int n = 0; n < poisson_sol.nComp(); ++n)
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
    // 1 of 4 options: 'pressure', 'phi', 'flux', 'velocity'
    //------------------------------------------------------
    const std::string& type1 = "pressure";
    const std::string& type2 = "phi";
    const std::string& type3 = "flux";
    const std::string& type4 = "velocity";

    // Contravariant velocities live in the face center
    Array<MultiFab, AMREX_SPACEDIM> velCont;
    Array<MultiFab, AMREX_SPACEDIM> velContPrev;
    Array<MultiFab, AMREX_SPACEDIM> velContDiff;

    // Right-Hand-Side terms of the Momentum equation have SPACEDIM as number of components, live in the face center
    Array<MultiFab, AMREX_SPACEDIM> momentum_rhs;

    // Half-node fluxes contribute to implementation of QUICK scheme in calculating the convective flux
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN1;
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN2;
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN3;

    // Extra velocity components for Fractional-Step Method
    Array<MultiFab, AMREX_SPACEDIM> velHat;
    Array<MultiFab, AMREX_SPACEDIM> velHatDiff;
    Array<MultiFab, AMREX_SPACEDIM> velStar;
    Array<MultiFab, AMREX_SPACEDIM> velStarDiff;
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
        velContPrev[dir].define(edge_ba, dm, 1, 0);
        velContDiff[dir].define(edge_ba, dm, 1, 0);

        momentum_rhs[dir].define(edge_ba, dm, 1, 0);

        fluxHalfN1[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN2[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN3[dir].define(edge_ba, dm, 1, 0);

        velHat[dir].define(edge_ba, dm, 1, 0);
        velHatDiff[dir].define(edge_ba, dm, 1, 0);
        velStar[dir].define(edge_ba, dm, 1, 0);
        velStarDiff[dir].define(edge_ba, dm, 1, 0);

        grad_phi[dir].define(edge_ba, dm, 1, 0);
    }

    // Print desired variables for debugging
    amrex::Print() << "INFO| number of dimensions: " << AMREX_SPACEDIM << "\n";
    amrex::Print() << "INFO| geometry: " << geom << "\n";
    amrex::Print() << "PARAMS| number of ghost cells for each array: " << Nghost << "\n";
    amrex::Print() << "PARAMS| number of components for each array: " << Ncomp << "\n";

    //--------------------------------------------------------------------------------------//
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Initialization =-=-=-=-=-=-=-=-=-=-=-=-=-=-------------//
    //--------------------------------------------------------------------------------------//

    amrex::Print() << "========================= INITIALIZATION STEP ========================= \n";
    // Current: Taylor-Green Vortex initial conditions
    // How partial periodic boundary conditions can be deployed?
    init(userCtx, velCart, velCartDiff, velContDiff, geom);
    MultiFab::Copy(poisson_sol, userCtx, 1, 0, 1, 1);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    Real coeff = AMREX_D_TERM(   1./(dx[0]*dx[0]),
                                 + 1./(dx[1]*dx[1]),
                                 + 1./(dx[2]*dx[2]) );
    Real dt = cfl/(2.0*coeff);

    // time = starting time in the simulation
    Real time = 0.0;

    amrex::Print() << "PARAMS| cfl value: " << cfl << "\n";
    amrex::Print() << "PARAMS| dt value from above cfl: " << dt << "\n";

    dt = 1E-3;
    //ren = ren*Real(2.0)*M_PI;
    amrex::Print() << "INFO| dt overided: " << dt << "\n";
    amrex::Print() << "INFO| Reynolds number from length scale: " << ren << "\n";
    
    // Write a plotfile of the initial data if plot_int > 0
    // (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        Export_Flow_Field("pltInit", userCtx, velCart, ba, dm, geom, time, 0);
    }

    // Setup stopping criteria
    Real Tol = 1.0e-19;

    // Setup Runge-Kutta scheme coefficients
    int RungeKuttaOrder = 4;
    GpuArray<Real, MAX_RK_ORDER> rk;
    {
        rk[0] = Real(0.25);
        rk[1] = Real(1.0)/Real(3.0);
        rk[2] = Real(0.5);
        rk[3] = Real(1.0);
    }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++   Begin time loop +++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++++++++++++++++
    for (int n = 1; n <= nsteps; ++n)
    {
        amrex::Print() << "============================ ADVANCE STEP " << n << " ============================ \n";
        // Update the time
        time = time + dt;

        // Forming boundary conditions
        userCtx.FillBoundary(geom.periodicity());
        // Enforce the physical boundary conditions
        // enforce_boundary_conditions(userCtx, type1, Nghost, bc_lo, bc_hi, n_cell);

        velCart.FillBoundary(geom.periodicity());
        // Enforce the boundary conditions again
        // enforce_boundary_conditions(velCart, type2, Nghost, bc_lo, bc_hi, n_cell);

        // Convert cartesian velocity to contravariant velocity after boundary conditions are enfoced
        // velCont is the main variable to be used in the momentum solver
        cart2cont(velCart, velCont);
        for (int comp=0; comp < AMREX_SPACEDIM; ++comp)
        {
            MultiFab::Copy(velContPrev[comp], velCont[comp], 0, 0, 1, 0);
            MultiFab::Copy(    velStar[comp], velCont[comp], 0, 0, 1, 0);
            // Assign the initial guess as the previous flow field
        }

        // Momentum solver
        // MOMENTUM |1| Setup counter
        int countIter = 0;
        Real normError = 1.0;

        // After debugging, all the code below will be modulized to the MOMENTUM module i.e.,:
        // momentum_km_runge_kutta();
        //-----------------------------------------------
        // This is the sub-iteration of the implicit RK4
        //-----------------------------------------------
        while ( countIter < IterNum && normError > Tol )
        {
            countIter++;
            amrex::Print() << "SOLVING| Momentum | performing Runge-Kutta at iteration: " << countIter
                           << " => normError = " << normError << "\n";

            // Immidiate velocity at the beginning of the RK4 sub-iteration
            for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
            {
                // Check the boundary conditions of velHat
                MultiFab::Copy(velHat[comp], velStar[comp], 0, 0, 1, 0);
            }

            // 4 sub-iterations of one RK4 iteration
            for (int sub = 0; sub < RungeKuttaOrder; ++sub )
            {
                // RUNGE-KUTTA | Calculate Cell-centered Convective terms
                convective_flux_calc(fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCart, velHat, bc_lo, bc_hi, geom, n_cell);

                // RUNGE-KUTTA | Calculate Cell-centered Viscous terms
                viscous_flux_calc(fluxViscous, velCart, geom, ren);

                // RUNGE-KUTTA | Calculate Cell-centered Pressure Gradient terms
                pressure_gradient_calc(fluxPrsGrad, userCtx, geom);

                // RUNGE-KUTTA | Calculate Cell-centered Total Flux = -fluxConvect + fluxViscous - fluxPrsGrad
                total_flux_calc(fluxTotal, fluxConvect, fluxViscous, fluxPrsGrad);

                fluxTotal.FillBoundary(geom.periodicity());
                // enforce_boundary_conditions(fluxTotal, type3, Nghost, bc_lo, bc_hi, n_cell);

                // RUNGE-KUTTA | Calculate the Face-centered Right-Hand-Side terms by averaging the Cell-centered fluxes
                momentum_righthand_side_calc(momentum_rhs, fluxTotal);
                // Is there a faster way to subtract two face-centered MultiFab?
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(velHatDiff[0]); mfi.isValid(); ++mfi)
                {
                    const Box &xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1, 0, 0)));
                    const Box &ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0, 1, 0)));
#if (AMREX_SPACEDIM > 2)
                    const Box &zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0, 0, 1)));
#endif
                    auto const &xhat_diff = velHatDiff[0].array(mfi);
                    auto const &yhat_diff = velHatDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                    auto const &zhat_diff = velHatDiff[2].array(mfi);
#endif
                    auto const &xhat = velHat[0].array(mfi);
                    auto const &yhat = velHat[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                    auto const &zhat = velHat[2].array(mfi);
#endif
                    auto const &xcont = velCont[0].array(mfi);
                    auto const &ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                    auto const &zcont = velCont[2].array(mfi);
#endif
                    amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        xhat_diff(i, j, k) = xhat(i, j, k) - xcont(i, j, k);
                    });
                    amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        yhat_diff(i, j, k) = yhat(i, j, k) - ycont(i, j, k);
                    });
#if (AMREX_SPACEDIM > 2)
                    amrex::ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        zhat_diff(i, j, k) = zhat(i, j, k) - zcont(i, j, k);
                    });
#endif
                }

                // RUNGE-KUTTA | Advance; increment momentum_rhs and use it to update velHat
                km_runge_kutta_advance(rk, sub, momentum_rhs, velHat, velHatDiff, velContDiff, velStar, dt, bc_lo, bc_hi, n_cell);

                // RUNGE-KUTTA | Update velCart from velHat
                cont2cart(velCart, velHat, geom);
                // Re-enforce the boundary conditions
                velCart.FillBoundary(geom.periodicity());
                // enforce_boundary_conditions(velCart, type4, Nghost, bc_lo, bc_hi, n_cell);

            } // RUNGE-KUTTA | END

            // RUNGE-KUTTA | Calculate the error norm
            normError = Error_Computation(velHat, velStar, velStarDiff, geom);
            //amrex::Print() << "error norm2 = " << normError << "\n";

            // Re-assign guess for the next iteration
            for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
            {
                MultiFab::Copy(velStar[comp], velHat[comp], 0, 0, 1, 0);
            }

        }// End of the Momentum loop iteration!
        //---------------------------------------
        // MOMENTUM |4| PLOTTING
        // This is just for debugging only !
        //---------------------------------------
        if (plot_int > 0 && n%plot_int == 0)
        {
            Export_Fluxes(fluxConvect, fluxViscous, fluxPrsGrad, ba, dm, geom, time, n);
        }
        amrex::Print() << "SOLVING| finished solving Momentum equation. \n";

        // Poisson solver
        //    Laplacian(\phi) = (Real(1.5)/dt)*Div(\hat{u}_i)

        // POISSON |1| Calculating the RSH
        poisson_righthand_side_calc(poisson_rhs, velHat, geom, dt);

        // POISSON |2| Init Phi at the begining of the Poisson solver
        poisson_sol.setVal(0.0);
        poisson_sol.FillBoundary(geom.periodicity());
        poisson_advance(poisson_sol, poisson_rhs, geom, ba, dm, bc);
        amrex::Print() << "SOLVING| finished solving Poisson equation. \n";

        MultiFab::Copy(userCtx, poisson_sol, 0, 1, 1, 0);
        userCtx.FillBoundary(geom.periodicity());
        // enforce_boundary_conditions(userCtx, type2, Nghost, bc_lo, bc_hi, n_cell);

        // Update the solution
        // u_i^{n+1} = \hat{u}_i- 2dt/3 * grad(\phi^{n+1})
        // (velCont = velHat - 2dt/3 grad_phi)
        // p^{n+1} = p^n  + \phi^{n+1}
        // (userCtx(comp=0) += userCtx(comp=1))
        // also update velContDiff = velCont-velContPrev
        update_solution(grad_phi, userCtx, velCont, velContPrev, velContDiff, velHat, geom, dt);
        amrex::Print() << "SOLVING| finished updating all fields \n";

        // Update velCart from the velCont solutions
        cont2cart(velCart, velCont, geom);
        // Update the halo exchange points!
        velCart.FillBoundary(geom.periodicity());
        // enforce_boundary_conditions(velCart, type4, Nghost, bc_lo, bc_hi, n_cell);

        // Before benchmarking, making sure that halo regions are updated
        analyticSol.FillBoundary(geom.periodicity());
        analytic_solution_calc(analyticSol, geom, time);

        {
            MultiFab::Copy(l2norm, velCart, 0, 0, 2, 0);
            MultiFab::Copy(l2norm, userCtx, 0, 2, 1, 0);
            MultiFab::Subtract(l2norm, analyticSol, 0, 0, 3, 0);

            long npts;
            Box my_domain = geom.Domain();
#if (AMREX_SPACEDIM == 2)
            npts = (my_domain.length(0) * my_domain.length(1));
#elif (AMREX_SPACEDIM == 3)
            npts = (my_domain.length(0) * my_domain.length(1) * my_domain.length(2));
#endif

            //amrex::Print() << my_domain.length(0) << "\n";
            //amrex::Print() << my_domain.length(1) << "\n";

            amrex::Print() << "BENCHMARKING| L2 ERROR NORM for x-velocity: " << l2norm.norm2(0)/std::sqrt(npts) << "\n";
            amrex::Print() << "BENCHMARKING| L2 ERROR NORM for y-velocity: " << l2norm.norm2(1)/std::sqrt(npts) << "\n";
            amrex::Print() << "BENCHMARKING| L2 ERROR NORM for pressure: " << l2norm.norm2(2)/std::sqrt(npts) << "\n";

            if (plot_int > 0 && n%plot_int == 0)
            {
                const std::string &analytic_export = amrex::Concatenate("pltAnalytic", n, 5);
                WriteSingleLevelPlotfile(analytic_export, analyticSol, {"U", "V", "pressure"}, geom, time, n);
                const std::string &benchmark_error_export = amrex::Concatenate("pltBenchmark", n, 5);
                WriteSingleLevelPlotfile(benchmark_error_export, l2norm, {"x-vel-err-norm", "y-vel-err-norm", "pressure-err-norm"}, geom, time, n);
            }
        } // End of benchmark

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            Export_Flow_Field("pltResults", userCtx, velCart, ba, dm, geom, time, n);
        }

        amrex::Print() << "========================== FINISH TIME: " << time << " ========================== \n";

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
