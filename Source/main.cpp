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

#include "./meTh_Fractional_Time/mdl_initialization/fn_init.H"
#include "./meTh_Fractional_Time/mdl_momentum/fn_momentum.H"
#include "./meTh_Fractional_Time/mdl_poisson/fn_poisson.H"
#include "./meTh_Fractional_Time/mdl_advance/fn_advance.H"

#include "./meTh_Fractional_Time/utl_boundary_conditions/fn_fill_ghostcells.H"
#include "./meTh_Fractional_Time/utl_conversion/fn_cart2cont.H"
#include "./meTh_Fractional_Time/utl_conversion/fn_cont2cart.H"

#include "main_main.H"
#include "myfunc.H"

using namespace amrex;

// ============================== MAIN SECTION ==============================
int main (int argc, char* argv[])
{
    // Initialization
    amrex::Initialize(argc,argv);
    // Solver
    main_main();
    // Finalization
    amrex::Finalize();
    // Error handeling
    return 0;
}

// ============================== SOLVER SECTION ==============================
void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions
    // These are stock params for AMReX
    int n_cell, max_grid_size, nsteps, plot_int;

    // Porting extra params from Julian code
    Real ren, vis;

    // Physical boundary condition mapping
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
        pp.get("n_cell", n_cell);
        amrex::Print() << "INFO| number of cells in each side of the domain: " << n_cell << "\n";

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size", max_grid_size);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int", plot_int);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps", nsteps);

        // Parsing the Reynolds number and viscosity from input file also
        pp.get("ren", ren);
        pp.get("vis", vis);

        // Parsing boundary condition from input file
        pp.queryarr("phy_bc_lo", phy_bc_lo);
        pp.queryarr("phy_bc_hi", phy_bc_hi);

        pp.queryarr("bc_lo", bc_lo);
        pp.queryarr("bc_hi", bc_hi);
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
    DistributionMapping dm(ba);

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

    // Contravariant velocities live in the face center
    Array<MultiFab, AMREX_SPACEDIM> velCont;
    Array<MultiFab, AMREX_SPACEDIM> velContPrev;
    Array<MultiFab, AMREX_SPACEDIM> velContDiff;

    // Right-Hand-Side terms of the Momentum equation have SPACEDIM as number of components, live in the face center
    Array<MultiFab, AMREX_SPACEDIM> rhs;
    // Half-node fluxes contribute to implementation of QUICK scheme in calculating the convective flux
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN1;
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN2;
//#if (AMREX_SPACEDIM > 2)
    Array<MultiFab, AMREX_SPACEDIM> fluxHalfN3;
//#endif
    // The physical quantities living at the face center need to be blowed out one once in the respective direction
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        BoxArray edge_ba = ba;
        edge_ba.surroundingNodes(dir);

        velCont[dir].define(edge_ba, dm, 1, 0);
        velContPrev[dir].define(edge_ba, dm, 1, 0);
        velContDiff[dir].define(edge_ba, dm, 1, 0);

        rhs[dir].define(edge_ba, dm, 1, 0);

        fluxHalfN1[dir].define(edge_ba, dm, 1, 0);
        // fluxHalfN1[0] is flux_xcont_xface
        // fluxHalfN1[1] is flux_xcont_yface
        // fluxHalfN1[2] is flux_xcont_zface
        fluxHalfN2[dir].define(edge_ba, dm, 1, 0);
        // fluxHalfN2[0] is flux_ycont_xface
        // fluxHalfN2[1] is flux_ycont_yface
        // fluxHalfN2[2] is flux_ycont_zface
//#if (AMREX_SPACEDIM > 2)
        fluxHalfN3[dir].define(edge_ba, dm, 1, 0);
        // fluxHalfN3[0] is flux_zcont_xface
        // fluxHalfN3[1] is flux_zcont_yface
        // fluxHalfN3[2] is flux_zcont_zface
//#endif
    }

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

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

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Initialization =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    amrex::Print() << "===================== INITIALIZATION STEP ===================== \n";
    // Current: Taylor-Green Vortex initial conditions
    // How partial periodic boundary conditions can be deployed?
    init(userCtx, velCart, velCartDiff, velContDiff, geom);

    Real cfl = 0.9;
    Real coeff = AMREX_D_TERM(   1./(dx[0]*dx[0]),
                               + 1./(dx[1]*dx[1]),
                               + 1./(dx[2]*dx[2]) );
    Real dt = cfl/(2.0*coeff);

    // time = starting time in the simulation
    Real time = 0.0;

    amrex::Print() << "PARAMS| cfl number is set to: " << cfl << "\n";
    amrex::Print() << "PARAMS| dt from above cfl: " << dt << "\n";

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Plotting =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Initial state

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
    int IterNum = 10;

    // Setup Runge-Kutta scheme coefficients
int RungeKuttaOrder = 4;
Vector<Real> rk(RungeKuttaOrder, 0);
    {
        rk[0] = Real(0.25);
        rk[1] = Real(1.0)/Real(3.0);
        rk[2] = Real(0.5);
        rk[3] = Real(1.0);
    }

    VisMF::Write(userCtx, "a_userCtx");

    //for (int n = 1; n <= 1; ++n)
    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(userCtxPrev, userCtx, 0, 0, 1, 0);
        MultiFab::Copy(userCtxPrev, userCtx, 1, 1, 1, 0);

        MultiFab::Copy(velCartPrev, velCart, 0, 0, AMREX_SPACEDIM, 0);

        velCartPrev.FillBoundary(geom.periodicity());
        userCtxPrev.FillBoundary(geom.periodicity());

        manual_fill_ghost_cells(velCartPrev, userCtxPrev, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

        cart2cont(velCartPrev, velCont);

        // Momentum solver
        // momentum_km_runge_kutta(rhs, fluxTotal,
        //                         fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3,
        //                         fluxViscous, fluxPrsGrad,
        //                         userCtxPrev, velCartPrev,
        //                         velCont, velContPrev, velContDiff,
        //                         rk, RungeKuttaOrder, countIter, normError,
        //                         geom, ren, dt, n_cell,
        //                         IterNum, Tol);

        // Setup counter
        int countIter = 0;
        Real normError = 1.0e1;

        while ( countIter < IterNum && normError > Tol )
        {
            countIter++;
            amrex::Print() << "SOLVING| Momentum | performing Runge-Kutta at iteration: " << countIter << "\n";

            copy_contravariant_velocity(velCont, velContPrev);

            for (int sub = 0; sub < RungeKuttaOrder; ++sub )
            {
                // Calculate Convective terms
                convective_flux_calc(fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCartPrev, velContPrev, geom, n_cell);
                fluxConvect.FillBoundary(geom.periodicity());
                enforce_boundary_conditions(fluxConvect, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

                // Calculate Viscous terms
                viscous_flux_calc(fluxViscous, velCartPrev, geom, ren, n_cell);
                fluxViscous.FillBoundary(geom.periodicity());
                enforce_boundary_conditions(fluxViscous, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

                // Calculate Pressure Gradient terms
                pressure_gradient_calc(fluxPrsGrad, userCtxPrev, geom, n_cell);
                fluxPrsGrad.FillBoundary(geom.periodicity());
                enforce_boundary_conditions(fluxPrsGrad, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

                // Calculate Total Flux
                total_flux_calc(fluxTotal, fluxConvect, fluxViscous, fluxPrsGrad, n_cell);
                fluxTotal.FillBoundary(geom.periodicity());
                enforce_boundary_conditions(fluxTotal, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
                // Calculate the Righ Hand Side
                righthand_side_calc(rhs, fluxTotal, n_cell);

                // Update new contravariant velocities
                for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
                {
                    const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
                    const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
                    const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
                    auto const& xcont = velCont[0].array(mfi);
                    auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                    auto const& zcont = velCont[2].array(mfi);
#endif
                    auto const& xprev = velContPrev[0].array(mfi);
                    auto const& yprev = velContPrev[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                    auto const& zprev = velContPrev[2].array(mfi);
#endif
                    auto const& xdiff = velContDiff[0].array(mfi);
                    auto const& ydiff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                    auto const& zdiff = velContDiff[2].array(mfi);
#endif
                    auto const& xrhs = rhs[0].array(mfi);
                    auto const& yrhs = rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                    auto const& zrhs = rhs[2].array(mfi);
#endif

                    amrex::ParallelFor(xbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k){
                        // Itermidiate velocity
                        Real xhat = xcont(i, j, k);
                        // Corection for right-hand-side term
                        xrhs(i, j, k) = xrhs(i, j, k) - (Real(0.5)/dt)*(xhat - xrhs(i, j, k)) + (Real(0.5)/dt)*(xdiff(i, j, k));
                        // RK4 substep to update the immediate velocity
                        if ( i==0 || i==(n_cell) ) {
                            xhat = amrex::Real(0.0);
                        } else {
                            xhat = xcont(i, j, k) + rk[sub]*dt*xrhs(i,j,k);
                        }
                        xcont(i, j, k) = xhat;
                    });

                    amrex::ParallelFor(ybx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k){
                        Real yhat = ycont(i, j, k);

                        yrhs(i, j, k) = yrhs(i, j, k) - (Real(0.5)/dt)*(yhat - yrhs(i, j, k)) + (Real(0.5)/dt)*(ydiff(i, j, k));

                        if ( j==0 || j==(n_cell) ) {
                            yhat = amrex::Real(0.0);
                        } else {
                            yhat = ycont(i, j, k) + rk[sub]*dt*yrhs(i,j,k);
                        }
                        ycont(i, j, k) = yhat;
                    });

#if (AMREX_SPACEDIM > 2)
                    amrex::ParallelFor(zbx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k){
                        Real zhat = zcont(i, j, k);

                        zrhs(i, j, k) = zrhs(i, j, k) - (Real(0.5)/dt)*(zhat - zrhs(i, j, k)) + (Real(0.5)/dt)*(zdiff(i, j, k));

                        if ( k==0 || k==(n_cell) ) {
                            zhat = amrex::Real(0.0);
                        } else {
                            zhat = zcont(i, j, k) + rk[sub]*dt*zrhs(i,j,k);
                        }
                        zcont(i, j, k) = zhat;
                    });
#endif
                }
            }
            if (plot_int > 0 ) {
                const std::string& plt_conv_flux = amrex::Concatenate("pltConvectFlux", n, 5);
                WriteSingleLevelPlotfile(plt_conv_flux, fluxConvect, {"conv_fluxx", "conv_fluxy"}, geom, time, n);
                const std::string& plt_visc_flux = amrex::Concatenate("pltViscousFlux", n, 5);
                WriteSingleLevelPlotfile(plt_visc_flux, fluxViscous, {"visc_fluxx", "visc_fluxy"}, geom, time, n);
                const std::string& plt_pres_grad = amrex::Concatenate("pltPressureGrad", n, 5);
                WriteSingleLevelPlotfile(plt_pres_grad, fluxPrsGrad, {"pres_gradx", "pres_grady"}, geom, time, n);
            }
            // Update contravariant velocity difference
            for ( MFIter mfi(velContDiff[0]); mfi.isValid(); ++mfi )
            {
                const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
                const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
                const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
                auto const& xcont = velCont[0].array(mfi);
                auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& zcont = velCont[2].array(mfi);
#endif
                auto const& xprev = velContPrev[0].array(mfi);
                auto const& yprev = velContPrev[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& zprev = velContPrev[2].array(mfi);
#endif
                auto const& xdiff = velContDiff[0].array(mfi);
                auto const& ydiff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& zdiff = velContDiff[2].array(mfi);
#endif
                amrex::ParallelFor(xbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    xdiff(i, j, k) = xcont(i, j, k) - xprev(i, j, k);
                });

                amrex::ParallelFor(ybx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k){
                    ydiff(i, j, k) = ycont(i, j, k) - yprev(i, j, k);
                });

#if (AMREX_SPACEDIM > 2)
                amrex::ParallelFor(zbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k){
                    zdiff(i, j, k) = zcont(i, j, k) - zprev(i, j, k);
                });
#endif
            }
            // Update error stopping condition
            Real xerror = velContDiff[0].norm2(0, geom.periodicity());
            Real yerror = velContDiff[1].norm2(0, geom.periodicity());

            normError = std::max(xerror, yerror);
#if (AMREX_SPACEDIM > 2)
            Real zerror = velContDiff[2].norm2(0, geom.periodicity());
            normError = std::max(normError, zerror);
#endif
        }
        amrex::Print() << "SOLVING| Momentum | ending after " << countIter << " iteration(s) with convergence: " << normError << "\n";

        // PSEUDO-CODE: POISSON SOLVER HERE

        // advance will do all above steps
        //advance(userCtxOld, userCtx, flux, dt, geom);
        time = time + dt;

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "INFO| End Advanced Step " << n << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            const std::string& plt_pressure_file = amrex::Concatenate("pltPressue", n, 5);
            const std::string& plt_velfield_file = amrex::Concatenate("pltVelocity", n, 5);
            WriteSingleLevelPlotfile(plt_pressure_file, userCtx, {"pressure", "phi"}, geom, time, n);
            WriteSingleLevelPlotfile(plt_velfield_file, velCart, {"U", "V"}, geom, time, n);
            // Extra
            const std::string& plt_totaflux_field = amrex::Concatenate("pltTotalFlux", n, 5);
            WriteSingleLevelPlotfile(plt_totaflux_field, fluxTotal, {"fluxx", "fluxy"}, geom, time, n);
        }
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    auto stop_time = ParallelDescriptor::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}
