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
    MultiFab userCtxPrev(ba, dm, Ncomp, Nghost);
    MultiFab analyticSol(ba, dm, 3, 0);

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

    MultiFab poisson_rhs(ba, dm, 1, 1);
    MultiFab poisson_sol(ba, dm, 1, 1);

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
    Array<MultiFab, AMREX_SPACEDIM> velContDiff;
    // Right-Hand-Side terms of the Momentum equation have SPACEDIM as number of components, live in the face center
    Array<MultiFab, AMREX_SPACEDIM> momentum_rhs;
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
        momentum_rhs[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN1[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN2[dir].define(edge_ba, dm, 1, 0);
        fluxHalfN3[dir].define(edge_ba, dm, 1, 0);

        velImRK[dir].define(edge_ba, dm, 1, 0);
        velImPrev[dir].define(edge_ba, dm, 1, 0);
        velImDiff[dir].define(edge_ba, dm, 1, 0);

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

    amrex::Print() << "==== INITIALIZATION STEP =========== \n";
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

    dt = Real(5.0)*1E-7;
    //ren = ren*Real(2.0)*M_PI;
    amrex::Print() << "INFO| dt overided: " << dt << "\n";
    amrex::Print() << "INFO| Reynolds number from length scale: " << ren << "\n";
    
    // Write a plotfile of the initial data if plot_int > 0
    // (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        Export_Flow_Field("pltInit", userCtx, velCart, ba, dm, geom, time, 0);
    }

    // ++++++++++++++++++++ KIM AND MOINE'S RUNGE-KUTTA +++++++++++++++++++++
    amrex::Print() << "================= ADVANCING STEP  ==================== \n";
    // Setup stopping criteria
    Real Tol = 1.0e-10;
    // int IterNum = 10;

    // Setup Runge-Kutta scheme coefficients
    int RungeKuttaOrder = 4;
    GpuArray<Real, MAX_RK_ORDER> rk;
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
        // Update the time
        time = time + dt;

        MultiFab::Copy(userCtxPrev, userCtx, 0, 0, Ncomp, Nghost);
        MultiFab::Copy(velCartPrev, velCart, 0, 0, AMREX_SPACEDIM, Nghost);
        // Forming boundary conditions
        userCtx.FillBoundary(geom.periodicity());

        // Enforce the physical boundary conditions
        // enforce_boundary_conditions(userCtx, type1, Nghost, bc_lo, bc_hi, n_cell);

        // Doing the HALO exchange
        // This is important
        // If the physical boundary are not periodic
        // Then the update will not touch those grid points
        velCart.FillBoundary(geom.periodicity());

        // Enforce the boundary conditions again
        // enforce_boundary_conditions(velCart, type2, Nghost, bc_lo, bc_hi, n_cell);

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
        Real normError = 1.0;

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
                // enforce_boundary_conditions(fluxTotal, type3, Nghost, bc_lo, bc_hi, n_cell);

                // RUNGE-KUTTA | Calculate the Face-centered Right-Hand-Side terms by averaging the Cell-centered fluxes
                momentum_righthand_side_calc(momentum_rhs, fluxTotal);

                // RUNGE-KUTTA | Advance; increment momentum_rhs and use it to update velImRK
                km_runge_kutta_advance(rk, sub, momentum_rhs, velImRK, velCont, velContDiff, dt, bc_lo, bc_hi, n_cell);
                // After advance through 4 sub-step we obtain guessed velCont at next time step

                // RUNGE-KUTTA | Update velCart from velImRK
                cont2cart(velCart, velImRK, geom);
                // This updated velCart will be used again next sub-iteration
                // So, we need to re-enforce the boundary conditions
                velCart.FillBoundary(geom.periodicity());
                // enforce_boundary_conditions(velCart, type4, Nghost, bc_lo, bc_hi, n_cell);

            } // RUNGE-KUTTA | END

            normError = Error_Computation(velImRK, velImPrev, velImDiff, geom);
            amrex::Print() << "error norm2 = " << normError << "\n";

        }// End of the RK-4 LOOP Iteration!
        amrex::Print() << "SOLVING| Momentum | ending Runge-Kutta after " << countIter << " iteration(s) with convergence: " << normError << "\n";
        //cont2cart(velCart, velCont, geom);
        //Export_Flow_Field(ba, dm, geom, userCtx, velCart, n, time, "PrePoisson");

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
        //    Laplacian(\phi) = (Real(1.5)/dt)*Div(V*)

        // POISSON |1| Calculating the RSH
        poisson_righthand_side_calc(poisson_rhs, velImRK, geom, dt);
        /*
        if (plot_int > 0 && n%plot_int == 0) {
            const std::string& rhs_export = amrex::Concatenate("pltPoissonRHS", n, 5);
            WriteSingleLevelPlotfile(rhs_export, poisson_rhs, {"RHS"}, geom, time, n);
        }
        */
        //VisMF::Write(poisson_rhs, "dbgRHS");
        // POISSON |2| Init Phi at the begining of the Poisson solver
        // --Don't see why
        // ================================= DEBUGGING BELOW ===================================
        poisson_advance(poisson_sol, poisson_rhs, geom, ba, dm, bc);
        /*
        if (plot_int > 0 && n%plot_int == 0) {
            const std::string& phi_export = amrex::Concatenate("pltPoissonSolution", n, 5);
            WriteSingleLevelPlotfile(phi_export, poisson_sol, {"Phi"}, geom, time, n);
        }
        */
        amrex::Print() << "SOLVING| finished solving Poisson equation. \n";

        MultiFab::Copy(userCtx, poisson_sol, 0, 1, 1, 0);
        userCtx.FillBoundary(geom.periodicity());

        // Update the solution
        // U^{n+1} = v* + grad (\phi)
        // p^{n+1} = p  + \phi
        update_solution(grad_phi, userCtx, velCont, velImRK, geom, ba, dm, bc, dt);
        amrex::Print() << "SOLVING| finished updating all fields \n";

        // Update velCart from the velCont solutions
        cont2cart(velCart, velCont, geom);
        /*
        if (plot_int > 0 && n%plot_int == 0) {
            amrex::Print() << "SOLVING| Export middle line here \n";
            line_extract(velCart, n_cell, n, dt, geom);
        }
        */

        // This updated velCart will be used again next sub-iteration
        // So, we need to re-enforce the boundary conditions
        // Update the halo exchange points!
        velCart.FillBoundary(geom.periodicity());
        // enforce_boundary_conditions(velCart, type4, Nghost, bc_lo, bc_hi, n_cell);

        // advance will do all above steps
        amrex::Print() << "SOLVING| finished at time: " << time << "\n";

        if ( n%plot_int == 0 )
        {
            GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
            for ( MFIter mfi(analyticSol); mfi.isValid(); ++mfi )
            {
                const Box& vbx = mfi.validbox();
                auto const& analytic = analyticSol.array(mfi);
                auto const& numelv = velCart.array(mfi);
                auto const& numelp = userCtx.array(mfi);
                amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Real coordinates of the cell center
                    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
                    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];

                    // u velocity
                    analytic(i, j, k, 0) = std::sin(Real(2.0) * M_PI * x) * std::cos(Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
                    // v velocity
                    analytic(i, j, k, 1) = - std::cos(Real(2.0) * M_PI * x) * std::sin(Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
                    // pressure
                    analytic(i, j, k, 2) = - Real(0.25) * ( std::cos(Real(4.0) * M_PI * x) + std::cos(Real(4.0) * M_PI * y) ) * std::exp(-Real(16.0) * M_PI * M_PI * time);

                    // STEP 1 is to calculate the solution at the middle line
                    // middle horizontal line (MHL)
                    /*
                    if (j == n_cell/2)
                    {
                        // u velocity
                        Real mhlu = Real(0.5) * ( numelv(i, j, k, 0) + numelv(i, j-1, k, 0) );
                        Real hanu = Real(0.5) * ( analytic(i, j, k, 0) + analytic(i, j-1, k, 0) );
                        // v velocity
                        Real mhlv = Real(0.5) * ( numelv(i, j, k, 1) + numelv(i, j-1, k, 1) );
                        Real hanv = Real(0.5) * ( analytic(i, j, k, 1) + analytic(i, j-1, k, 1) );
                        // Pressure
                        Real mhlp = Real(0.5) * ( numelp(i, j, k, 0) + numelp(i, j-1, k, 0) );
                        Real hanp = Real(0.5) * ( analytic(i, j, k, 2) + analytic(i, j-1, k, 2) );
                        
                        // Write the solution to file
                        write_midline_solution(x, y, mhlu, mhlv, mhlp, hanu, hanv, hanp, n);
                    }
                    // middle vertical line (MVL)
                    if (i == n_cell/2)
                    {
                        // u velocity
                        Real mvlu = Real(0.5) * ( numelv(i, j, k, 0) + numelv(i-1, j, k, 0) );
                        Real vanu = Real(0.5) * ( analytic(i, j, k, 0) + analytic(i-1, j, k, 0) );
                        // v velocity
                        Real mvlv = Real(0.5) * ( numelv(i, j, k, 1) + numelv(i-1, j, k, 1) );
                        Real vanv = Real(0.5) * ( analytic(i, j, k, 1) + analytic(i-1, j, k, 1) );
                        // Pressure
                        Real mvlp = Real(0.5) * ( numelp(i, j, k, 0) + numelp(i-1, j, k, 0) );
                        Real vanp = Real(0.5) * ( analytic(i, j, k, 2) + analytic(i-1, j, k, 2) );
                        
                        // Write the solution to file
                        write_midline_solution(x, y, mvlu, mvlv, mvlp, vanu, vanv, vanp, n);
                    }
                    */
                });

            }

            MultiFab l2norm(ba, dm, 3, 0);
            // Comp 0 is velocity field along x-axis
            // Comp 1 is velocity field along y-axis
            // Comp 2 is pressure field
            MultiFab::Copy(l2norm, velCart, 0, 0, 2, 0);
            MultiFab::Copy(l2norm, userCtx, 1, 2, 1, 0);
            MultiFab::Subtract(l2norm, analyticSol, 0, 0, 3, 0);

            long npts;
            Box my_domain = geom.Domain();
            #if (AMREX_SPACEDIM == 2)
                npts = (my_domain.length(0)*my_domain.length(1));
            #elif (AMREX_SPACEDIM == 3)
                npts = (my_domain.length(0)*my_domain.length(1)*my_domain.length(2));
            #endif

            amrex::Print() << "BENCHMARKING| L2 ERROR NORM for x-velocity: " << l2norm.norm2(0)/std::sqrt(npts) << "\n";
            amrex::Print() << "BENCHMARKING| L2 ERROR NORM for y-velocity: " << l2norm.norm2(1)/std::sqrt(npts) << "\n";
            amrex::Print() << "BENCHMARKING| L2 ERROR NORM for pressure: " << l2norm.norm2(2)/std::sqrt(npts) << "\n";
            if (plot_int > 0)
            {
                const std::string& analytic_export = amrex::Concatenate("pltAnalytic", n, 5);
                WriteSingleLevelPlotfile(analytic_export, analyticSol, {"U", "V", "pressure"}, geom, time, n);
            }
        }

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            Export_Flow_Field("pltResults", userCtx, velCart, ba, dm, geom, time, n);
        }

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
