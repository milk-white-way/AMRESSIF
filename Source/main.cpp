/**
 * @file main.cpp
 * @author Thien-Tam Nguyen (tam.thien.nguyen@ndsu.edu)
 * @brief This is the main code
 * @version 0.3
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
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
    Real ren, vis, cfl, fixed_dt;

    // Physical boundary condition mapping
    // 0 is periodic
    // -1 is no-slip
    // 1 is slip
    Vector<int> phy_bc_lo(AMREX_SPACEDIM, 0);
    Vector<int> phy_bc_hi(AMREX_SPACEDIM, 0);

    int target_resolution;

    Real momentum_tolerance;

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

        fixed_dt = -1.0;
        pp.query("fixed_dt",fixed_dt);

        // Parsing the Reynolds number and viscosity from input file also
        pp.get("ren", ren);
        pp.get("vis", vis);

        // Parsing boundary condition from input file
        pp.queryarr("phy_bc_lo", phy_bc_lo);
        pp.queryarr("phy_bc_hi", phy_bc_hi);

        // Parsing the target resolution from input file
        target_resolution = -1;
        pp.query("target_resolution", target_resolution);

        momentum_tolerance = 1.e-10;
        pp.query("momentum_tolerance", momentum_tolerance);
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
    // The userCtx has 02 components:
    // userCtx(0) = Pressure
    // userCtx(1) = Phi
    int Ncomp = 2;

    // Calculating number of step to reach the targeted resolution
    int nsteps_target = target_resolution == -1 ? 0 : n_cell/target_resolution - 1;
    amrex::Print() << "INFO| target resolution: " << target_resolution << "\n";
    amrex::Print() << "INFO| number of steps to reach the target resolution: " << nsteps_target << "\n";

    // How Boxes are distrubuted among MPI processes
    // Distribution mapping between the processors
    DistributionMapping dm(ba);

    /*
     * -----------------------
     *   Volume center
     *  ----------------------
     *  |                   |
     *  |                   |
     *  |         0         |
     *  |                   |
     *  |                   |
     *  ----------------------
     */

    // User Contex MultiFab contains 2 components, pressure and Phi, at the cell center
    MultiFab userCtx(ba, dm, Ncomp, Nghost);

    // Cartesian velocities have SPACEDIM as number of components, live in the cell center
    MultiFab velCart(ba, dm, AMREX_SPACEDIM, Nghost);
    MultiFab velCartPrev(ba, dm, AMREX_SPACEDIM, Nghost);

    // Three type of fluxes contributing the the total flux live in the cell center
    MultiFab fluxConvect(ba, dm, AMREX_SPACEDIM, 0);
    MultiFab fluxViscous(ba, dm, AMREX_SPACEDIM, 0);
    MultiFab fluxPrsGrad(ba, dm, AMREX_SPACEDIM, 0);
    MultiFab fluxTotal(ba, dm, AMREX_SPACEDIM, 1);

    MultiFab cc_grad_phi(ba, dm, AMREX_SPACEDIM, Nghost);    

    MultiFab poisson_rhs(ba, dm, 1, 1);
    MultiFab poisson_sol(ba, dm, 1, 1);

    MultiFab cc_analytical_diff(ba, dm, 3, 0);
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
            // If phy_bc_lo == 0 then 209
            //Internal Dirichlet Periodic Boundary conditions, or bc_lo = bc_hi = 0
            if (phy_bc_lo[idim] == 0) {
                bc[n].setLo(idim, BCType::int_dir);
            }
                //First Order Extrapolation for Neumann boundary conditions or bc_lo, bc_hi = 2
            else if (std::abs(phy_bc_lo[idim]) == 1) {
                bc[n].setLo(idim, BCType::foextrap);
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            //Internal Dirichlet Periodic Boundary conditions, or bc_lo = bc_hi = 0
            if (phy_bc_hi[idim] == 0) {
                bc[n].setHi(idim, BCType::int_dir);
            }
                //First Order Extrapolation for Neumann boundary conditions or bc_lo, bc_hi = 2
            else if (std::abs(phy_bc_hi[idim]) == 1) {
                bc[n].setHi(idim, BCType::foextrap);
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
     *              ______________________
     *             |                      |
     *             |                      |
     *             |                      |
     *             |----> velCont[1]      |
     *             |                      |
     *             |                      |
     *             |________----> ________|
     *                      velCont[2]
     *
     */

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
    Array<MultiFab, AMREX_SPACEDIM> velStar;
    Array<MultiFab, AMREX_SPACEDIM> velStarDiff;

    // gradient variables
    Array<MultiFab, AMREX_SPACEDIM> array_grad_p;
    Array<MultiFab, AMREX_SPACEDIM> array_grad_phi;

    Array<MultiFab, AMREX_SPACEDIM> array_analytical_vel;

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

        velStar[dir].define(edge_ba, dm, 1, 0);
        velStarDiff[dir].define(edge_ba, dm, 1, 0);

        array_grad_p[dir].define(edge_ba, dm, 1, 0);
        array_grad_phi[dir].define(edge_ba, dm, 1, 0);

        array_analytical_vel[dir].define(edge_ba, dm, 1, 0);
    }

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    Real coeff = AMREX_D_TERM( 1./(dx[0]*dx[0]),
                             + 1./(dx[1]*dx[1]),
                             + 1./(dx[2]*dx[2]) );
    amrex::Real dt = cfl/(2.0*coeff);

    // time = starting time in the simulation
    amrex::Real time = 0.0;

    amrex::Print() << "PARAMS| cfl value: " << cfl << "\n";
    amrex::Print() << "PARAMS| dt value from above cfl: " << dt << "\n";

    if (fixed_dt != -1.0) {
        dt = fixed_dt;
        amrex::Print() << "INFO| dt overidden with fixed_dt: " << dt << "\n";
    }
    //amrex::Real d_tau = Real(0.9889)*dt;
    amrex::Real d_tau = Real(0.4321)*dt;

    //ren = ren*Real(2.0)*M_PI;
    amrex::Print() << "INFO| Reynolds number from length scale: " << ren << "\n";

    // Print desired variables for debugging
    amrex::Print() << "INFO| number of dimensions: " << AMREX_SPACEDIM << "\n";
    amrex::Print() << "INFO| geometry: " << geom << "\n";
    amrex::Print() << "PARAMS| number of ghost cells for each array: " << Nghost << "\n";
    amrex::Print() << "PARAMS| number of components for each array: " << Ncomp << "\n";

    /**--------------------------------------------------------------------------------------
     * =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Initialization =-=-=-=-=-=-=-=-=-=-=-=-=-=-------------
     *--------------------------------------------------------------------------------------
     */

    amrex::Print() << "========================= INITIALIZATION STEP ========================= \n";
    // Current: Taylor-Green Vortex initial conditions
    // How partial periodic boundary conditions can be deployed?
    staggered_grid_init(userCtx, velCont, velContPrev, velContDiff, velCart, velCartPrev, geom, Nghost, phy_bc_lo, phy_bc_hi, time, dt, n_cell);
    // Quickly init other fields as zero
    fluxConvect.setVal(0.0);
    fluxViscous.setVal(0.0);
    fluxPrsGrad.setVal(0.0);
    fluxTotal.setVal(0.0);
    cc_grad_phi.setVal(0.0);
    poisson_rhs.setVal(0.0);
    poisson_sol.setVal(0.0);
    for (int comp=0; comp < AMREX_SPACEDIM; ++comp)
    {
        array_grad_p[comp].setVal(0.0);
        array_grad_phi[comp].setVal(0.0);
        momentum_rhs[comp].setVal(0.0);
        fluxHalfN1[comp].setVal(0.0);
        fluxHalfN2[comp].setVal(0.0);
        fluxHalfN3[comp].setVal(0.0);
    }

    //gradient_calc_approach1(fluxPrsGrad, cc_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    gradient_calc_approach2(array_grad_p, array_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

    // Write a plotfile of the initial data if plot_int > 0
    // (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        Export_Flow_Field("pltInit", userCtx, velCart, ba, dm, geom, time, 0);
        Export_Flow_Field("pltInitPrev", userCtx, velCartPrev, ba, dm, geom, time, 0);
    }

    // Setup Runge-Kutta scheme coefficients
    int RungeKuttaOrder = 4;
    GpuArray<Real, MAX_RK_ORDER> rk;
    {
        rk[0] = d_tau * Real(0.25);
        rk[1] = d_tau *(Real(1.0)/Real(3.0));
        rk[2] = d_tau * Real(0.5);
        rk[3] = d_tau * Real(1.0);
    }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++   Begin time loop +++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++++++++++++++++
    for (int n = 1; n <= nsteps; ++n)
    {
        amrex::Print() << "============================ ADVANCE STEP " << n << " ============================ \n";
        // Update the time
        time = time + dt;

        for (int comp=0; comp < AMREX_SPACEDIM; ++comp)
        {
            // Save the previous velocity field to update the diff field later
            MultiFab::Copy(velContPrev[comp], velCont[comp], 0, 0, 1, 0);
        }

        // Momentum solver
        // MOMENTUM |1| Setup counter
        int countIter = 0;
        Real normError = 1.e19;

        //-----------------------------------------------
        // This is the sub-iteration of the implicit RK4
        //-----------------------------------------------
        while ( normError > momentum_tolerance )
        {
            amrex::Print() << "SOLVING| Momentum | performing Runge-Kutta at pseudo step: " << countIter
                           << " => normError = " << normError << "\n";
            
            for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
            {
                // Assign the initial guess as the previous flow field
                MultiFab::Copy(velStar[comp], velCont[comp], 0, 0, 1, 0);
                velStarDiff[comp].setVal(0.0);
            }

            // 4 sub-iterations of one RK4 iteration
            for (int sub = 0; sub < RungeKuttaOrder; ++sub )
            {
#if (EXPLICIT_MOMENTUM == 1)
                // Explicit time marching
                sub = 3;
#endif 
                // ------------------------- FLUX CALCULATION -------------------------
                convective_flux_calc(fluxTotal, fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCart, velStar, phy_bc_lo, phy_bc_hi, geom, n_cell, sub);
                viscous_flux_calc(fluxTotal, fluxViscous, velCart, geom, ren);
                momentum_righthand_side_calc(fluxTotal, array_grad_p, momentum_rhs, phy_bc_lo, phy_bc_hi, geom);

                // --------------------------- MOMENTUM SOLVER ---------------------------
                runge_kutta4_pseudo_time_stepping(rk, sub, momentum_rhs, velStar, velCont, velContDiff, velContPrev, velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell, dt);
                //explicit_time_marching(momentum_rhs, velStar, velCont, velContDiff, velContPrev, velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell, dt);
                fluxTotal.setVal(0.0);
            } // RUNGE-KUTTA | END

            normError = Error_Computation(velCont, velStar, velStarDiff, geom);
            //amrex::Print() << "error norm2 = " << normError << "\n";

            // Re-assign guess for the next iteration
            for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
            {
                MultiFab::Copy(velCont[comp], velStar[comp], 0, 0, 1, 0);
            }
            countIter++;
            // Handler for blowing-up situation
            if (countIter > IterNum) {
                amrex::Print() << "Exceeded number of momenum iterations; exiting loop\n";
                break;
            }
            if ( normError > 1.e-1 )
            {
                amrex::Print() << "WARNING| Momentum | normError = " << normError << "\n";
                break;
            }
        }// End of the Momentum loop iteration!
        //---------------------------------------
        // MOMENTUM |4| PLOTTING
        // This is just for debugging only !
        shift_face_to_center(velCartPrev, velCont);
        if (plot_int > 0 && n%plot_int == 0)
        {
            Export_Fluxes(fluxConvect, fluxViscous, fluxPrsGrad, ba, dm, geom, time, n);
            const std::string &momentum_export = amrex::Concatenate("pltMomentum", n, 5);
            WriteSingleLevelPlotfile(momentum_export, velCartPrev, {"ucont-star", "vcont-star"}, geom, time, n);
        }
        //---------------------------------------
        amrex::Print() << "\nSOLVING| finished solving Momentum equation. \n";
        amrex::Print() << "\n";

        // Poisson solver
        //    Laplacian(\phi) = (Real(1.5)/dt)*Div(u_i^*)
        // POISSON |1| Calculating the RSH
        poisson_righthand_side_calc(poisson_rhs, velCont, geom, dt);
        // POISSON |2| Init Phi at the begining of the Poisson solver
        poisson_advance(poisson_sol, poisson_rhs, geom, ba, dm, bc);
        amrex::Print() << "\nSOLVING| finished solving Poisson equation. \n";
        amrex::Print() << "\n";
        if (plot_int > 0 && n%plot_int == 0)
        {
            const std::string &rhs_export = amrex::Concatenate("pltPoisson", n, 5);
            WriteSingleLevelPlotfile(rhs_export, poisson_sol, {"phi"}, geom, time, n);
        }
        MultiFab::Copy(userCtx, poisson_sol, 0, 1, 1, 0);

        // Update the solution
        // u_i^{n+1} = u_i^*- 2dt/3 * grad(\phi^{n+1})
        // p^{n+1} = p^n  + \phi^{n+1}
        // also update velContDiff = velCont-velContPrev
        update_solution(array_grad_phi, array_grad_phi, fluxPrsGrad, cc_grad_phi, userCtx, velCart, velCont, velContPrev, velContDiff, geom, dt, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

        amrex::Print() << "SOLVING| finished updating all fields \n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            // Copy the numerical pressure field
            MultiFab::Copy(cc_analytical_diff, userCtx, 0, 2, 1, 0);
            // Shift numerical face-centered velocity to cell-centered
            shift_face_to_center(cc_analytical_diff, velCont);
            const std::string &export_numel_sol = amrex::Concatenate("pltNumel", n, 5);
            WriteSingleLevelPlotfile(export_numel_sol, cc_analytical_diff, {"ucont", "vcont", "pressure"}, geom, time, n);
            
            // Calculate the analytical cell-center velocity and pressure
            cc_analytical_calc(cc_analytical_diff, geom, time);
            // Calculate the analytical face-center velocity
            array_analytical_vel_calc(array_analytical_vel, geom, time);

            // Shifting analytical face-centered velocity to cell-centered
            shift_face_to_center(cc_analytical_diff, array_analytical_vel);
            const std::string &export_exact_sol = amrex::Concatenate("pltExact", n, 5);
            WriteSingleLevelPlotfile(export_exact_sol, cc_analytical_diff, {"ucont", "vcont", "pressure"}, geom, time, n);

            MultiFab::Subtract(cc_analytical_diff, userCtx, 0, 2, 1, 0);
            for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
            {
                MultiFab::Subtract(array_analytical_vel[dir], velCont[dir], 0, 0, 1, 0);
            }

            long npts;
            Box my_domain = geom.Domain();
#if (AMREX_SPACEDIM == 2)
            npts = (my_domain.length(0) * my_domain.length(1));
#elif (AMREX_SPACEDIM == 3)
            npts = (my_domain.length(0) * my_domain.length(1) * my_domain.length(2));
#endif

            Vector<Real> l1_sum(AMREX_SPACEDIM);
            Vector<Real> l2_sum(AMREX_SPACEDIM);

            SumAbsStag(array_analytical_vel, l1_sum);
            StagL2Norm(array_analytical_vel, 0, l2_sum);

            amrex::Print() << "_________________________________________________________________________________________ \n";
            amrex::Print() << "|\t BENCHMARKING| L0 ERROR NORM for contravariant x-velocity: " << array_analytical_vel[0].norm0(0) << "\t|\n";
            amrex::Print() << "|\t BENCHMARKING| L0 ERROR NORM for contravariant y-velocity: " << array_analytical_vel[1].norm0(0) << "\t|\n";
            amrex::Print() << "|\t BENCHMARKING| L0 ERROR NORM for pressure: " << cc_analytical_diff.norm0(2) << "\t \t \t|\n";
            amrex::Print() << "| --------------------------------------------------------------------------------------| \n";
            amrex::Print() << "|\t BENCHMARKING| L1 ERROR NORM for contravariant x-velocity: " << l1_sum[0]/npts << "\t|\n";
            amrex::Print() << "|\t BENCHMARKING| L1 ERROR NORM for contravariant y-velocity: " << l1_sum[1]/npts << "\t|\n";
            amrex::Print() << "|\t BENCHMARKING| L1 ERROR NORM for pressure: " << cc_analytical_diff.norm1(2)/npts << "\t \t \t|\n";
            amrex::Print() << "| --------------------------------------------------------------------------------------| \n";
            amrex::Print() << "|\t BENCHMARKING| L2 ERROR NORM for contravariant x-velocity: " << l2_sum[0]/std::sqrt(npts) << "\t|\n";
            amrex::Print() << "|\t BENCHMARKING| L2 ERROR NORM for contravariant y-velocity: " << l2_sum[1]/std::sqrt(npts) << "\t|\n";
            amrex::Print() << "|\t BENCHMARKING| L2 ERROR NORM for pressure: " << cc_analytical_diff.norm2(2)/std::sqrt(npts) << "\t \t \t|\n";
            amrex::Print() << "|_______________________________________________________________________________________| \n";

            /*
            std::string hline_filename = std::to_string(n_cell) + "n_cell-output_hline_" + std::to_string(n) + ".txt";
            std::string vline_filename = std::to_string(n_cell) + "n_cell-output_vline_" + std::to_string(n) + ".txt";

            // Export predefined line
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
            GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

            // Here is an interpolation algorithm for contravariant velocity components
            // Only inner domain values are interpolated using 6 points stencil
            amrex::Print() << "==================== INTERPOLATION TO TARGET RESOLUTION ==================== \n";
            if ( nsteps_target == 0 ) {
                amrex::Print() << "INFO| Already at the target resolution. \n";
            } else {
                int current_resolution = n_cell;

                for ( int coarse = 1; coarse <= nsteps_target; ++coarse )
                {
                    current_resolution = n_cell/(2*coarse);
                    amrex::Print() << "INFO| At level " << coarse << "/" << nsteps_target << " and n_cell = " << current_resolution << "\n";
                    
                    // Interpolation at each coarsening step
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
                    {
                        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
                        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
                        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
                        auto const& vel_cont_x = velCont[0].array(mfi);
                        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                        auto const& vel_cont_z = velCont[2].array(mfi);
#endif
                        amrex::ParallelFor(xbx,
                                        [=] AMREX_GPU_DEVICE(int i, int j, int k){
                            if ( i > 0 && i < xbx.bigEnd(0) && i%2 == 0 && j%2 != 0 )
                            {
                                
                                auto const& interp_vel_cont_x = amrex::Real(1/6)*( vel_cont_x(i, j, k) + vel_cont_x(i+1, j, k) + vel_cont_x(i-1, j, k) + vel_cont_x(i, j-1, k) + vel_cont_x(i+1, j-1, k) + vel_cont_x(i-1, j-1, k) );

                                if ( current_resolution == target_resolution ) {
                                    int const& ii = i/2;
                                    amrex::Real x = prob_lo[0] + (ii + Real(0.0)) * dx[0];
                                    amrex::Print() << x << "\n";
                                    if ( x == 0.250 ) {
                                        write_interp_line_solution(interp_vel_cont_x, hline_filename);
                                    }
                                }
                            }
                        });
                        amrex::ParallelFor(ybx,
                                        [=] AMREX_GPU_DEVICE(int i, int j, int k){
                            if ( j > 0 && j < ybx.bigEnd(1) && j%2 == 0 && i%2 != 0 )
                            {
                                auto const& interp_vel_cont_y = amrex::Real(1/6)*( vel_cont_y(i, j, k) + vel_cont_y(i, j+1, k) + vel_cont_y(i, j-1, k) + vel_cont_y(i-1, j, k) + vel_cont_y(i-1, j+1, k) + vel_cont_y(i-1, j-1, k) );

                                if ( current_resolution == target_resolution ) {
                                    int const& jj = j/2;
                                    amrex::Real y = prob_lo[1] + (jj + Real(0.0)) * dx[1];
                                    amrex::Print() << y << "\n";
                                    if ( y == 0.250 ) {
                                        write_interp_line_solution(interp_vel_cont_y, vline_filename);
                                    }
                                }
                            }
                        });
#if (AMREX_SPACEDIM > 2)
                        amrex::ParallelFor(zbx,
                                        [=] AMREX_GPU_DEVICE(int i, int j, int k){
                            vel_cont_z(i, j, k) = amrex::Real(1.0);
                        });
#endif
                    }
                }
            }

            // Initialize velocity components at face centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
            {
                const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
                const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
                const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
                auto const& vel_cont_x = velCont[0].array(mfi);
                auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& vel_cont_z = velCont[2].array(mfi);
#endif
                amrex::ParallelFor(xbx,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k){
                    amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
                    amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
            
                    if ( x == 0.250 ) {
                        //amrex::Print() << x << " ";
                        //amrex::Print() << y << " ";
                        //amrex::Print() << vel_cont_x(i, j, k) << "\n";

                        amrex::Real const& analytical_vel_cont_x = std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y);

                        write_exact_line_solution(x, y, vel_cont_x(i, j, k), analytical_vel_cont_x, hline_filename);
                    }
                });
                amrex::ParallelFor(ybx,
                                  [=] AMREX_GPU_DEVICE(int i, int j, int k){
                    amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                    amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];

                    if ( y == 0.250 ) {
                        //amrex::Print() << x << " ";
                        //amrex::Print() << y << " ";
                        //amrex::Print() << vel_cont_y(i, j, k) << "\n";

                        amrex::Real const& analytical_vel_cont_y = -std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y);

                        write_exact_line_solution(x, y, vel_cont_y(i, j, k), analytical_vel_cont_y, vline_filename);
                    }
                });
#if (AMREX_SPACEDIM > 2)
                amrex::ParallelFor(zbx,
                                  [=] AMREX_GPU_DEVICE(int i, int j, int k){
                    vel_cont_z(i, j, k) = amrex::Real(1.0);
                });
#endif
            }
        */
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
