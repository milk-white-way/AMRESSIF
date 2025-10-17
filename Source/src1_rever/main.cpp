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
#include <GMRES_Poisson.H>

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
   	amrex::Print() << "Happy AMRESSIF~ing!\n";
   	amrex::Finalize();
   	return 0;
}

void main_main ()
{
	// What time is it now?  We'll use this to compute total run time.
	auto strt_time = ParallelDescriptor::second();

	// AMREX_SPACEDIM: number of dimensions
	int n_cell; 		// number of cells on each side of a square (or cubic) domain 
	int max_grid_size; 	// The domain is broken into boxes of size max_grid_size
	int nsteps; 		// Steps to run in the simulation  

	int plot_int; 		// How often to write plot files			; input <=0 to turn off
	int txt_int;   		// How often to write text files			; input <=0 to turn off
	int chk_int; 		// How often to write checkpoint files ; input <=0 to turn off
	int chk_out; 		// Checkpoint frame to load

	int IterNum, PSEUDO_TIMESTEPPING;

	Real ren; 			// Reynolds number
	Real vis;   	  	// Kinematic Viscosity
	Real cfl;   		// CFL number
	Real fixed_dt; 		// Input time step (more preferred in CFD compared to auto-calculated)

	// Physical boundary condition
	Vector<int> phy_bc_lo(AMREX_SPACEDIM, 0);
	Vector<int> phy_bc_hi(AMREX_SPACEDIM, 0);

	Vector<amrex::Real> inflow_waveform(AMREX_SPACEDIM, 0.0);

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

		// The domain is broken into boxes of size max_grid_size
		pp.get("max_grid_size", max_grid_size);

		pp.get("IterNum", IterNum);

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

		pp.queryarr("inflow_waveform", inflow_waveform);

		// Parsing the target resolution from input file
		target_resolution = -1;
		pp.query("target_resolution", target_resolution);

		momentum_tolerance = 1.e-10;
		pp.query("momentum_tolerance", momentum_tolerance);

		PSEUDO_TIMESTEPPING = 1;
		pp.query("PSEUDO_TIMESTEPPING", PSEUDO_TIMESTEPPING);

		// Default plot_int to -1, allow us to set it to something else in the inputs file
		// If int < 0 then no plot files will be written
		plot_int = -1;
		pp.query("plot_int", plot_int);

		chk_int = -1;
		pp.query("chk_int", chk_int);

		txt_int = 0;
		pp.query("txt_int", txt_int);

		// Read checkpoint frame to load
		chk_out = 0;
		pp.query("chk_out", chk_out);
	}

	Vector<int> is_periodic(AMREX_SPACEDIM, 0);
	// BCType::int_dir = 0
	for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
		if (phy_bc_lo[idim] == 111 && phy_bc_hi[idim] == 111) {
			is_periodic[idim] = 1;
		}
		amrex::Print() << "INFO| periodicity in " << idim+1 << "th dimension: " << is_periodic[idim] << "\n";
	}

	// Calculating number of step to reach the targeted resolution
	int nsteps_target = target_resolution == -1 ? 0 : n_cell/target_resolution - 1;
	amrex::Print() << "INFO| target resolution: " << target_resolution << "\n";
	amrex::Print() << "INFO| number of steps to reach the target resolution: " << nsteps_target << "\n";

	// Nghost = number of ghost cells for each array
	int Nghost = 2; // 2nd order accuracy scheme is used for convective terms

	// Ncomp = number of components for userCtx
	// The userCtx has 02 components:
	// userCtx(0) = Pressure
	// userCtx(1) = Phi
	int Ncomp = 2;

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-= Defining System's Variables =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
	// FLOW VARIABLES
	// Note: hybrid staggerred/non-staggered grid
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

	MultiFab userCtx; 	 // store the pressure and phi
	MultiFab velCart; 	 // store the Cartesian velocity components living in the cell center;
	MultiFab velCartPrev;

	MultiFab fluxConvect; // store the convective fluxes used in solving for contravariant velocities, 		 hence lives in the cell center
	MultiFab fluxViscous; // store the viscous fluxes used in solving for contravariant velocities, 			 hence lives in the cell center
	MultiFab fluxPrsGrad; // store the pressure gradient fluxes used in solving for contravariant velocities, hence lives in the cell center
	MultiFab fluxTotal;   // store the total fluxes used in solving for contravariant velocities, 				 hence lives in the cell center

	MultiFab poisson_rhs; // store the right-hand-side of the Poisson equation for phi, lives in the cell center
	MultiFab poisson_sol; // store the solution of the Poisson equation, which is phi,  lives in the cell center

	MultiFab cc_grad_phi; // store the gradient of phi used to update the Cartesian velocity, lives at the cell center

	MultiFab cc_kinetic_energy; // store the total kinetic energy of the system

	MultiFab cc_analytical_diff; // store the analytical solution (if present) of the non-staggered grid
	// Comp 0 is velocity field along x-axis
	// Comp 1 is velocity field along y-axis
	// Comp 2 is pressure field

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

	Array<MultiFab, AMREX_SPACEDIM> velCont; // store the contravariant velocity components living in the face center
	Array<MultiFab, AMREX_SPACEDIM> velContPrev;
	Array<MultiFab, AMREX_SPACEDIM> velContDiff;

	Array<MultiFab, AMREX_SPACEDIM> momentum_rhs; // store the right-hand-side of the momentum equation

	Array<MultiFab, AMREX_SPACEDIM> fluxHalfN1; // these are half-node fluxed used in the QUICK scheme
	Array<MultiFab, AMREX_SPACEDIM> fluxHalfN2;
	Array<MultiFab, AMREX_SPACEDIM> fluxHalfN3;

	Array<MultiFab, AMREX_SPACEDIM> velStar; // store the intermediate velocity field in the Fractional Step Method
	Array<MultiFab, AMREX_SPACEDIM> velStarDiff;

	Array<MultiFab, AMREX_SPACEDIM> array_grad_p; 	// store the gradient of pressure
	Array<MultiFab, AMREX_SPACEDIM> array_grad_phi; // store the gradient of phi

	Array<MultiFab, AMREX_SPACEDIM> array_analytical_vel; // store the analytical velocity (if present) of the staggered grid

	// Variables at check-out time 
	/*
	MultiFab pressure;
	MultiFab vel_xCont;
	MultiFab vel_yCont;
	MultiFab vel_xContPrev;
	MultiFab vel_yContPrev;
	*/
	Real time, dt;
	int starting_step;

	Geometry geom;
	// make Geometry
	IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
	IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
	Box domain(dom_lo, dom_hi);
	// This defines the physical box, [0,1] in each direction.
	// CASE: Taylor-Green vortex
	//RealBox real_box({AMREX_D_DECL( Real(0.0), Real(0.0), Real(0.0))},
	//				 {AMREX_D_DECL( Real(2.0)*M_PI, Real(2.0)*M_PI, Real(2.0)*M_PI)}); 

	// CASE: Lid-driven cavity
	RealBox real_box({AMREX_D_DECL( Real(0.0), Real(0.0), Real(0.0) )},
					 {AMREX_D_DECL( Real(1.0), Real(2.0), Real(1.0) )}); 
	// This defines a Geometry object
	// NOTE: the coordinate system is Cartesian
	geom.define(domain, &real_box, CoordSys::cartesian, is_periodic.data());

	GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
	Real coeff = AMREX_D_TERM( 1./(dx[0]*dx[0]),
							 + 1./(dx[1]*dx[1]),
							 + 1./(dx[2]*dx[2]) );
	dt = cfl/(2.0*coeff);

	amrex::Print() << "INFO| number of dimensions: " << AMREX_SPACEDIM << "\n";
	if (fixed_dt != -1.0) {
		dt = fixed_dt;
		amrex::Print() << "INFO| dt overridden with fixed_dt: " << dt << "\n";
	}

	// Setup the target point for extracting the velocity field
	GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
	Real const i_target = n_cell/4 - 1;
	//Real const i_target = 0;
	Real const x_cart_target = (i_target + Real(0.5)) * dx[0] + prob_lo[0];
	Real const x_cont_target = (i_target + Real(0.0)) * dx[0] + prob_lo[0];

	Real const j_target = n_cell/4 - 1;
	//Real const j_target = 0;
	Real const y_target = (j_target + Real(0.5)) * dx[1] + prob_lo[1];

	amrex::Print() << "DEBUG| Extract Cartesian solution at (x ; y) = (" << x_cart_target << " ; " << y_target << ") \n";
	amrex::Print() << "DEBUG| Extract contravariant solution at (x ; y) = (" << x_cont_target << " ; " << y_target << ") \n";

	BoxArray ba, edge_ba;
   // make BoxArray
	DistributionMapping dm;

	if (chk_out > 0) {
		amrex::Print() << "INFO| REQUEST FROM USER TO START FROM CHECKPOINT " << chk_out << "\n";
		LoadCheckpoint(ba, dm, userCtx, velCont, velContPrev, time, chk_out);
	} else {
		// Initialize the boxarray "ba" from the single box "bx"
		ba.define(domain);
		// Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
		ba.maxSize(max_grid_size);
		
		// How Boxes are distrubuted among MPI processes
		// Distribution mapping between the processors
		dm.define(ba, ParallelDescriptor::NProcs());

		userCtx.define(ba, dm, Ncomp, 1);
		
		for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
		{
			edge_ba = ba;
			edge_ba.surroundingNodes(dir);

			velCont[dir].define(edge_ba, dm, 1, 0);
			velContPrev[dir].define(edge_ba, dm, 1, 0);
		}
	}
		
	// Cell-centered variables
	velCart.define(ba, dm, AMREX_SPACEDIM, Nghost);
	velCartPrev.define(ba, dm, AMREX_SPACEDIM, Nghost);

	fluxConvect.define(ba, dm, AMREX_SPACEDIM, 0);
	fluxViscous.define(ba, dm, AMREX_SPACEDIM, 0);
	fluxPrsGrad.define(ba, dm, AMREX_SPACEDIM, 0);
	fluxTotal.define(ba, dm, AMREX_SPACEDIM, 1);

	cc_grad_phi.define(ba, dm, AMREX_SPACEDIM, 1);

	poisson_rhs.define(ba, dm, 1, 1);
	poisson_sol.define(ba, dm, 1, 1);
	cc_analytical_diff.define(ba, dm, 3, 0);
	cc_kinetic_energy.define(ba, dm, 1, 0);
					
	// Face-centered variables
	for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
	{
		BoxArray edge_ba = ba;
		edge_ba.surroundingNodes(dir);

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

	if (chk_out > 0) {
		// Print information in the checkpoint file
		amrex::Print() << "INFO| checkout time: " << time << "\n";
		amrex::Print() << "INFO| checkout box array: " << ba << "\n";
		amrex::Print() << "INFO| checkout geometry: " << geom << "\n";

    	for (int dir=0; dir<AMREX_SPACEDIM; ++dir) {
        	MultiFab::Copy(velContDiff[dir], velCont[dir], 0, 0, 1, 0);
			// Subtract src from dst
			// MultiFab::Subtract(MultiFab& dst, MultiFab& src, int srccomp, int dstcomp, int numcomp, IntVect& nghost) 
        	MultiFab::Subtract(velContDiff[dir], velContPrev[dir], 0, 0, 1, 0);
    	}

		// convert contravarient to cartesian velocity
		cont2cart(velCart, velCont, geom, Nghost, phy_bc_lo, phy_bc_hi, inflow_waveform, time, n_cell);
		amrex::Print() << "\n";
		cont2cart(velCartPrev, velContPrev, geom, Nghost, phy_bc_lo, phy_bc_hi, inflow_waveform, time, n_cell);
		amrex::Print() << "\n";

		// Extract middle line from Checkpoint files

		amrex::Print() << "DEBUG| Ploting flow fields loaded from checkpoint file \n";
		Export_Flow_Field("pltCheckout", userCtx, velCart, ba, dm, geom, time, chk_out);
		Export_Flow_Field("pltCheckoutPrev", userCtx, velCartPrev, ba, dm, geom, time, chk_out);

		starting_step = chk_out + 1;
	} else {
		// time = starting time in the simulation
		time = 0.0;
		amrex::Print() << "INFO| start time: " << time << "\n";
		amrex::Print() << "INFO| configured box array: " << ba << "\n";
		amrex::Print() << "INFO| configured geometry: " << geom << "\n";

		amrex::Print() << "========================= INITIALIZATION STEP ========================= \n";
		hybrid_grid_init(userCtx, velCont, velContPrev, velCart, velCartPrev, geom, Nghost, phy_bc_lo, phy_bc_hi, inflow_waveform, time, dt, n_cell);
		// Write a plotfile of the initial data if plot_int > 0
		// (plot_int was defined in the inputs file)
		if (plot_int > 0)
		{
			Export_Flow_Field("pltInit", userCtx, velCart, ba, dm, geom, time, 0);
			Export_Flow_Field("pltInitPrev", userCtx, velCartPrev, ba, dm, geom, time, 0);
		}

		starting_step = 1;
	}

	//amrex::Abort("INFO | STOP HERE FOR DEBUGGING RESTART ROUTINE");

	//---------------------------------------------------------
	// Boundary conditions for the Poisson equation
	// --------------------------------------------------------
	Vector<BCRec> bc(poisson_sol.nComp());
	for ( int n = 0; n < poisson_sol.nComp(); ++n )
	{
		for( int idim = 0; idim < AMREX_SPACEDIM; ++idim )
		{
			if ( phy_bc_lo[idim] == 111 ) {
				bc[n].setLo(idim, BCType::int_dir);
			} else if ( std::abs(phy_bc_lo[idim]) == 135 || 
						std::abs(phy_bc_lo[idim]) == 165 || 
						std::abs(phy_bc_lo[idim]) == 195 ) {
				bc[n].setLo(idim, BCType::foextrap);
			} else {
				amrex::Abort("Invalid bc_lo");
			}

			if ( phy_bc_hi[idim] == 111 ) {
				bc[n].setHi(idim, BCType::int_dir);
			} else if ( std::abs(phy_bc_lo[idim]) == 135 || 
						std::abs(phy_bc_hi[idim]) == 165 || 
						std::abs(phy_bc_hi[idim]) == 195 ) {
				bc[n].setHi(idim, BCType::foextrap);
			} else {
				amrex::Abort("Invalid bc_hi");
			}
		}
	}

	// Print desired variables for debugging
	//amrex::Print() << "PARAMS| cfl value calculated from geometry: " << cfl << "\n";
	//amrex::Print() << "PARAMS| dt value from above cfl: " << dt << "\n";
	amrex::Print() << "PARAMS| reynolds number: " << ren << "\n";
	amrex::Print() << "PARAMS| number of ghost cells for each array: " << Nghost << "\n";

	// Quickly init other fields as zero
	fluxConvect.setVal(0.0);
	fluxViscous.setVal(0.0);
	fluxPrsGrad.setVal(0.0);
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

	// Pseudo-time step for the RK4 momentum solver
	amrex::Real d_tau = Real(0.4)*dt;
	// Setup RK4 scheme coefficients
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
	for (int n = starting_step; n <= nsteps; ++n)
	{
		// Update velContDiff
		for (int comp=0; comp < AMREX_SPACEDIM; ++comp) {
			MultiFab::Copy(velContDiff[comp], velCont[comp], 0, 0, 1, 0);
			MultiFab::Subtract(velContDiff[comp], velContPrev[comp], 0, 0, 1, 0);
			MultiFab::Copy(velContPrev[comp], velCont[comp], 0, 0, 1, 0);
			MultiFab::Copy(velStar[comp], velCont[comp], 0, 0, 1, 0);
		}

		// Update the time
		time = time + dt;

		// Momentum solver
		// MOMENTUM |1| Setup counter
		int countIter = 0;
		Real normError = 1.e9;
		
		Box dom = geom.Domain();

		amrex::Print() << "============================ ADVANCE STEP " << n << " ============================ \n";
		//-----------------------------------------------
		// This is the sub-iteration of the semi-implicit scheme
		//-----------------------------------------------
		while ( normError > momentum_tolerance )
		{
			//amrex::Print() << "SOLVING| Momentum | performing Runge-Kutta at pseudo step: " << countIter
			//					 << " => latest error norm = " << normError << "\n";
			if ( PSEUDO_TIMESTEPPING == 0 ) {
				// EXPLICIT TIME MARCHING
				// ------------------------- PRESSURE GRADIENT CALCULATION -------------------------
				//gradient_calc_approach1(fluxTotal, fluxPrsGrad, cc_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
				gradient_calc_approach2(array_grad_p, array_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
				// ------------------------- FLUX CALCULATION -------------------------
				fluxTotal.setVal(0.0);
				convective_flux_calc_new_quick(fluxTotal, fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCart, velStar, phy_bc_lo, phy_bc_hi, geom);
				viscous_flux_calc(fluxTotal, fluxViscous, velCart, ren, geom);
				fluxTotal.FillBoundary(geom.periodicity());
				// --------------------------- MOMENTUM SOLVER ---------------------------
				momentum_righthand_side_calc(fluxTotal, array_grad_p, momentum_rhs, phy_bc_lo, phy_bc_hi, geom);
				amrex::Print() << "SOLVING| Momentum | performing Explicit Time Marching ";
				explicit_time_marching(momentum_rhs, velStar, velContDiff, geom, phy_bc_lo, phy_bc_hi, dt);
				
				normError = Error_Computation(velCont, velStar, velStarDiff, geom);
				amrex::Print() << "=> latest error norm = " << normError << "\n";

				for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
				{
					MultiFab::Copy(velCont[comp], velStar[comp], 0, 0, 1, 0);
				}

				break;
			} else {
				// 4 sub-iterations of one RK4 iteration
				amrex::Print() << "SOLVING| Momentum | performing RK4 Pseudo-Time Marching ";
				for (int sub = 0; sub < RungeKuttaOrder; ++sub )
				{
					// ------------------------- PRESSURE GRADIENT CALCULATION -------------------------
					//gradient_calc_approach1(fluxTotal, fluxPrsGrad, cc_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
					gradient_calc_approach2(array_grad_p, array_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
					// ------------------------- FLUX CALCULATION -------------------------
					fluxTotal.setVal(0.0);
					convective_flux_calc_new_quick(fluxTotal, fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCart, velStar, phy_bc_lo, phy_bc_hi, geom);
					viscous_flux_calc(fluxTotal, fluxViscous, velCart, ren, geom);

					fluxTotal.FillBoundary(geom.periodicity());
					// Fluxes' normal component on the wall boundaries are set to zero
					// Enforced by 1 layer of ghost cells
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
					for (MFIter mfi(fluxTotal); mfi.isValid(); ++mfi)
					{
						const Box& vbx = mfi.growntilebox(1);
						auto const& flux_total = fluxTotal.array(mfi);

						int lo = dom.smallEnd(0); //amrex::Print() << lo << "\n";
						int hi = dom.bigEnd(0);   //amrex::Print() << hi << "\n";

						if (vbx.smallEnd(0) < lo) {
							amrex::ParallelFor(vbx,
											   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
								if ( i < lo ) {
									flux_total(i, j, k, 0) = -flux_total(-i -1, j, k, 0);
									/*
									amrex::Print() << "DEBUG| Total Flux Ghost Cell at (i, j) = (" 
											   	   << i << ", " 
											   	   << j << ") = "
											   	   << flux_total(i, j, k, 0) << " ; "
											   	   << flux_total(i, j, k, 1) << "\n";
									*/
								}
							});
						}

						if (vbx.bigEnd(0) > hi) {
							amrex::ParallelFor(vbx,
											   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
								if ( i > hi ) {
									flux_total(i, j, k, 0) = -flux_total(((n_cell - i) + (n_cell - 1)), j, k, 0);
								}
							});
						}

						lo = dom.smallEnd(1);
						hi = dom.bigEnd(1);

						if (vbx.smallEnd(1) < lo) {
							amrex::ParallelFor(vbx,
											   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
								if ( j < lo ) {
									flux_total(i, j, k, 1) = -flux_total(i, -j -1, k, 1);
								}
							});
						}

						if (vbx.bigEnd(1) > hi) {
							amrex::ParallelFor(vbx,
											   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
								if ( j > hi ) {
									flux_total(i, j, k, 1) = -flux_total(i, ((n_cell - j) + (n_cell - 1)), k, 1);
								}
							});
						}

#if (AMREX_SPACEDIM > 2)
						lo = dom.smallEnd(2);
						hi = dom.bigEnd(2);

						if (vbx.smallEnd(2) < lo) {
							amrex::ParallelFor(vbx,
											   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
								if ( k < lo ) {
									flux_total(i, j, k, 2) = -flux_total(i, j, -k -1, 2);
								}
							});
						}

						if (vbx.bigEnd(2) > hi) {
							amrex::ParallelFor(vbx,
											   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
								if ( k > hi ) {
									flux_total(i, j, k, 2) = -flux_total(i, j, ((n_cell - k) + (n_cell - 1)), 2);
								}
							});
						}
#endif
					}
					momentum_righthand_side_calc(fluxTotal, array_grad_p, momentum_rhs, phy_bc_lo, phy_bc_hi, geom);
					// --------------------------- MOMENTUM SOLVER ---------------------------
					runge_kutta4_pseudo_time_stepping(rk, sub, momentum_rhs, velStar, velCont, velContDiff, velContPrev, velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, inflow_waveform, n_cell, time, dt);
				} // RUNGE-KUTTA | END
				normError = Error_Computation(velCont, velStar, velStarDiff, geom);
				amrex::Print() << "=> step = " << countIter << "; error norm = " << normError << "\n";
			}
			// Re-assign guess for the next iteration
			for ( int comp=0; comp < AMREX_SPACEDIM; ++comp)
			{
				MultiFab::Copy(velCont[comp], velStar[comp], 0, 0, 1, 0);
			}
			countIter++;
			// Handler for blowing-up situation
			//if (countIter == 2) {
			if (countIter > IterNum) {
				amrex::Print() << "WARNING| Exceeded number of momenum iterations; exiting loop\n";
				//amrex::Print() << "Forced break at pseudo step " << countIter << "\n";
				break;
			}
			if ( normError > 1.e2 )
			{
				amrex::Print() << "WARNING| Error Norm diverges, exiting loop\n";
				break;
			}
			//break; // Tactical breakpoint
		}// End of the Momentum loop iteration!
		//---------------------------------------
		// MOMENTUM |4| PLOTTING
		// This is just for debugging only !
		if (plot_int > 0 && n%plot_int == 0)
		{
			Export_Fluxes(fluxConvect, fluxViscous, fluxPrsGrad, ba, dm, geom, time, n);

			/*
			amrex::Print() << "DEBUG| Intermediate contravariant velocity \n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    		for ( MFIter mfi(velStar[0]); mfi.isValid(); ++mfi )
    		{
        		const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        		const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        		const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        		auto const& xcont_im = velStar[0].array(mfi);
        		auto const& ycont_im = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
				auto const& zcont_im = velStar[2].array(mfi);
#endif

        		amrex::ParallelFor(xbx,
                           		   [=] AMREX_GPU_DEVICE (int i, int j, int k){
            		amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
            		amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
            		amrex::Print() << x << ";" << y << ";" << xcont_im(i, j, k) << "\n";
        		});
    		}
			*/
		}
		//---------------------------------------
		amrex::Print() << "\nSOLVING| finished solving Momentum equation. \n";
		amrex::Print() << "\n";
		//break; // Tactical breakpoint

		// Poisson solver
		//    Laplacian(\phi) = (Real(1.5)/dt)*Div(u_i^*)
		// POISSON |1| Calculating the RSH
		//poisson_righthand_side_calc(poisson_rhs, velCont, geom, dt);
		poisson_righthand_side_calc(poisson_rhs, velStar, geom, dt);
		// POISSON |2| Init Phi at the begining of the Poisson solver
		poisson_advance(poisson_sol, poisson_rhs, geom, ba, dm, bc);
		//GMRESPOISSON gmres_poisson(ba, dm, geom);

		//poisson_sol.setVal(0.0);

		//gmres_poisson.usePrecond(1); //<------ Contribution
		//gmres_poisson.setVerbose(2);
		//gmres_poisson.solve(poisson_sol, poisson_rhs, 1.0e-10, 0.0);

		/*
		amrex::Print() << "DEBUGGING| Phi: \n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
		for ( MFIter mfi(poisson_sol); mfi.isValid(); ++mfi )
		{
			const Box& vbx = mfi.validbox();
			auto const& phi = poisson_sol.array(mfi);
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k){
				amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
				amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
				amrex::Print() << x << ";" << y << ";" << phi(i, j, k) << "\n";
			});
		}
		*/

		amrex::Print() << "\nSOLVING| finished solving Poisson equation. \n";
		amrex::Print() << "\n";
		if (plot_int > 0 && n%plot_int == 0)
		{
			const std::string &rhs_export = amrex::Concatenate("pltPoissonRHS", n, 5);
			WriteSingleLevelPlotfile(rhs_export, poisson_rhs, {"poissonRHS"}, geom, time, n);
			const std::string &poisson_export = amrex::Concatenate("pltPhi", n, 5);
			WriteSingleLevelPlotfile(poisson_export, poisson_sol, {"phi"}, geom, time, n);
		}
		MultiFab::Copy(userCtx, poisson_sol, 0, 1, 1, 0);
		userCtx.FillBoundary(geom.periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for (MFIter mfi(userCtx); mfi.isValid(); ++mfi)
	{
		const Box& vbx = mfi.growntilebox(1);
		auto const& ctx = userCtx.array(mfi);

		int lo = dom.smallEnd(0); //amrex::Print() << lo << "\n";
		int hi = dom.bigEnd(0);   //amrex::Print() << hi << "\n";

		if (vbx.smallEnd(0) < lo) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( i < lo ) {
					ctx(i, j, k, 0) = ctx(-i - 1, j, k, 0);
					ctx(i, j, k, 1) = ctx(-i - 1, j, k, 1);
				}
			});
		}

		if (vbx.bigEnd(0) > hi) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( i > hi ) {
					ctx(i, j, k, 0) = ctx(((n_cell - i) + (n_cell - 1)), j, k, 0);
					ctx(i, j, k, 1) = ctx(((n_cell - i) + (n_cell - 1)), j, k, 1);
				}
			});
		}

		lo = dom.smallEnd(1);
		hi = dom.bigEnd(1);

		if (vbx.smallEnd(1) < lo) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( j < lo ) {
					ctx(i, j, k, 0) = ctx(i, -j - 1, k, 0);
					ctx(i, j, k, 1) = ctx(i, -j - 1, k, 1);
				}
			});
		}

		if (vbx.bigEnd(1) > hi) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( j > hi ) {
					ctx(i, j, k, 0) = ctx(i, ((n_cell - j) + (n_cell - 1)), k, 0);
					ctx(i, j, k, 1) = ctx(i, ((n_cell - j) + (n_cell - 1)), k, 1);
				}
			});
		}

#if (AMREX_SPACEDIM > 2)
		lo = dom.smallEnd(2);
		hi = dom.bigEnd(2);

		if (vbx.smallEnd(2) < lo) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( k < lo ) {
					ctx(i, j, k, 0) = ctx(i, j, -k - 1, 0);
					ctx(i, j, k, 1) = ctx(i, j, -k - 1, 1);
				}
			});
		}

		if (vbx.bigEnd(2) > hi) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( k > hi ) {
					ctx(i, j, k, 0) = ctx(i, j, ((n_cell - k) + (n_cell - 1)), 0);
					ctx(i, j, k, 1) = ctx(i, j, ((n_cell - k) + (n_cell - 1)), 1);
				}
			});
		}
#endif
	}

		// Update the solution
		// u_i^{n+1} = u_i^*- 2dt/3 * grad(\phi^{n+1})
		// p^{n+1} = p^n  + \phi^{n+1} - Re^-1 * div(u_i^*)
		// also update velContDiff = velCont-velContPrev
		update_solution(array_grad_p, array_grad_phi, fluxPrsGrad, cc_grad_phi, poisson_rhs, userCtx, velCart, velCont, velStar, geom, time, dt, Nghost, phy_bc_lo, phy_bc_hi, inflow_waveform, n_cell, ren);
	
		// Writing checkpoint files
		if (chk_int > 0 && n%chk_int == 0)
		{
			SaveCheckpoint(ba, dm, userCtx, velCont, velContPrev, time, n);
		}
		amrex::Print() << "SOLVING| finished updating all fields \n";

		// Assert the divergence of the updated velocity field
		// Divergence should be zero
		poisson_righthand_side_calc(poisson_rhs, velCont, geom, dt);

		// Compare the solution with the analytical solution
		amrex::Print() << "CALCULATING ANALYTICAL \n";
		cc_analytical_calc(cc_analytical_diff, geom, time);
		cc_spectral_analysis(cc_kinetic_energy, cc_analytical_diff, geom);
		if (plot_int > 0 && n%plot_int == 0)
		{
			const std::string &analytical_export = amrex::Concatenate("pltAnalytic", n, 5);
			WriteSingleLevelPlotfile(analytical_export, cc_analytical_diff, {"p-exac","u-exac", "v-exac"}, geom, time, n);
    
			Export_Flow_Field("pltResults", userCtx, velCart, ba, dm, geom, time, n);

			/*
			amrex::Print() << "DEBUG| Calculated the updated contravariant velocity \n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    		for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    		{
        		const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        		const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        		const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        		auto const& xcont_new = velCont[0].array(mfi);
        		auto const& ycont_new = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        		auto const& zcont_new = velCont[2].array(mfi);
#endif

        		amrex::ParallelFor(xbx,
                           		   [=] AMREX_GPU_DEVICE (int i, int j, int k){
            		amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
            		amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
            		amrex::Print() << x << ";" << y << ";" << xcont_new(i, j, k) << "\n";
        		});
    		}

			*/
		}

		// Extract horizontal velocity at the line (0.5, y, 0.5)
		if ( plot_int > 0 && n%plot_int == 0 )
		{
			std::string vertical_line_filename = "ren" + std::to_string(static_cast<int>(ren)) + "_vertical_numel_positive" + std::to_string(n);

			for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
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
#if (AMREX_SPACEDIM > 2)
					amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
#endif
					if ( i == n_cell/2 && k == (n_cell/2) ) {
						write_exact_line_solution(time, x, y, z, vel_cont_x(i, j, k), vertical_line_filename);
					}
				});
			}

			vertical_line_filename = "ren" + std::to_string(static_cast<int>(ren)) + "_vertical_numel_negative" + std::to_string(n);

			for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
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
#if (AMREX_SPACEDIM > 2)
					amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
#endif
					if ( i == (n_cell/2) && k == (n_cell/2 - 1) ) {
						write_exact_line_solution(time, x, y, z, vel_cont_x(i, j, k), vertical_line_filename);
					}
				});
			}
		}
		/*
		if (plot_int > 0 && n%plot_int == 0)
		{
			// Write the exact solution at the line x = 0.5 and y = 0.5
			std::string vertical_line_filename = "ren" + std::to_string(static_cast<int>(ren)) + "_vertical_numel" + std::to_string(n);
			std::string horizontal_line_filename = "ren" + std::to_string(static_cast<int>(ren)) + "_horizontal_numel" + std::to_string(n);

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
					if ( x == M_PI ) {
						auto vel_cont_exact_x = std::sin(x) * std::cos(y) * std::exp(-Real(2.0) * time);
						write_exact_line_solution(time, x, y, vel_cont_x(i, j, k), vel_cont_exact_x, vertical_line_filename);
					}
				});
				amrex::ParallelFor(ybx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k){
					amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
					amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];
					if ( y == M_PI ) {
						auto vel_cont_exact_y = -std::cos(x) * std::sin(y) * std::exp(-Real(2.0) * time);
						write_exact_line_solution(time, x, y, vel_cont_y(i, j, k), vel_cont_exact_y, horizontal_line_filename);
					}
				});
#if (AMREX_SPACEDIM > 2)
#endif
			}
		}
		*/
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
