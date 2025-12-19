#include "poisson.H"
#include "kn_poisson.H"

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>

using namespace amrex;
//--------------------------------------
//++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++

void init_phi(MultiFab& userCtx)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx  = userCtx.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            ctx(i, j, k, 1) = Real(0.0);
        });
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//----- Poisson solver -- main one ---------------------------
//------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void poisson_advance( MultiFab& poisson_sol,
                      MultiFab& poisson_rhs,
                      Geometry const& geom,
                      BoxArray const& grids,
                      DistributionMapping const& dmap,
                      Vector<BCRec> const& bc )
{
	BL_PROFILE_VAR("poisson_advance()", poisson_advance);
/*
 * We use an MLABecLaplacian operator:
 * (ascalar*acoef - bscalar div bcoef grad) phi = RHS
 * for an implicit discretization of the heat equation
 * (I - div dt grad) phi^{n+1} = phi^n
*/
    bool semicoarsening = false;
    int semicoarsening_direction = 1;
    int semicoarsening_level = 1;
    // int max_coarsening_level = 0;

    bool use_hypre = false;

    // initialize valid and ghost region to zero
    poisson_sol.setVal(0.0);

    // assorment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;
    info.setSemicoarsening(semicoarsening);
    info.setSemicoarseningDirection(semicoarsening_direction);
    info.setMaxSemicoarseningLevel(semicoarsening_level);
    //info.setMaxCoarseningLevel(max_coarsening_level);

    // Implicit solve using MLABecLaplacian class
    MLABecLaplacian mlabec({geom}, {grids}, {dmap}, info);

    // order of stencil
    int linop_maxorder = 3;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_lo;
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_hi;

    for (int n = 0; n < poisson_sol.nComp(); ++n)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            // lo-side BCs
            if (bc[n].lo(idim) == BCType::int_dir) {
                LinOp_bc_lo[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].lo(idim) == BCType::foextrap) {
                LinOp_bc_lo[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].lo(idim) == BCType::ext_dir) {
                LinOp_bc_lo[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            // hi-side BCs
            if (bc[n].hi(idim) == BCType::int_dir) {
                LinOp_bc_hi[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].hi(idim) == BCType::foextrap) {
                LinOp_bc_hi[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].hi(idim) == BCType::ext_dir) {
                LinOp_bc_hi[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_hi");
            }
        }
    }

    // tell the solver what the domain boundary conditions are
    mlabec.setDomainBC(LinOp_bc_lo, LinOp_bc_hi);

    // set the boundary conditions
    // This loads the value of the Neumman boundary condition at the ghost cells
    // Ghost cell stores the value of the BCs for the Neumann
    mlabec.setLevelBC(0, &poisson_sol);

    // scaling factors
    Real ascalar = 0.0;
    Real bscalar = -1.0;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices
    MultiFab acoef(grids, dmap, 1, 0);

    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(0.0);
    mlabec.setACoeffs(0, acoef);
    // We need to check this ? What is the coefficent for b for ??
    // bcoef.setVal(1.0);
    // mlabec.setBCoeffs(0, bcoef);


    // bcoef lives on faces so we make an array of face-centered MultiFabs
    // then we will in face_bcoef MultiFabs and load them into the solver.
    std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const BoxArray& ba = amrex::convert(acoef.boxArray(),
                                            IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, acoef.DistributionMap(), 1, 0);
        face_bcoef[idim].setVal(1.0);
    }
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));

    // build an MLMG solver
    MLMG mlmg(mlabec);

    // set solver parameters
    int max_iter = 2000;
    mlmg.setMaxIter(max_iter);

    int max_fmg_iter = 0;
    mlmg.setMaxFmgIter(max_fmg_iter);

    int verbose = 1;
    mlmg.setVerbose(verbose);

    int bottom_verbose = 2;
    mlmg.setBottomVerbose(bottom_verbose);

#ifdef AMREX_USE_HYPRE
    if (use_hypre) {
        amrex::Print() << "INFO| Using Hypre as bottom solver\n";
        mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
        mlmg.setHypreInterface(Hypre::Interface::structed);
        //mlmg.setHypreInterface(hypre_interface);
    }
#endif

    // relative and absolute tolerances for linear solve
    const Real tol_rel = 1.0e-10;
    const Real tol_abs = 0.0;

    // Solve linear system
    mlmg.solve({&poisson_sol}, {&poisson_rhs}, tol_rel, tol_abs);
}

//++++++++++++++++++++++++++++++++++++++++++++
//-- Update the pressure and velocity field---
//-- After the projection step ---------------
//++++++++++++++++++++++++++++++++++++++++++++
void update_solution( Array<MultiFab, AMREX_SPACEDIM>& array_grad_p,
                      Array<MultiFab, AMREX_SPACEDIM>& array_grad_phi,
                      MultiFab& fluxPrsGrad,
                      MultiFab& cc_grad_phi,
                      MultiFab& poisson_rhs,
                      MultiFab& userCtx,
                      MultiFab& velCart,
                      Array<MultiFab, AMREX_SPACEDIM>& velCont,
                      Array<MultiFab, AMREX_SPACEDIM>& velStar,
                      Geometry const& geom,
					  Real& time,
                      Real const& dt,
                      const int& Nghost,
                      Vector<int> const& phy_bc_lo,
                      Vector<int> const& phy_bc_hi,
                      Vector<amrex::Real> const& inflow_waveform,
                      const int& n_cell,
                      Real const& ren )
{
    Box dom(geom.Domain());
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    //gradient_calc_approach1(fluxTotal, fluxPrsGrad, cc_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    gradient_calc_approach2(array_grad_p, array_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

    //==============================================
    //----- Update the contravariant velocity ------
    //----------------------------------------------
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

        auto const& xcont = velCont[0].array(mfi);
        auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zcont = velCont[2].array(mfi);
#endif

        auto const& xstar = velStar[0].array(mfi);
        auto const& ystar = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zstar = velStar[2].array(mfi);
#endif

        auto const& grad_phi_x = array_grad_phi[0].array(mfi);
        auto const& grad_phi_y = array_grad_phi[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& grad_phi_z = array_grad_phi[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            xcont(i, j, k) = xstar(i, j, k) - ( grad_phi_x(i, j, k) * dt / Real(1.5) );
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            ycont(i, j, k) = ystar(i, j, k) - ( grad_phi_y(i, j, k) * dt / Real(1.5) );
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            zcont(i, j, k) = zstar(i, j, k) - ( grad_phi_z(i, j, k) * dt / Real(1.5) );
        });
#endif
    } // End of the loop for boxes
    // Update velCart from the velCont solutions
    cont2cart(velCart, velCont, geom, Nghost, phy_bc_lo, phy_bc_hi, inflow_waveform, time, n_cell);
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
    //==============================================
    //----- Update the pressure field --------------
    //----------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        auto const& grad_u = poisson_rhs.array(mfi);

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            //Update the pressure field
            ctx(i, j, k, 0) = ctx(i, j, k, 0) + ctx(i, j, k, 1); // - ( dt * grad_u(i, j, k, 0) )/Real(1.5)/ren;
        });
    } // End of the loop for boxes
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
}
