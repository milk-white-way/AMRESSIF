#include "myfunc.H"
#include "poisson.H"
#include "kn_poisson.H"

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>


//--------------------------------------
//++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++

void Set_Phi_To_Zero(amrex::MultiFab& phi)
{
  
 
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(phi); mfi.isValid(); ++mfi )
    {
      // This includes one ghost cell
        const Box& vbx = mfi.growntilebox(1);
        auto const& vphi  = phi.array(mfi);


        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

	  vphi(i,j,k) = 0;

        });
    }

    amrex::Print() << "Setting up Phi to Zero completes.....\n";

}


//+++++++++++++++++++++++++++++++++++++++++++++
//------- Setup the RHS -----------------------
//-- For the Poisson equation -----------------
//+++++++++++++++++++++++++++++++++++++++++++++
void Poisson_RHS(amrex::Geometry const& geom,                       
		 amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
		 amrex::MultiFab& rhs, amrex::Real &dt)
{
  amrex::Print() << "Setting up the right hand side of the Poisson equation\n";

  
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

    // Loop for all boxes
    for ( MFIter mfi(rhs); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vrhs  = rhs.array(mfi);

	auto const& xcont = velCont[0].array(mfi);
        auto const& ycont = velCont[1].array(mfi);

	//Loop for all i,j,k in the local domain
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

	  
#if (AMREX_SPACEDIM > 2)
          auto const& zcont = velCont[2].array(mfi);
          compute_flux_divergence_3D(i, j, k, vrhs, xcont, ycont, zcont, dx);
#else
  	  compute_flux_divergence_2D(i, j, k, vrhs, xcont, ycont, dx);

#endif

        });
    }// End of all box loops


    // Scaling the right-hand side to include time-step here
   for ( MFIter mfi(rhs); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vrhs  = rhs.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

	  vrhs(i,j,k) = vrhs(i,j,k) * 1.5 / dt;

        });
    }

    amrex::Print() << "Setting up completes.....\n";

}



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//----- Poisson solver -- main one ---------------------------
//------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void Poisson_Solver (
	      amrex::MultiFab& phi_solution,
	      amrex::MultiFab& rhs_ptr,
              const Geometry& geom,
              const BoxArray& grids,
              const DistributionMapping& dmap,
              const Vector<BCRec>& bc)
{
    /*
      We use an MLABecLaplacian operator:

      (ascalar*acoef - bscalar div bcoef grad) phi = RHS

      for an implicit discretization of the heat equation

      (I - div dt grad) phi^{n+1} = phi^n
     */


    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
     phi_solution.FillBoundary(geom.periodicity());

    // Fill non-periodic physical boundaries
     FillDomainBoundary(phi_solution, geom, bc);

    // assorment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;

    // Implicit solve using MLABecLaplacian class
    MLABecLaplacian mlabec({geom}, {grids}, {dmap}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_lo;
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_hi;

     for (int n = 0; n < phi_solution.nComp(); ++n)
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
    mlabec.setLevelBC(0, &phi_solution);

    // scaling factors
    Real ascalar = 0.0;
    Real bscalar = -1.0;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices
    MultiFab acoef(grids, dmap, 1, 0);

    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(1.0);
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
    int max_iter = 100;
    mlmg.setMaxIter(max_iter);

    int max_fmg_iter = 0;
    mlmg.setMaxFmgIter(max_fmg_iter);

    int verbose = 2;
    mlmg.setVerbose(verbose);

    int bottom_verbose = 0;
    mlmg.setBottomVerbose(bottom_verbose);

    // relative and absolute tolerances for linear solve
    const Real tol_rel = 1.0e-10;
    const Real tol_abs = 0.0;

    // Solve linear system
     mlmg.solve({&phi_solution}, {&rhs_ptr}, tol_rel, tol_abs);
   
}

//++++++++++++++++++++++++++++++++++++++++++++
//-- Update the pressure and velocity field---
//-- After the projection step ---------------
//++++++++++++++++++++++++++++++++++++++++++++
void Poisson_Update_Solution (
	      amrex::MultiFab& phi_solution,
	      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& grad_phi,
	      MultiFab& userCtx,
	      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
	      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velImRK,
	      const Geometry& geom,
              const BoxArray& grids,
              const DistributionMapping& dmap,
              const Vector<BCRec>& bc,
	      const amrex::Real &dt)
{
  amrex::Print() << " Updating the solution \n";

  GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

  // Set the grad_phi components to be zeros
  // Scaling the right-hand side to include time-step here
  for ( MFIter mfi(grad_phi[0]); mfi.isValid(); ++mfi )
    {
        
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
	 // is it OK to handle this tile with the 
	auto const& vphi = phi_solution.array(mfi);
	auto const& vpressure = userCtx.array(mfi);
// 	// grad X, Y, and Z
        auto const& grad_x  = grad_phi[0].array(mfi);
   	auto const& grad_y  = grad_phi[1].array(mfi);
#if (AMREX_SPACEDIM > 2)

 	auto const& grad_z  = grad_phi[2].array(mfi);
#endif


	//+++++++++++++++++++++++++++++++++++
	//--- Zero out the gradient values --
	//+++++++++++++++++++++++++++++++++++
	
	//i direction
        amrex::ParallelFor(xbx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {

	   grad_x(i,j,k) = 0;
	  
         });

	//j direction
        amrex::ParallelFor(ybx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {

	   grad_y(i,j,k) = 0;
	  
         });


#if (AMREX_SPACEDIM > 2)

	//k direction
        amrex::ParallelFor(zbx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   grad_z(i,j,k) = 0;

         });

#endif


	//============================================
	//-- Compute the gradient at the face-center--
	//-- using 2nd order approximation -----------
	// -----  i direction -----------------------
        amrex::ParallelFor(xbx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   //Gradient of phi at half-node (face-center) i + 1/2
	   grad_x(i,j,k) = (vphi(i+1,j,k) - vphi(i,j,k))/dx[0];
	  
         });

	amrex::ParallelFor(ybx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   //Gradient of phi at half-node (face-center) j + 1/2
	   grad_y(i,j,k) = (vphi(i,j+1,k) - vphi(i,j,k))/dx[1];
	  
         });

#if (AMREX_SPACEDIM > 2)

	amrex::ParallelFor(zbx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   //Gradient of phi at half-node (face-center) j + 1/2
	   grad_y(i,j,k) = (vphi(i,j,k+1) - vphi(i,j,k))/dx[2];
	  
         });

#endif

	//===============================================
	//----- Update the pressure field ---------------
	//+++++++++++++++++++++++++++++++++++++++++++++++
	const Box& vbx = mfi.validbox();
	amrex::ParallelFor(vbx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   //Update the pressure field
	   vpressure(i,j,k, 0) = vpressure(i,j,k, 0) + vphi(i,j,k);
	  
         });

	// Take the value of the contravariant velocities out!
	   auto const& xcont = velCont[0].array(mfi);
	   auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
	   auto const& zcont = velCont[2].array(mfi);
#endif

	   auto const& xcontRK = velImRK[0].array(mfi);
	   auto const& ycontRK = velImRK[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
	   auto const& zcontRK = velImRK[2].array(mfi);
#endif

	//===============================================
	//----- Update the velocity field ---------------
	//----------------------------------------------
	// Ucont in the i - direction
	// -----  i direction -----------------------
        amrex::ParallelFor(xbx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   //Gradient of phi at half-node (face-center) i + 1/2
	   xcont(i,j,k) = xcontRK(i,j,k) + grad_x(i,j,k) * 1.5 /dt;
	  
         });

	//----------------------------------------------
	// Ucont in the j - direction
	// -----  j direction -----------------------
        amrex::ParallelFor(ybx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   //Gradient of phi at half-node (face-center) i + 1/2
	   ycont(i,j,k) = ycontRK(i,j,k) + grad_y(i,j,k) * 1.5 /dt;
	  
         });
#if (AMREX_SPACEDIM > 2)
	//----------------------------------------------
	// Ucont in the k - direction
	// -----  k direction -----------------------
        amrex::ParallelFor(zbx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k)
         {
	   //Gradient of phi at half-node (face-center) i + 1/2
	   zcont(i,j,k) = zcontRK(i,j,k) + grad_z(i,j,k) * 1.5 /dt;
	  
         });
#endif
  
  
    }// End of the loop for boxes



  
    amrex::Print() << " Updating ends. \n";
}
