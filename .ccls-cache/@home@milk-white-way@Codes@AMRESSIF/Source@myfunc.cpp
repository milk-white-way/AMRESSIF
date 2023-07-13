#include <AMReX_MultiFabUtil.H>

#include "myfunc.H"
#include "mykernel.H"

using namespace amrex;

void advance (MultiFab& phi_old,
              MultiFab& phi_new,
              Array<MultiFab, AMREX_SPACEDIM>& flux,
              Real dt,
              Geometry const& geom)
{
    BL_PROFILE("advance");

    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries.
    // There are no physical domain boundaries to fill in this example.
    phi_old.FillBoundary(geom.periodicity());

    //
    // Note that this simple example is not optimized.
    // The following two MFIter loops could be merged
    // and we do not have to use flux MultiFab.
    //
    // =======================================================

    // This example supports both 2D and 3D.  Otherwise,
    // we would not need to use AMREX_D_TERM.
    AMREX_D_TERM(const Real dxinv = geom.InvCellSize(0);,
                 const Real dyinv = geom.InvCellSize(1);,
                 const Real dzinv = geom.InvCellSize(2););

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    // Compute fluxes one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.nodaltilebox(0);
        const Box& ybx = mfi.nodaltilebox(1);
        auto const& fluxx = flux[0].array(mfi);
        auto const& fluxy = flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.nodaltilebox(2);
        auto const& fluxz = flux[2].array(mfi);
#endif
        auto const& phi = phi_old.const_array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_x(i,j,k,fluxx,phi,dxinv);
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_y(i,j,k,fluxy,phi,dyinv);
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_z(i,j,k,fluxz,phi,dzinv);
        });
#endif
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    // Advance the solution one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& fluxx = flux[0].const_array(mfi);
        auto const& fluxy = flux[1].const_array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& fluxz = flux[2].const_array(mfi);
#endif
        auto const& phiOld = phi_old.const_array(mfi);
        auto const& phiNew = phi_new.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            update_phi(i,j,k,phiOld,phiNew,
                       AMREX_D_DECL(fluxx,fluxy,fluxz),
                       dt,
                       AMREX_D_DECL(dxinv,dyinv,dzinv));
        });
    }
}

void init(MultiFab& userCtx,
          MultiFab& velCart,
          MultiFab& velCartDiff,
          Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
          Geometry const& geom)
{

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& cbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        auto const& ucart = velCart.array(mfi);
        auto const& udiff = velCartDiff.array(mfi);
        amrex::ParallelFor(cbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_init(i, j, k, ctx, ucart, udiff, dx, prob_lo);
        });
    }

    for ( MFIter mfi(velContDiff[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& xdiff = velContDiff[0].array(mfi);
        auto const& ydiff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zdiff = velContDiff[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            xdiff(i, j, k) = Real(0.0);
        });
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            ydiff(i, j, k) = Real(0.0);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            zdiff(i, j, k) = Real(0.0);
        });
#endif
    }
}

void fill_physical_ghost_cells (MultiFab& velCart,
                                int const& Nghost,
                                int const& n_cell,
                                Vector<int> const& bc_lo,
                                Vector<int> const& bc_hi)
{
   for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
   {
       const Box& gbx = mfi.growntilebox(Nghost);
       auto const& ucart = velCart.array(mfi);

       amrex::ParallelFor(gbx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k)
       {
           if ( i<0 ) // west wall
           {
               if ( bc_lo[0] == -1 ) // non-slip wall
               {
                   for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                   {
                       ucart(i, j, k, dir) = - ucart(i+1, j, k, dir);
                   }
               }
               else if ( bc_lo[0] == 1 ) // slip wall
               {
                   // Modified for >= 2 layer of ghost cells
                   ucart(i, j, k, 0) = - ucart(-i-1, j, k, 0);
                   ucart(i, j, k, 1) = ucart(i+1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
                   ucart(i, j, k, 2) = ucart(i+1, j, k, 2);
#endif
               }
           }
           else if ( i>(n_cell-1) ) // east wall
           {
               if ( bc_hi[0] == -1 ) // non-slip wall
               {
                   for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                   {
                       ucart(i, j, k, dir) = - ucart(i-1, j, k, dir);
                   }
               }
               else if ( bc_hi[0] == 1 ) // slip wall
               {
                   ucart(i, j, k, 0) = - ucart(i-1, j, k, 0);
                   ucart(i, j, k, 1) = ucart(i-1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
                   ucart(i, j, k, 2) = ucart(i-1, j, k, 2);
#endif
               }
           }
           else if ( j<0 ) // south wall
           {
               if ( bc_lo[1] == -1 ) // non-slip wall
               {
                   for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                   {
                       ucart(i, j, k, dir) = - ucart(i, j+1, k, dir);
                   }
               }
               else if ( bc_lo[1] == 1 ) // slip wall
               {
                   ucart(i, j, k, 0) = ucart(i, j+1, k, 0);
                   ucart(i, j, k, 1) = - ucart(i, j+1, k, 1);
#if (AMREX_SPACEDIM > 2)
                   ucart(i, j, k, 2) = ucart(i, j+1, k, 2);
#endif
               }
           }
           else if ( j>(n_cell-1) ) // north wall
           {
               if ( bc_hi[1] == -1 ) // non-slip wall
               {
                   for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                   {
                       ucart(i, j, k, dir) = - ucart(i, j-1, k, dir);
                   }
               }
               else if ( bc_hi[1] == 1 ) // slip wall
               {
                   ucart(i, j, k, 0) = ucart(i, j-1, k, 0);
                   ucart(i, j, k, 1) = - ucart(i, j-1, k, 1);
#if (AMREX_SPACEDIM > 2)
                   ucart(i, j, k, 2) = ucart(i, j-1, k, 2);
#endif
               }
           }
       });
   }
}

void cont2cart (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont,
                const Geometry& geom)
{
    // Step one: Each cell center is avaraged from the face center components.
    // AMReX has a function for this task:
    average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);

    // Step two: copy the value of the velocity components of velBcs
    // Aproach 1: Lazy
    //avarege_face_to_cellcenter(ANewMultiFab, amrex::GetArrOfConstPtrs(velBcs), geom);

    // Loop on the on the edge of domain??
    // Not really needed now!
}

void cart2cont (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    {

        Box xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        Box ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
        auto const& ucont_x = velCont[0].array(mfi);
        auto const& ucont_y = velCont[1].array(mfi);

        auto const& ucart = velCart.array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_z = velCont[2].array(mfi);
        Box zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            ucont_x(i, j, k) = Real(0.5) * (ucart(i-1, j, k, 0) + ucart(i, j, k, 0)) ;
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            ucont_y(i, j, k) = Real(0.5) * (ucart(i, j-1, k, 1) + ucart(i, j, k, 1)) ;
        });
    }
}

void righthand_side_calc ( Array<MultiFab, AMREX_SPACEDIM>& rhs,
                           MultiFab& fluxConvect,
                           MultiFab& fluxViscous,
                           MultiFab& fluxPrsGrad,
                           Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                           Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                           MultiFab& userCtx,
                           MultiFab& velCart,
                           Array<MultiFab, AMREX_SPACEDIM>& velCont,
                           Geometry const& geom,
                           int const& n_cell,
                           Real const& ren )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    const Real& coef = Real(0.125);
    // ++++++++++++++++++++++++++++++ Convective Flux ++++++++++++++++++++++++++++++
    // QUICK scheme: Calculation of the half-node fluxes
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
        auto const& ucont_z = velCont[2].array(mfi);
#endif
        auto const& fpx1 = fluxHalfN1[0].array(mfi);
        auto const& fpx2 = fluxHalfN2[0].array(mfi);
        auto const& fpy1 = fluxHalfN1[1].array(mfi);
        auto const& fpy2 = fluxHalfN2[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& fpz1 = fluxHalfN1[2].array(mfi);
        auto const& fpz2 = fluxHalfN2[2].array(mfi);
#endif
        auto const& vcart = velCart.array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Contribution of x-directional terms: fpx1, fpx2
            if ( xcont(i, j, k) < 0 ) { // down stream
                if ( i==0 ) { // west wall (if any)
                    fpx1(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i+2, j, k, 0) - Real(2.0) * vcart(i+1, j, k, 0) + Real(3.0) * vcart(i, j, k, 0) ) + vcart(i+1, j, k, 0) );

                    fpx2(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i+2, j, k, 1) - Real(2.0) * vcart(i+1, j, k, 1) + Real(3.0) * vcart(i, j, k, 1) ) + vcart(i+1, j, k, 1) );
                }
                else if ( i==(n_cell-1) ) { // east wall (if any)
                    fpx1(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i+1, j, k, 0) - Real(2.0) * vcart(i+1, j, k, 0) + Real(3.0) * vcart(i, j, k, 0) ) + vcart(i+1, j, k, 0) );

                    fpx2(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i+1, j, k, 1) - Real(2.0) * vcart(i+1, j, k, 1) + Real(3.0) * vcart(i, j, k, 1) ) + vcart(i+1, j, k, 1) );
                }
                else { // inner domain
                    fpx1(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i+2, j, k, 0) - 2 * vcart(i+1, j, k, 0) + 3 * vcart(i, j, k, 0) ) + vcart(i+1, j, k, 0) );

                    fpx2(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i+2, j, k, 1) - Real(2.0) * vcart(i+1, j, k, 1) + Real(3.0) * vcart(i, j, k, 1) ) + vcart(i+1, j, k, 1) );
                }
            }
            else if ( xcont(i, j, k) > 0) { // up stream
                if ( i==0 ) { // west wall (if any)
                    fpx1(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, k, 0) - Real(2.0) * vcart(i, j, k, 0) + Real(3.0) * vcart(i+1, j, k, 0) ) + vcart(i, j, k, 0) );

                    fpx2(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, k, 1) - Real(2.0) * vcart(i, j, k, 1) + Real(3.0) * vcart(i+1, j, k, 1) ) + vcart(i, j, k, 1) );
                }
                else if ( i==(n_cell-1) ) { // east wall (if any)
                    fpx1(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i-1, j, k, 0) - Real(2.0) * vcart(i, j, k, 0) + Real(3.0) * vcart(i+1, j, k, 0) ) + vcart(i, j, k, 0) );

                    fpx2(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i-1, j, k, 1) - Real(2.0) * vcart(i, j, k, 1) + Real(3.0) * vcart(i+1, j, k, 1) ) + vcart(i, j, k, 1) );
                }
                else { // inner domain
                    fpx1(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i-1, j, k, 0) - Real(2.0) * vcart(i, j, k, 0) + Real(3.0) * vcart(i+1, j, k, 0) ) + vcart(i, j, k, 0) );

                    fpx2(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i-1, j, k, 1) - Real(2.0) * vcart(i, j, k, 1) + Real(3.0) * vcart(i+1, j, k, 1) ) + vcart(i, j, k, 1) );
                }
            }
            else { // zero
                fpx1(i, j, k) = Real(0.0);
                fpx2(i, j, k) = Real(0.0);
            }
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Contribution of y-directional terms: fpy1, fpy2
            if ( ycont(i, j, k) < 0 ) { // down stream
                if ( j==0 ) { // south wall (if any)
                    fpy1(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j+2, k, 0) - Real(2.0) * vcart(i, j+1, k, 0) + Real(3.0) * vcart(i, j, k, 0) ) + vcart(i, j+1, k, 0) );

                    fpy2(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j+2, k, 1) - Real(2.0) * vcart(i, j+1, k, 1) + Real(3.0) * vcart(i, j, k, 1) ) + vcart(i, j, k, 1) );
                }
                else if ( j==(n_cell-1) ) { // north wall (if any)
                    fpy1(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j+1, k, 0) - Real(2.0) * vcart(i, j+1, k, 0) + Real(3.0) * vcart(i, j, k, 0) ) + vcart(i, j+1, k, 0) );

                    fpy2(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j+1, k, 1) - Real(2.0) * vcart(i, j+1, k, 1) + Real(3.0) * vcart(i, j, k, 1) ) + vcart(i, j+1, k, 1) );
                }
                else { // inner domain
                    fpy1(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j+2, k, 0) - 2 * vcart(i, j+1, k, 0) + 3 * vcart(i, j, k, 0) ) + vcart(i, j+1, k, 0) );

                    fpy2(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j+2, k, 1) - Real(2.0) * vcart(i, j+1, k, 1) + Real(3.0) * vcart(i, j, k, 1) ) + vcart(i, j+1, k, 1) );
                }
            }
            else if ( ycont(i, j, k) > 0) { // up stream
                if ( j==0 ) { // south wall (if any)
                    fpy1(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, k, 0) - Real(2.0) * vcart(i, j, k, 0) + Real(3.0) * vcart(i, j+1, k, 0) ) + vcart(i, j, k, 0) );

                    fpy2(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, k, 1) - Real(2.0) * vcart(i, j, k, 1) + Real(3.0) * vcart(i, j+1, k, 1) ) + vcart(i, j, k, 1) );
                }
                else if ( j==(n_cell-1) ) { // north wall (if any)
                    fpy1(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j-1, k, 0) - Real(2.0) * vcart(i, j, k, 0) + Real(3.0) * vcart(i, j+1, k, 0) ) + vcart(i, j, k, 0) );

                    fpy2(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j-1, k, 1) - Real(2.0) * vcart(i, j, k, 1) + Real(3.0) * vcart(i, j+1, k, 1) ) + vcart(i, j, k, 1) );
                }
                else { // inner domain
                    fpy1(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j-1, k, 0) - Real(2.0) * vcart(i, j, k, 0) + Real(3.0) * vcart(i, j+1, k, 0) ) + vcart(i, j, k, 0) );

                    fpy2(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j-1, k, 1) - Real(2.0) * vcart(i, j, k, 1) + Real(3.0) * vcart(i, j+1, k, 1) ) + vcart(i, j, k, 1) );
                }
            }
            else { // zero
                fpy1(i, j, k) = Real(0.0);
                fpy2(i, j, k) = Real(0.0);
            }
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxConvect); mfi.isValid(); ++mfi ) {
        const Box& vbx = mfi.validbox();
        auto const& conv_flux = fluxConvect.array(mfi);

        auto const& fpx1 = fluxHalfN1[0].array(mfi);
        auto const& fpx2 = fluxHalfN2[0].array(mfi);
        auto const& fpy1 = fluxHalfN1[1].array(mfi);
        auto const& fpy2 = fluxHalfN2[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& fpz1 = fluxHalfN1[2].array(mfi);
        auto const& fpz2 = fluxHalfN2[2].array(mfi);
#endif
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){
           conv_flux(i, j, k, 0) = (fpx1(i+1, j, k) - fpx1(i, j, k))/(dx[0]) + (fpy1(i, j+1, k) - fpy1(i, j, k))/(dx[1]);
           conv_flux(i, j, k, 1) = (fpx2(i+1, j, k) - fpx2(i, j, k))/(dx[0]) + (fpy2(i, j+1, k) - fpy2(i, j, k))/(dx[1]);
#if (AMREX_SPACEDIM > 2)
           conv_flux(i, j, k, 2) = Real(0.0);
#endif
        });
    }

    // ++++++++++++++++++++++++++++++ Viscous Flux ++++++++++++++++++++++++++++++
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxViscous); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart = velCart.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
            {
                auto const& centerMAC = vcart(i, j, k, dir);
                auto const& northMAC = vcart(i, j+1, k, dir);
                auto const& southMAC = vcart(i, j-1, k, dir);
                auto const& westMAC = vcart(i-1, j, k, dir);
                auto const& eastMAC = vcart(i+1, j, k, dir);

                visc_flux(i, j, k, dir) = ((westMAC - 2*centerMAC + eastMAC)/(dx[0]*dx[0]) + (southMAC - 2*centerMAC + northMAC)/(dx[1]*dx[1]))/ren;
            }
        });
    }
    // +++++++++++++++++++++++++ Presure Gradient Flux  +++++++++++++++++++++++++
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxPrsGrad); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& prs_field = userCtx.array(mfi);
        auto const& prsgrad_flux = fluxPrsGrad.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( i==0 ) {
                prsgrad_flux(i, j, k, 0) = Real(0.0);
            }
            else if ( i==(n_cell-1) ) {
                prsgrad_flux(i, j, k, 0) = Real(0.0);
            }
            else if ( i==1 ) {
                prsgrad_flux(i, j, k, 0) = (prs_field(i+1, j, k, 0) - prs_field(i, j, k, 0))/(dx[0]);
            }
            else if ( i==(n_cell-2) ) {
                prsgrad_flux(i, j, k, 0) = (prs_field(i, j, k, 0) - prs_field(i-1, j, k, 0))/(dx[0]);
            }
            else {
                prsgrad_flux(i, j, k, 0) = (prs_field(i+1, j, k, 0) - prs_field(i-1, j, k, 0))/(Real(2.0)*dx[0]);
            }

            if ( i==0 ) {
                prsgrad_flux(i, j, k, 1) = Real(0.0);
            }
            else if ( i==(n_cell-1) ) {
                prsgrad_flux(i, j, k, 1) = Real(0.0);
            }
            else if ( i==1 ) {
                prsgrad_flux(i, j, k, 1) = (prs_field(i, j+1, k, 1) - prs_field(i, j, k, 1))/(dx[1]);
            }
            else if ( i==(n_cell-2) ) {
                prsgrad_flux(i, j, k, 1) = (prs_field(i, j, k, 1) - prs_field(i, j-1, k, 1))/(dx[1]);
            }
            else {
                prsgrad_flux(i, j, k, 1) = (prs_field(i, j+1, k, 1) - prs_field(i, j-1, k, 1))/(Real(2.0)*dx[1]);
            }
        });
    }

    // +++++++++++++++++++++++++ Total Flux and the RHS  +++++++++++++++++++++++++
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(rhs[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& xrhs = rhs[0].array(mfi);
        auto const& yrhs = rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zrhs = rhs[2].array(mfi);
#endif
        auto const& conv_flux = fluxConvect.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
        auto const& prsgrad_flux = fluxPrsGrad.array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( i==0 ) {
                xrhs(i, j, k) = Real(0.0);
            }
            else if ( i==(n_cell-1) ) {
                xrhs(i, j, k) = Real(0.0);
            }
            else {
                // total_flux_my_volume and total_flux_pr_volume live in volume center
                auto const& total_flux_my_volume = conv_flux(i, j, k, 0) + visc_flux(i, j, k, 0) + prsgrad_flux(i, j, k, 0);
                auto const& total_flux_pr_volume = conv_flux(i-1, j, k, 0) + visc_flux(i-1, j, k, 0) + prsgrad_flux(i-1, j, k, 0);
                xrhs(i, j, k) = Real(0.5)*(total_flux_my_volume + total_flux_pr_volume);
            }
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( j==0 ) {
                yrhs(i, j, k) = Real(0.0);
            }
            else if ( j==(n_cell-1) ) {
                yrhs(i, j, k) = Real(0.0);
            }
            else {
                auto const& total_flux_my_volume = conv_flux(i, j, k, 1) + visc_flux(i, j, k, 1) + prsgrad_flux(i, j, k, 1);
                auto const& total_flux_pr_volume = conv_flux(i, j-1, k, 0) + visc_flux(i, j-1, k, 0) + prsgrad_flux(i, j-1, k, 0);
                yrhs(i, j, k) = Real(0.5)*(total_flux_my_volume + total_flux_pr_volume);
            }
        });
    }
}

void momentum_km_runge_kutta ( Array<MultiFab, AMREX_SPACEDIM>& rhs,
                               MultiFab& fluxConvect,
                               MultiFab& fluxViscous,
                               MultiFab& fluxPrsGrad,
                               Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                               Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                               MultiFab& userCtx,
                               MultiFab& velCart,
                               Array<MultiFab, AMREX_SPACEDIM>& velCont,
                               Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                               Real const& dt,
                               Geometry const& geom,
                               int const& n_cell,
                               Real const& ren )
{
    //--Runge-Kutta time integration
    Real Tol = 1.0e-8;
    int IterNumCycle = 50;
    // Setup stopping conditions
    Real normError = 1.0e8;
    int countIter = 0;
    // Setup Runge-Kutta intermediate coefficients
    int RungeKuttaOrder = 4;
    Vector<Real> rk(RungeKuttaOrder, 0);
    {
        rk[0] = Real(0.25);
        rk[1] = Real(1.0)/Real(3.0);
        rk[2] = Real(0.5);
        rk[3] = Real(1.0);
    }
    // Runge-Kutta time integration to update the contravariant velocity components
    // Loop over stopping condition
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    while ( countIter <= IterNumCycle || normError > Tol )
    {
        countIter++;
        amrex::Print() << "MOMENTUM | Performing Runge-Kutta at iteration: " << countIter << "\n";

        for (int n = 0; n < RungeKuttaOrder; ++n )
        {
            // Calculating the righ-hand-side term
            righthand_side_calc(rhs, fluxConvect, fluxViscous, fluxPrsGrad, fluxHalfN1, fluxHalfN2, userCtx, velCart, velCont, geom, n_cell, ren);
            for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi  )
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
                auto const& xdiff = velContDiff[0].array(mfi);
                auto const& ydiff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& zdiff = velContDiff[2].array(mfi);
#endif
                auto const& xrhs = rhs[0].array(mfi);
                auto const& yrhs = rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& zcont = rhs[2].array(mfi);
#endif
                amrex::ParallelFor(xbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // Itermidiate velocity
                    Real xhat = xcont(i, j, k);
                    // Corection for right-hand-side term
                    xrhs(i, j, k) = xrhs(i, j, k) - (Real(0.5)/dt)*(xhat - xrhs(i, j, k)) + (Real(0.5)/dt)*(xdiff(i, j, k));
                    // RK4 substep to update the immediate velocity
                    xhat = xcont(i, j, k) + rk[n]*dt*xrhs(i,j,k);
                    xcont(i, j, k) = xhat;
                });
                amrex::ParallelFor(ybx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    Real yhat = ycont(i, j, k);
                    yrhs(i, j, k) = yrhs(i, j, k) - (Real(0.5)/dt)*(yhat - yrhs(i, j, k)) + (Real(0.5)/dt)*(ydiff(i, j, k));
                    yhat = ycont(i, j, k) + rk[n]*dt*yrhs(i,j,k);
                    ycont(i, j, k) = yhat;
                });
#if (AMREX_SPACEDIM > 2)
                amrex::ParallelFor(zbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    Real zhat = zcont(i, j, k);
                    zrhs(i, j, k) = zrhs(i, j, k) - (Real(0.5)/dt)*(zhat - zrhs(i, j, k)) + (Real(0.5)/dt)*(zdiff(i, j, k));
                    zhat = zcont(i, j, k) + rk[n]*dt*zrhs(i,j,k);
                    zcont(i, j, k) = zhat;
                });
#endif
            }
        }
    }
}
