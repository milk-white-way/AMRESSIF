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

void init_userCTX(MultiFab& userCtx, Geometry const& geom){

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(userCtx); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto const& userCtxNow = userCtx.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, userCtxNow, dx, prob_lo);
        });
    }
}

void init_velocity (Array<MultiFab, AMREX_SPACEDIM>& velCont,
                    MultiFab& velCart,
                    Geometry const& geom)
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    {

        Box xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        Box ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
        auto const& ucont_x = velCont[0].array(mfi);
        auto const& ucont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        Box zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));

        auto const& ubcs_z = velBcs[2].array(mfi);
        auto const& ucon_z = velCont[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = i*dx[0];
            Real y = ( j+0.5 )*dx[1];

            ucont_x(i,j,k) = - std::cos(2.0 * M_PI * x)*std::sin(2.0 * M_PI * y);
#if (AMREX_SPACEDIM > 2)
            Real z = k*dx[2];

            ubcs_x(i,j,k) = 0;
            vel_x(i,j,k) = vel_x(i,j,k)*std::sin( 2.0*M_PI*z );
#endif
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = (i + 0.5)*dx[0];
            Real y = j*dx[1];

            ucont_y(i,j,k) = std::sin( 2.0*M_PI*x )*std::cos( 2.0*M_PI*y );
#if (AMREX_SPACEDIM > 2)
            Real z = k*dx[2];

            ucon_y(i,j,k) = vel_y(i,j,k)*std::sin( 2.0*M_PI*z )
#endif
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // 3D stuffs
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
                   ucart(i, j, k, 0) = - ucart(i+1, j, k, 0);
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
                   ucart(i, j, k, 0) = - ucart(i, j+1, k, 0);
                   ucart(i, j, k, 1) = ucart(i, j+1, k, 1);
#if (AMREX_SPACEDIM > 2)
                   velCart(i, j, k, 2) = velCart(i, j+1, k, 2);
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
                   ucart(i, j, k, 0) = - ucart(i, j-1, k, 0);
                   ucart(i, j, k, 1) = ucart(i, j-1, k, 1);
#if (AMREX_SPACEDIM > 2)
                   ucart(i, j, k, 2) = ucart(i, j-1, k, 2);
#endif
               }
           }
       });
   }
}

void cartesian_velocity_interpolation (MultiFab& velCart,
                                       Array<MultiFab, AMREX_SPACEDIM>& velCont,
                                       MultiFab& velBcs,
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

void viscous_flux_calc (MultiFab& fluxViscous,
                        MultiFab& velCart,
                        Geometry const& geom,
                        Real const& ren){

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& cbx = mfi.validbox();

        auto const& ucart = velCart.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& ucart_z = velocity[2].array(mfi);
        auto const& visc_flux_z = fluxViscous[2].array(mfi);
#endif

        amrex::ParallelFor(cbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
            {
                auto const& centerMAC = ucart(i, j, k, dir);
                auto const& northMAC = ucart(i, j+1, k, dir);
                auto const& southMAC = ucart(i, j-1, k, dir);
                auto const& westMAC = ucart(i-1, j, k, dir);
                auto const& eastMAC = ucart(i+1, j, k, dir);

                visc_flux(i, j, k, dir) = ((westMAC - 2*centerMAC + eastMAC)/(dx[0]*dx[0]) + (southMAC - 2*centerMAC + northMAC)/(dx[1]*dx[1]))/ren;
            }
        });
    }
}
/*
void convective_flux_calc (Array<MultiFab, AMREX_SPACEDIM>& convective_flux, Array<MultiFab, AMREX_SPACEDIM>& velocity, Geometry const& geom, Real const& ren){

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velocity[0]); mfi.isValid(); ++mfi )
    {
        Box xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        Box ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));

        auto const& vel_x = velocity[0].array(mfi);
        auto const& vel_y = velocity[1].array(mfi);

        auto const& conv_flux_x = viscous_flux[0].array(mfi);
        auto const& conv_flux_y = viscous_flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        Box zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
        auto const& vel_z = velocity[2].array(mfi);
        auto const& visc_flux_z = viscous_flux[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            auto const& eastQUICK = 0.5*(vel_x(i+1, j, k) + vel_x(i, j, k)) - 0.125*(vel_x(i+1, j, k) - 2*vel_x(i, j, k) + vel_x(i-1, j, kl));
            auto const& westQUICK = 0.5*(vel_x(i, j, k) + vel_x(i-1, j, k)) - 0.125*(vel_x(i, j, k) - 2*vel_x(i-1, j, k) + vel_x(i-2, j, kl));


            visc_flux_x(i, j, k) = ((westmac - 2*centermac + eastmac)/(dx[0]*dx[0]) + (southmac - 2*centermac + northmac)/(dx[1]*dx[1]))/ren;
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // y-stencils
            auto const& northmac = vel_y(i, j+1, k);
            auto const& southmac = vel_y(i, j-1, k);

            auto const& westmac = vel_y(i-1, j, k);
            auto const& eastmac = vel_y(i+1, j, k);

            auto const& centermac = vel_y(i, j, k);

            visc_flux_y(i, j, k) = ((westmac - 2*centermac + eastmac)/(dx[0]*dx[0]) + (southmac - 2*centermac + northmac)/(dx[1]*dx[1]))/ren;
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // dummy code
            auto const& northmac = vel_z(i, j+1, k);
            auto const& southmac = vel_z(i, j-1, k);

            auto const& westmac = vel_z(i-1, j, k);
            auto const& eastmac = vel_z(i+1, j, k);

            auto const& centermac = vel_z(i, j, k);

            visc_flux_z(i, j, k) = ((westmac - 2*centermac + eastmac)/(dx[0]*dx[0]) + (southmac - 2*centermac + northmac)/(dx[1]*dx[1]))/ren;
        });
#endif
    }
}
*/
