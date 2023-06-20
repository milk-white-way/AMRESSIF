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

void init_velocity (MultiFab& userCtx, Array<MultiFab, AMREX_SPACEDIM>& velocity, Geometry const& geom){

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
#if (AMREX_SPACEDIM > 2)
        auto const& vel_z = velocity[2].array(mfi);
        Box zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = i*dx[0];
            Real y = (j + 0.5)*dx[1];

            // vel_x(i,j,k) = Real(0.0);
            vel_x(i,j,k) = std::sin(2.0 * M_PI * x)*std::cos(2.0 * M_PI * y);
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = (i + 0.5)*dx[0];
            Real y = j*dx[1];

            // vel_y(i,j,k) = Real(0.0);
            vel_y(i,j,k) = - std::cos(2.0 * M_PI * x)*std::sin(2.0 * M_PI * y);
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            vel_z(i,j,k) = Real(0.0);
        });
#endif
    }
}

void viscous_flux_calc (Array<MultiFab, AMREX_SPACEDIM>& viscous_flux, Array<MultiFab, AMREX_SPACEDIM>& velocity, Geometry const& geom, Real const& ren){

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

        auto const& visc_flux_x = viscous_flux[0].array(mfi);
        auto const& visc_flux_y = viscous_flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        Box zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
        auto const& vel_z = velocity[2].array(mfi);
        auto const& visc_flux_z = viscous_flux[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // x-stencils
            auto const& northmac = vel_x(i, j+1, k);
            auto const& southmac = vel_x(i, j-1, k);

            auto const& westmac = vel_x(i-1, j, k);
            auto const& eastmac = vel_x(i+1, j, k);

            auto const& centermac = vel_x(i, j, k);

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
