#include <AMReX_MultiFabUtil.H>
// #include <AMReX_Utility.H>
// #include <AMReX_PlotFileUtil.H>
// #include <AMReX_Print.H>

#include "myfunc.H"
#include "mykernel.H"
#include "myflux.H"

using namespace amrex;

// ================================= MODULE | INITIALIZATION =================================
void init (amrex::MultiFab& userCtx,
           amrex::MultiFab& velCart,
           amrex::MultiFab& velCartDiff,
           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
           amrex::Geometry const& geom)
{

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, ctx, dx, prob_lo);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart = velCart.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity(i, j, k, vcart, dx, prob_lo);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCartDiff); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart_diff = velCartDiff.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity_difference(i, j, k, vcart_diff);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velContDiff[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& xcont_diff = velContDiff[0].array(mfi);
        auto const& ycont_diff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zcont_diff = velContDiff[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_contravariant_velocity_difference(i, j, k, xcont_diff);
        });
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_contravariant_velocity_difference(i, j, k, ycont_diff);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_contravariant_velocity_difference(i, j, k, zcont_diff);
        });
#endif
    }
}

// ==================================== MODULE | MOMENTUM ====================================
void convective_flux_calc ( MultiFab& fluxConvect,
                            Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                            Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                            Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                            MultiFab& velCart,
                            Array<MultiFab, AMREX_SPACEDIM>& velCont,
                            Vector<int> const& phy_bc_lo,
                            Vector<int> const& phy_bc_hi,
                            Geometry const& geom,
                            int const& n_cell )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    const Real& qkcoef = Real(0.125);
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
       auto const& zcont = velCont[2].array(mfi);
#endif
        auto const& fluxx_xcont = fluxHalfN1[0].array(mfi);
        auto const& fluxy_xcont = fluxHalfN2[0].array(mfi);
        auto const& fluxz_xcont = fluxHalfN3[0].array(mfi);

        auto const& fluxx_ycont = fluxHalfN1[1].array(mfi);
        auto const& fluxy_ycont = fluxHalfN2[1].array(mfi);
        auto const& fluxz_ycont = fluxHalfN3[1].array(mfi);

#if (AMREX_SPACEDIM > 2)
        auto const& fluxx_zcont = fluxHalfN1[2].array(mfi);
        auto const& fluxy_zcont = fluxHalfN2[2].array(mfi);
        auto const& fluxz_zcont = fluxHalfN3[2].array(mfi);
#endif
        auto const& vcart = velCart.array(mfi);

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( west_wall_bcs==0 && east_wall_bcs==0 ) {
                compute_halfnode_convective_flux_x_contrib_periodic(i, j, k, fluxx_xcont, fluxy_xcont, fluxz_xcont, xcont, vcart, qkcoef);
            } else {
                compute_halfnode_convective_flux_x_contrib_wall(i, j, k, fluxx_xcont, fluxy_xcont, fluxz_xcont, xcont, vcart, qkcoef, n_cell);
            }
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( south_wall_bcs==0 && north_wall_bcs==0 ) {
                compute_halfnode_convective_flux_y_contrib_periodic(i, j, k, fluxx_ycont, fluxy_ycont, fluxz_ycont, ycont, vcart, qkcoef);
            } else {
                compute_halfnode_convective_flux_y_contrib_wall(i, j, k, fluxx_ycont, fluxy_ycont, fluxz_ycont, ycont, vcart, qkcoef, n_cell);
            }
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( front_wall_bcs==0 && back_wall_bcs==0 ) {
                compute_halfnode_convective_flux_z_contrib_periodic(i, j, k, fluxx_zcont, fluxy_zcont, fluxz_zcont, zcont, vcart, qkcoef);
            } else {
                compute_halfnode_convective_flux_z_contrib_wall(i, j, k, fluxx_zcont, fluxy_zcont, fluxz_zcont, zcont, vcart, qkcoef, n_cell);
            }
        });
#endif
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxConvect); mfi.isValid(); ++mfi ) {
        const Box& vbx = mfi.validbox();
        auto const& conv_flux = fluxConvect.array(mfi);

        auto const& fluxx_xcont = fluxHalfN1[0].array(mfi);
        auto const& fluxy_xcont = fluxHalfN2[0].array(mfi);
        auto const& fluxz_xcont = fluxHalfN3[0].array(mfi);

        auto const& fluxx_ycont = fluxHalfN1[1].array(mfi);
        auto const& fluxy_ycont = fluxHalfN2[1].array(mfi);
        auto const& fluxz_ycont = fluxHalfN3[1].array(mfi);

#if (AMREX_SPACEDIM > 2)
        auto const& fluxx_zcont = fluxHalfN1[2].array(mfi);
        auto const& fluxy_zcont = fluxHalfN2[2].array(mfi);
        auto const& fluxz_zcont = fluxHalfN3[2].array(mfi);
#endif
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            conv_flux(i, j, k, 0) = (fluxx_xcont(i, j, k) - fluxx_xcont(i+1, j, k))/(dx[0]) + (fluxx_ycont(i, j, k) - fluxx_ycont(i, j+1, k))/(dx[1])
#if (AMREX_SPACEDIM > 2)
                + (fluxx_zcont(i, j, k) - fluxx_zcont(i, j, k+1))/(dx[2]);
#else
            ;
#endif

            conv_flux(i, j, k, 1) = (fluxy_xcont(i, j, k) - fluxy_xcont(i+1, j, k))/(dx[0]) + (fluxy_ycont(i, j, k) - fluxy_ycont(i, j+1, k))/(dx[1])
#if (AMREX_SPACEDIM > 2)
                + (fluxy_zcont(i, j, k) - fluxy_zcont(i, j, k+1))/(dx[2]);
#else
            ;
#endif

#if (AMREX_SPACEDIM > 2)
            conv_flux(i, j, k, 2) = (fluxz_xcont(i, j, k) - fluxz_xcont(i+1, j, k))/(dx[0]) + (fluxz_ycont(i, j, k) - fluxz_ycont(i, j+1, k))/(dx[1]) + (fluxz_zcont(i, j, k) - fluxz_zcont(i, j, k+1))/(dx[2]);
#endif
        });
    }
}

void viscous_flux_calc ( MultiFab& fluxViscous,
                         MultiFab& velCart,
                         Geometry const& geom,
                         Real const& ren )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

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
            for ( int dir=0; dir < AMREX_SPACEDIM; ++dir )
            {
                auto const& centerMAC = vcart(i, j, k, dir);
                auto const& northMAC = vcart(i, j+1, k, dir);
                auto const& southMAC = vcart(i, j-1, k, dir);
                auto const& westMAC = vcart(i-1, j, k, dir);
                auto const& eastMAC = vcart(i+1, j, k, dir);

                visc_flux(i, j, k, dir) = ( (westMAC - 2*centerMAC + eastMAC)/(dx[0]*dx[0]) + (southMAC - 2*centerMAC + northMAC)/(dx[1]*dx[1]) )/ren;
            }
        });
    }
}

void pressure_gradient_calc ( MultiFab& fluxPrsGrad,
                              MultiFab& userCtx,
                              Geometry const& geom )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

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
            prsgrad_flux(i, j, k, 0) = (prs_field(i+1, j, k, 0) - prs_field(i-1, j, k, 0))/(Real(2.0)*dx[0]);
            prsgrad_flux(i, j, k, 1) = (prs_field(i, j+1, k, 0) - prs_field(i, j-1, k, 0))/(Real(2.0)*dx[1]);
#if (AMREX_SPACEDIM > 2)
            prsgrad_flux(i, j, k, 2) = (prs_field(i, j, k+1, 0) - prs_field(i, j, k-1, 0))/(Real(2.0)*dx[2]);
#endif
        });
    }
}

void total_flux_calc ( MultiFab& fluxTotal,
                       MultiFab& fluxConvect,
                       MultiFab& fluxViscous,
                       MultiFab& fluxPrsGrad )
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxTotal); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        // Components
        auto const& conv_flux = fluxConvect.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
        auto const& prsgrad_flux = fluxPrsGrad.array(mfi);

        auto const& total_flux = fluxTotal.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_total_flux(i, j, k, total_flux, conv_flux, visc_flux, prsgrad_flux);
        });
    }
}

void righthand_side_calc ( Array<MultiFab, AMREX_SPACEDIM>& rhs,
                           MultiFab& fluxTotal )
{
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
        auto const& total_flux = fluxTotal.array(mfi);

        int const& box_id = mfi.LocalIndex();
        //print_box(box_id);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_x(i, j, k, xrhs, total_flux); });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_y(i, j, k, yrhs, total_flux); });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_z(i, j, k, zrhs, total_flux); });
#endif
    }
}

// +++++++++++++++++++++++++ Subroutine | Kim and Moine's Runge-Kutta ++++++++++++++++++++++++

// ==================================== MODULE | POISSON =====================================
// Dathi's Module

// ==================================== MODULE | ADVANCE =====================================
/*
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
*/

// ==================================== UTILITY | CONVERSION  ================================
void copy_contravariant_velocity (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContPrev)
{
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

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ xprev(i, j, k) = xcont(i, j, k); });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k){ yprev(i, j, k) = ycont(i, j, k); });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k){ zprev(i, j, k) = zcont(i, j, k); });
#endif
    }
}

void cont2cart (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont,
                Geometry const& geom)
{
    average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);
}

void cart2cont (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont)
{
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
        auto const& vcart = velCart.const_array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_x(i, j, k, xcont, vcart); });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_y(i, j, k, ycont, vcart); });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_z(i, j, k, zcont, vcart); });
#endif
    }
}

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void manual_fill_ghost_cells (MultiFab& velCart,
                              MultiFab& userCtx,
                              int const& Nghost,
                              Vector<int> const& phy_bc_lo,
                              Vector<int> const& phy_bc_hi,
                              int const& n_cell)
{
    for (MFIter mfi(velCart); mfi.isValid(); ++mfi)
    {
        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif
        const Box& gbx = mfi.growntilebox(Nghost);
        auto const& ctx = userCtx.array(mfi);
        auto const& vcart = velCart.array(mfi);

        if ( west_wall_bcs != 0 ) {
            // amrex::Print() << "========================== west wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_west(i, j, k, n_cell, west_wall_bcs, vcart, ctx);
            });
        }

        if ( east_wall_bcs != 0 ) {
            // amrex::Print() << "========================== east wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_east(i, j, k, n_cell, east_wall_bcs, vcart, ctx);
            });
        }

        if ( south_wall_bcs != 0 ) {
            // amrex::Print() << "========================= south wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_south(i, j, k, n_cell, south_wall_bcs, vcart, ctx);
            });
        }

        if ( north_wall_bcs != 0 ) {
            // amrex::Print() << "========================= north wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_north(i, j, k, n_cell, north_wall_bcs, vcart, ctx);
            });
        }
#if (AMREX_SPACEDIM > 2)
        if ( front_wall_bcs != 0 ) {
            // amrex::Print() << "========================= front wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_front(i, j, k, n_cell, front_wall_bcs, vcart. ctx);
            });
        }

        if ( back_wall_bcs != 0 ) {
            // amrex::Print() << "========================== back wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_back(i, j, k, n_cell, back_wall_bcs, vcart, ctx);
            });
        }
#endif
    }
}

void enforce_boundary_conditions (MultiFab& inputFlux,
                                  int const& Nghost,
                                  Vector<int> const& phy_bc_lo,
                                  Vector<int> const& phy_bc_hi,
                                  int const& n_cell)
{
    for (MFIter mfi(inputFlux); mfi.isValid(); ++mfi)
    {
        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif
        const Box& gbx = mfi.growntilebox(Nghost);
        auto const& flux = inputFlux.array(mfi);

        if ( west_wall_bcs != 0 ) {
            // amrex::Print() << "========================== west wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                enforcing_flux_bcs_west(i, j, k, n_cell, west_wall_bcs, flux);
            });
        }

        if ( east_wall_bcs != 0 ) {
            // amrex::Print() << "========================== east wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                enforcing_flux_bcs_east(i, j, k, n_cell, east_wall_bcs, flux);
            });
        }

        if ( south_wall_bcs != 0 ) {
            // amrex::Print() << "========================= south wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                enforcing_flux_bcs_south(i, j, k, n_cell, south_wall_bcs, flux);
            });
        }

        if ( north_wall_bcs != 0 ) {
            // amrex::Print() << "========================= north wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                enforcing_flux_bcs_north(i, j, k, n_cell, north_wall_bcs, flux);
            });
        }
#if (AMREX_SPACEDIM > 2)
        if ( front_wall_bcs != 0 ) {
            // amrex::Print() << "========================= front wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                enforcing_flux_bcs_front(i, j, k, n_cell, front_wall_bcs, flux);
            });
        }

        if ( back_wall_bcs != 0 ) {
            // amrex::Print() << "========================== back wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                enforcing_flux_bcs_back(i, j, k, n_cell, back_wall_bcs, flux);
            });
        }
#endif
    }
}
