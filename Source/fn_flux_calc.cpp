#include <AMReX_MultiFabUtil.H>

#include "fn_flux_calc.H"
#include "kn_flux_calc.H"
#include "kn_poisson.H"

using namespace amrex;

// ++++++++++++++++++++++++++++++ Convective Flux ++++++++++++++++++++++++++++++
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

    const Real& alp = Real(0.5);
    const Real& bet = Real(0.125);
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
                auto const& ucon = Real(0.5) * xcont(i, j, k);
                auto const& up = ucon + std::abs(ucon);
                auto const& um = ucon - std::abs(ucon);

                // West face for Quick scheme
                auto const& ucat_x_ww = vcart(i-2, j, k, 0);
                auto const& ucat_x_w  = vcart(i-1, j, k, 0);
                auto const& ucat_x_p  = vcart(i  , j, k, 0);
                auto const& ucat_x_e  = vcart(i+1, j, k, 0);

                auto const& ucat_y_ww = vcart(i-2, j, k, 1);
                auto const& ucat_y_w  = vcart(i-1, j, k, 1);
                auto const& ucat_y_p  = vcart(i  , j, k, 1);
                auto const& ucat_y_e  = vcart(i+1, j, k, 1);

                fluxx_xcont(i, j, k) = up * ( alp * (ucat_x_p + ucat_x_w) - bet * (ucat_x_ww - 2*ucat_x_w + ucat_x_p) )  + um * ( alp * (ucat_x_p + ucat_x_w) - bet * (ucat_x_w - 2*ucat_x_p + ucat_x_e) );

                fluxy_xcont(i, j, k) = up * ( alp * (ucat_y_p + ucat_y_w) - bet * (ucat_y_ww - 2*ucat_y_w + ucat_y_p) )  + um * ( alp * (ucat_y_p + ucat_y_w) - bet * (ucat_y_w - 2*ucat_y_p + ucat_y_e) );
            } else {
                compute_halfnode_convective_flux_x_contrib_wall(i, j, k, fluxx_xcont, fluxy_xcont, fluxz_xcont, xcont, vcart, bet, n_cell);
            }
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( south_wall_bcs==0 && north_wall_bcs==0 ) {
                auto const& ucon = Real(0.5) * ycont(i, j, k);
                auto const& up = ucon + std::abs(ucon);
                auto const& um = ucon - std::abs(ucon);

                // South face for Quick scheme
                auto const& ucat_x_ss = vcart(i, j-2, k, 0);
                auto const& ucat_x_s  = vcart(i, j-1, k, 0);
                auto const& ucat_x_p  = vcart(i, j  , k, 0);
                auto const& ucat_x_n  = vcart(i, j+1, k, 0);

                auto const& ucat_y_ss = vcart(i, j-2, k, 1);
                auto const& ucat_y_s  = vcart(i, j-1, k, 1);
                auto const& ucat_y_p  = vcart(i, j  , k, 1);
                auto const& ucat_y_n  = vcart(i, j+1, k, 1);

                fluxx_ycont(i, j, k) = um * ( alp * (ucat_x_p + ucat_x_s) - bet * (ucat_x_ss - 2*ucat_x_s + ucat_x_p) )  + up * ( alp * (ucat_x_p + ucat_x_s) - bet * (ucat_x_s - 2*ucat_x_p + ucat_x_n) );
                
                fluxy_ycont(i, j, k) = um * ( alp * (ucat_y_p + ucat_y_s) - bet * (ucat_y_ss - 2*ucat_y_s + ucat_y_p) )  + up * ( alp * (ucat_y_p + ucat_y_s) - bet * (ucat_y_s - 2*ucat_y_p + ucat_y_n) );
            } else {
                compute_halfnode_convective_flux_y_contrib_wall(i, j, k, fluxx_ycont, fluxy_ycont, fluxz_ycont, ycont, vcart, bet, n_cell);
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
            conv_flux(i, j, k, 0) = (fluxx_xcont(i+1, j, k) - fluxx_xcont(i, j, k))/(dx[0]) + (fluxx_ycont(i, j+1, k) - fluxx_ycont(i, j, k))/(dx[1])
#if (AMREX_SPACEDIM > 2)
                + (fluxx_zcont(i, j, k+1) - fluxx_zcont(i, j, k))/(dx[2]);
#else
            ;
#endif

            conv_flux(i, j, k, 1) = (fluxy_xcont(i+1, j, k) - fluxy_xcont(i, j, k))/(dx[0]) + (fluxy_ycont(i, j+1, k) - fluxy_ycont(i, j, k))/(dx[1])
#if (AMREX_SPACEDIM > 2)
                + (fluxy_zcont(i, j, k+1) - fluxy_zcont(i, j, k))/(dx[2]);
#else
            ;
#endif

#if (AMREX_SPACEDIM > 2)
            conv_flux(i, j, k, 2) = (fluxz_xcont(i+1, j, k) - fluxz_xcont(i, j, k))/(dx[0]) + (fluxz_ycont(i, j+1, k) - fluxz_ycont(i, j, k))/(dx[1]) + (fluxz_zcont(i, j, k+1) - fluxz_zcont(i, j, k))/(dx[2]);
#endif
        });
    }
}

// ++++++++++++++++++++++++++++++ Viscous Flux ++++++++++++++++++++++++++++++
void viscous_flux_calc ( MultiFab& fluxViscous,
                         MultiFab& velCart,
                         Geometry const& geom,
                         Real const& ren )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxViscous); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vel_cart = velCart.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // 2D PERIODIC ONLY
            compute_viscous_flux_periodic(i, j, k, visc_flux, dx, vel_cart, ren);
        });
    }
}

// +++++++++++++++++++++++++ Presure Gradient Flux  +++++++++++++++++++++++++
void pressure_gradient_calc ( MultiFab& fluxPrsGrad,
                              MultiFab& userCtx,
                              Geometry const& geom )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxPrsGrad); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& pressurefield = userCtx.array(mfi);
        auto const& presgrad_flux = fluxPrsGrad.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // PERIODIC ONLY
            compute_pressure_gradient_periodic(i, j, k, presgrad_flux, dx, pressurefield);
        });
    }
}

// +++++++++++++++++++++++++ Total Flux  +++++++++++++++++++++++++
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
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            compute_total_flux(i, j, k, total_flux, conv_flux, visc_flux, prsgrad_flux);
        });
    }
}
