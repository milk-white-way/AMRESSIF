#include <AMReX_MultiFabUtil.H>

#include "fn_rhs.H"

using namespace amrex;

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
        auto const& flux_xcont_xface = fluxHalfN1[0].array(mfi);
        auto const& flux_ycont_xface = fluxHalfN2[0].array(mfi);
        auto const& flux_xcont_yface = fluxHalfN1[1].array(mfi);
        auto const& flux_ycont_yface = fluxHalfN2[1].array(mfi);
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

        auto const& flux_xcont_xface = fluxHalfN1[0].array(mfi);
        auto const& flux_ycont_xface = fluxHalfN2[0].array(mfi);
        auto const& flux_xcont_yface = fluxHalfN1[1].array(mfi);
        auto const& flux_ycont_yface = fluxHalfN2[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& flux_xcont_zface = fluxHalfN1[2].array(mfi);
        auto const& flux_ycont_zface = fluxHalfN2[2].array(mfi);

        auto const& flux_zcont_xface = fluxHalfN3[0].array(mfi);
        auto const& flux_zcont_yface = fluxHalfN3[1].array(mfi);
        auto const& flux_zcont_zface = fluxHalfN3[2].array(mfi);
#endif
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){
           conv_flux(i, j, k, 0) = (flux_xcont_xface(i+1, j, k) - flux_xcont_xface(i, j, k))/(dx[0]) + (flux_ycont_xface(i, j+1, k) - flux_ycont_xface(i, j, k))/(dx[1]);
           conv_flux(i, j, k, 1) = (flux_xcont_yface(i+1, j, k) - flux_xcont_yface(i, j, k))/(dx[0]) + (flux_ycont_yface(i, j+1, k) - flux_ycont_yface(i, j, k))/(dx[1]);
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
