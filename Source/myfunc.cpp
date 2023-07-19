#include <AMReX_MultiFabUtil.H>
// #include <AMReX_Utility.H>
// #include <AMReX_PlotFileUtil.H>
// #include <AMReX_Print.H>

#include "myfunc.H"
#include "mykernel.H"

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
        auto const& vcart = velCart.array(mfi);
        auto const& vcart_diff = velCartDiff.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, ctx, dx, prob_lo);
            init_cartesian_velocity(i, j, k, vcart, dx, prob_lo);
            init_cartesian_velocity_difference(i, j, k, vcart_diff);
        });
    }

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
        auto const& fluxx_xcont = fluxHalfN1[2].array(mfi);
        auto const& fluxy_ycont = fluxHalfN2[2].array(mfi);
        auto const& fluxz_zcont = fluxHalfN3[2].array(mfi);
#endif
        auto const& vcart = velCart.array(mfi);

        int const& box_id = mfi.LocalIndex();
        //int const& box_id = mfi.LocalTileIndex();
        print_box(box_id);

        amrex::Print() << "===================== x-contributive terms ==================== \n";
        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_half_node_convective_flux_x_contrib(i, j, k, fluxx_xcont, fluxy_xcont, fluxz_xcont, xcont, vcart, qkcoef, n_cell, box_id);
        });

        amrex::Print() << "===================== y-contributive terms ==================== \n";
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_half_node_convective_flux_y_contrib(i, j, k, fluxx_ycont, fluxy_ycont, fluxz_ycont, ycont, vcart, qkcoef, n_cell, box_id);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::Print() << "===================== z-contributive terms ==================== \n";
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_half_node_convective_flux_z_contrib(i, j, k, fluxx_zcont, fluxy_zcont, fluxz_zcont, zcont, vcart, qkcoef, n_cell, box_id);
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
        auto const& fluxx_xcont = fluxHalfN1[2].array(mfi);
        auto const& fluxy_ycont = fluxHalfN2[2].array(mfi);
        auto const& fluxz_zcont = fluxHalfN3[2].array(mfi);
#endif
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if ( i >= 0 && i <= (n_cell-1) ) {
                if ( j >= 0 && j <= (n_cell-1) ) {
                    if (k >= 0 && k <= (n_cell-1) ) {
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

                    }
                }
            } else {
                for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                {
                    conv_flux(i, j, k, dir) = Real(0.0);
                }
            }
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

                visc_flux(i, j, k, dir) = ((westMAC - 2*centerMAC + eastMAC)/(dx[0]*dx[0]) + (southMAC - 2*centerMAC + northMAC)/(dx[1]*dx[1]))/ren;
            }
        });
    }
}

void pressure_gradient_calc ( MultiFab& fluxPrsGrad,
                              MultiFab& userCtx,
                              Geometry const& geom)
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

void righthand_side_calc ( Array<MultiFab, AMREX_SPACEDIM>& rhs,
                           MultiFab& fluxConvect,
                           MultiFab& fluxViscous,
                           MultiFab& fluxPrsGrad,
                           MultiFab& fluxTotal,
                           int const& n_cell )
{
    // +++++++++++++++++++++++++ Total Flux and the RHS  +++++++++++++++++++++++++
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
            for ( int dir=0; dir < AMREX_SPACEDIM; ++dir )
            {
                total_flux(i, j, k, dir) = conv_flux(i, j, k, dir) + visc_flux(i, j, k, dir) + prsgrad_flux(i, j, k, dir);
            }
        });
    }

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
                xrhs(i, j, k) = Real(0.5)*( total_flux((i)-1, j, k, 0) + total_flux(i, j, k, 0) );
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
                yrhs(i, j, k) = Real(0.5)*( total_flux(i, (j)-1, k, 1) + total_flux(i, j, k, 1) );
            }
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if ( k==0 ) {
                zrhs(i, j, k) = Real(0.0);
            }
            else if ( k==(n_cell-1) ) {
                zrhs(i, j, k) = Real(0.0);
            }
            else {
                zrhs(i, j, k) = Real(0.5)*( total_flux(i, j, (k)-1, 2) + total_flux(i, j, k, 2) );
            }
        });
#endif
    }
}

// +++++++++++++++++++++++++ Subroutine | Kim and Moine's Runge-Kutta ++++++++++++++++++++++++
/*
void momentum_km_runge_kutta ( Array<MultiFab, AMREX_SPACEDIM>& rhs,
                               MultiFab& fluxTotal,
                               MultiFab& fluxConvect,
                               Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                               Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                               Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                               MultiFab& fluxViscous,
                               MultiFab& fluxPrsGrad,
                               MultiFab& userCtx,
                               MultiFab& velCart,
                               Array<MultiFab, AMREX_SPACEDIM>& velCont,
                               Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
                               Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                               Vector<Real>& rk,
                               int const& RungeKuttaOrder,
                               int const& countIter,
                               Real const& normError,
                               Geometry& geom,
                               Real const& ren,
                               Real const& dt,
                               int const& n_cell,
                               int const& IterNum,
                               Real const& Tol)
{
    while ( countIter < IterNum && normError > Tol )
    {
        countIter++;
        //amrex::Print() << "SOLVING| Momentum | performing Runge-Kutta at iteration: " << countIter << "\n";

        for (int sub = 0; sub < RungeKuttaOrder; ++sub )
        {
            convective_flux_calc(fluxConvect, fluxHalfN1, fluxHalfN2, fluxHalfN3, velCart, velCont, geom, n_cell);
            viscous_flux_calc(fluxViscous, velCart, geom, ren);
            pressure_gradient_calc(fluxPrsGrad, userCtx, geom);
            righthand_side_calc(rhs, fluxConvect, fluxViscous, fluxPrsGrad, fluxTotal, n_cell);

            // Update new contravariant velocities
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
                auto const& xdiff = velContDiff[0].array(mfi);
                auto const& ydiff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& zdiff = velContDiff[2].array(mfi);
#endif
                auto const& xrhs = rhs[0].array(mfi);
                auto const& yrhs = rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
                auto const& zrhs = rhs[2].array(mfi);
#endif

                amrex::ParallelFor(xbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    // Itermidiate velocity
                    Real xhat = xcont(i, j, k);
                    // Corection for right-hand-side term
                     xrhs(i, j, k) = xrhs(i, j, k) - (Real(0.5)/dt)*(xhat - xrhs(i, j, k)) + (Real(0.5)/dt)*(xdiff(i, j, k));
                    // RK4 substep to update the immediate velocity
                    if ( i==0 || i==(n_cell-1) ) {
                        xhat = amrex::Real(0.0);
                    } else {
                        xhat = xcont(i, j, k) + rk[sub]*dt*xrhs(i,j,k);
                    }
                    xprev(i, j, k) = xcont(i, j, k);
                    xcont(i, j, k) = xhat;
                });

                amrex::ParallelFor(ybx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k){
                     Real yhat = ycont(i, j, k);

                    yrhs(i, j, k) = yrhs(i, j, k) - (Real(0.5)/dt)*(yhat - yrhs(i, j, k)) + (Real(0.5)/dt)*(ydiff(i, j, k));

                    if ( j==0 || j==(n_cell-1) ) {
                        yhat = amrex::Real(0.0);
                    } else {
                        yhat = ycont(i, j, k) + rk[sub]*dt*yrhs(i,j,k);
                    }
                    yprev(i, j, k) = ycont(i, j, k);
                    ycont(i, j, k) = yhat;
                });

#if (AMREX_SPACEDIM > 2)
                amrex::ParallelFor(zbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k){
                    Real zhat = zcont(i, j, k);

                    zrhs(i, j, k) = zrhs(i, j, k) - (Real(0.5)/dt)*(zhat - zrhs(i, j, k)) + (Real(0.5)/dt)*(zdiff(i, j, k));

                    if ( k==0 || k==(n_cell-1) ) {
                        zhat = amrex::Real(0.0);
                    } else {
                        zhat = zcont(i, j, k) + rk[sub]*dt*zrhs(i,j,k);
                    }
                    zprev(i, j, k) = zcont(i, j, k);
                    zcont(i, j, k) = zhat;
                });
#endif
            }
        }
        // Update contravelocity difference
        for ( MFIter mfi(velContDiff[0]); mfi.isValid(); ++mfi )
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
            auto const& xdiff = velContDiff[0].array(mfi);
            auto const& ydiff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
            auto const& zdiff = velContDiff[2].array(mfi);
#endif
            amrex::ParallelFor(xbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                xdiff(i, j, k) = xcont(i, j, k) - xprev(i, j, k);
            });

            amrex::ParallelFor(ybx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k){
                ydiff(i, j, k) = ycont(i, j, k) - yprev(i, j, k);
            });

#if (AMREX_SPACEDIM > 2)
            amrex::ParallelFor(zbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k){
                zdiff(i, j, k) = zcont(i, j, k) - zprev(i, j, k);
            });
#endif
        }
        // Update error stopping condition
        for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
        {
            const Box& vbx = mfi.validbox();

            auto const& xdiff = velContDiff[0].array(mfi);
            auto const& ydiff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
            auto const& zdiff = velContDiff[2].array(mfi);
#endif
            auto const& norm_error = normError;

            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if ( i > 0 && j > 0 && k > 0 ) {
                    Real square_L2_norm = xdiff(i, j, k)*xdiff(i, j, k) + ydiff(i,j,k)*ydiff(i,j,k)
#if (AMREX_SPACEDIM > 2)
                        + zdiff(i, j, k)*zcont(i, j, k) ;
#else
                    ;
#endif
                }

                Real l2_norm = std::sqrt(square_L2_norm);

                if ( norm_error > l2_norm ) {
                    norm_error = l2_norm;
                }
            });
        }
    }
}
*/

// ==================================== MODULE | POISSON =====================================
// Dathi's Module

// ==================================== MODULE | ADVANCE =====================================
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


// ==================================== UTILITY | CONVERSION  ================================
void cont2cart (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom)
{
    average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);
}

void cart2cont (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    {

        Box xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        Box ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
        auto const& xcont = velCont[0].array(mfi);
        auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        Box zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
        auto const& zcont = velCont[2].array(mfi);
#endif
        auto const& vcart = velCart.array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            xcont(i, j, k) = Real(0.5) * (vcart(i-1, j, k, 0) + vcart(i, j, k, 0)) ;
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            ycont(i, j, k) = Real(0.5) * (vcart(i, j-1, k, 1) + vcart(i, j, k, 1)) ;
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            zcont(i, j, k) = Real(0.5) * (vcart(i, j, k-1, 2) + vcart(i, j, k, 2)) ;
        });
#endif
    }
}

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void manual_filling_ghost_cells (amrex::MultiFab& velCart,
                                int const& Nghost,
                                amrex::Vector<int> const& phy_bc_lo,
                                amrex::Vector<int> const& phy_bc_hi,
                                int const& n_cell)
{
    for (MFIter mfi(velCart); mfi.isValid(); ++mfi)
    {
        //amrex::Print() << "GHOST FILL MANUALLY| at box: " << mfi << "\n";
        amrex::Print() << "====================== Entering New Box  ====================== \n";

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif
        const Box& gbx = mfi.growntilebox(Nghost);
        auto const& vcart = velCart.array(mfi);

        if ( west_wall_bcs != 0 ) {
            amrex::Print() << "========================== west wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_west(i, j, k, n_cell, west_wall_bcs, vcart);
            });
        }

        if ( east_wall_bcs != 0 ) {
            amrex::Print() << "========================== east wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_east(i, j, k, n_cell, west_wall_bcs, vcart);
            });
        }

        if ( south_wall_bcs != 0 ) {
            amrex::Print() << "========================= south wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_south(i, j, k, n_cell, south_wall_bcs, vcart);
            });
        }

        if ( north_wall_bcs != 0 ) {
            amrex::Print() << "========================= north wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_north(i, j, k, n_cell, north_wall_bcs, vcart);
            });
        }
#if (AMREX_SPACEDIM > 2)
        if ( front_wall_bcs != 0 ) {
            amrex::Print() << "========================= front wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_front(i, j, k, n_cell, front_wall_bcs, vcart);
            });
        }

        if ( back_wall_bcs != 0 ) {
            amrex::Print() << "========================== back wall  ========================= \n";

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                filling_ghost_back(i, j, k, n_cell, back_wall_bcs, vcart);
            });
        }
#endif
    }
}
