#include "myfunc.H"
//#include "momentum.H"

#include <AMReX_MultiFabUtil.H>

using namespace amrex;

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
    // Setup stopping conditions
    Real normError = 1.0e8;
    int countIter = 0;
    int IterNumCycle = 50;
    // Setup Runge-Kutta intermediate coefficients
    int RungeKuttaOrder = 4;
    Vector<Real> rk(RungeKuttaOrder, 0);
    {
        rk[0] = Real(0.25);
        rk[1] = Real(1.0)/Real(3.0);
        rk[2] = Real(0.5);
        rk[4] = Real(1.0);
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
                    // Immediate velocity
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
                    Real yhat = ycont(i, k, k);
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
