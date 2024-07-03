#include <AMReX_MultiFabUtil.H>

#include "momentum.H"

using namespace amrex;

// ==================================== MODULE | ADVANCE =====================================
void runge_kutta4_pseudo_time_stepping (const GpuArray<Real,MAX_RK_ORDER>& rk,
                                         int const& sub,
                                         Array<MultiFab, AMREX_SPACEDIM>& momentum_rhs,
                                         Array<MultiFab, AMREX_SPACEDIM>& velStar,
                                         Array<MultiFab, AMREX_SPACEDIM>& velHat,
                                         Array<MultiFab, AMREX_SPACEDIM>& velHatDiff,
                                         Array<MultiFab, AMREX_SPACEDIM>& velCont,
                                         Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                                         Real const& dt)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(velHat[0]); mfi.isValid(); ++mfi)
    {
        const Box &xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box &ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box &zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const &vel_cont_x = velCont[0].array(mfi);
        auto const &vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_z = velCont[2].array(mfi);
#endif

        auto const &vel_star_x = velStar[0].array(mfi);
        auto const &vel_star_y = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_star_z = velStar[2].array(mfi);
#endif

        auto const &vel_hat_x = velHat[0].array(mfi);
        auto const &vel_hat_y = velHat[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_hat_z = velHat[2].array(mfi);
#endif

        auto const &vel_hat_diff_x = velHatDiff[0].array(mfi);
        auto const &vel_hat_diff_y = velHatDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_hat_diff_z = velHatDiff[2].array(mfi);
#endif

        auto const &vel_cont_diff_x = velContDiff[0].array(mfi);
        auto const &vel_cont_diff_y = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_diff_z = velContDiff[2].array(mfi);
#endif

        auto const& xrhs = momentum_rhs[0].array(mfi);
        auto const& yrhs = momentum_rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zrhs = momentum_rhs[2].array(mfi);
#endif

        amrex::ParallelFor(xbx, 
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            xrhs(i, j, k) = xrhs(i, j, k) - ( Real(1.5)/dt )*vel_hat_diff_x(i, j, k) + ( Real(0.5)/dt )*vel_cont_diff_x(i, j, k);

            vel_hat_x(i, j, k) = vel_star_x(i, j, k) + ( rk[sub] * xrhs(i, j, k) );

            vel_hat_diff_x(i, j, k) = vel_hat_x(i, j, k) - vel_cont_x(i, j, k);
        });
        
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            yrhs(i, j, k) = yrhs(i, j, k) - ( Real(1.5)/dt )*vel_hat_diff_y(i, j, k) + ( Real(0.5)/dt )*vel_cont_diff_y(i, j, k);
                        
            vel_hat_y(i, j, k) = vel_star_y(i, j, k) + ( rk[sub] * yrhs(i, j, k) );

            vel_hat_diff_y(i, j, k) = vel_hat_y(i, j, k) - vel_cont_y(i, j, k);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            zrhs(i, j, k) = zrhs(i, j, k) - ( Real(1.5)/dt )*vel_hat_diff_z(i, j, k) + ( Real(0.5)/dt )*vel_cont_diff_z(i, j, k);
                        
            vel_hat_z(i, j, k) = vel_star_z(i, j, k) + ( rk[sub] * zrhs(i, j, k) );

            vel_hat_diff_z(i, j, k) = vel_hat_z(i, j, k) - vel_cont_z(i, j, k);
        });
#endif
    }
}
