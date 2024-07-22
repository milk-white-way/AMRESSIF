#include <AMReX_MultiFabUtil.H>

#include "momentum.H"
#include "utilities.H"

using namespace amrex;

// ==================================== MODULE | ADVANCE =====================================
void runge_kutta4_pseudo_time_stepping (const GpuArray<Real,MAX_RK_ORDER>& rk,
                                        int const& sub,
                                        Array<MultiFab, AMREX_SPACEDIM>& momentum_rhs,
                                        Array<MultiFab, AMREX_SPACEDIM>& velStar,
                                        Array<MultiFab, AMREX_SPACEDIM>& velCont,
                                        Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                                        Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
                                        MultiFab& velCart,
                                        Geometry const& geom,
                                        int const& Nghost,
                                        Vector<int> const& phy_bc_lo,
                                        Vector<int> const& phy_bc_hi,
                                        int const& n_cell,
                                        Real const& dt)
{
    Box dom(geom.Domain());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(velCont[0]); mfi.isValid(); ++mfi)
    {
        const Box &xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box &ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box &zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& xrhs = momentum_rhs[0].array(mfi);
        auto const& yrhs = momentum_rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zrhs = momentum_rhs[2].array(mfi);
#endif

        auto const &vel_star_x = velStar[0].array(mfi);
        auto const &vel_star_y = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_star_z = velStar[2].array(mfi);
#endif

        auto const &vel_cont_x = velCont[0].array(mfi);
        auto const &vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_z = velCont[2].array(mfi);
#endif

        auto const &vel_cont_diff_x = velContDiff[0].array(mfi);
        auto const &vel_cont_diff_y = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_diff_z = velContDiff[2].array(mfi);
#endif

        auto const &vel_cont_prev_x = velContPrev[0].array(mfi);
        auto const &vel_cont_prev_y = velContPrev[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_prev_z = velContPrev[2].array(mfi);
#endif

        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0)+1;
        amrex::ParallelFor(xbx, 
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            xrhs(i, j, k) = xrhs(i, j, k) - ( Real(1.5)/dt )*( vel_star_x(i, j, k) - vel_cont_prev_x(i, j, k) ) + ( Real(0.5)/dt )*vel_cont_diff_x(i, j, k);
            if ( phy_bc_lo[0] != 0 || phy_bc_lo[0] != 0 || phy_bc_hi[0] != 0 || phy_bc_hi[0] != 0 ) {
                if ( i == lo || i == hi ) {
                    xrhs(i, j, k) = Real(0.0);
                } 
            }

            vel_star_x(i, j, k) = vel_cont_x(i, j, k) + ( rk[sub] * xrhs(i, j, k) );
        });
        
        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1)+1;
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            yrhs(i, j, k) = yrhs(i, j, k) - ( Real(1.5)/dt )*( vel_star_y(i, j, k) - vel_cont_prev_y(i, j, k) ) + ( Real(0.5)/dt )*vel_cont_diff_y(i, j, k);
            if ( phy_bc_lo[0] != 0 || phy_bc_lo[0] != 0 || phy_bc_hi[0] != 0 || phy_bc_hi[0] != 0 ) {
                if ( j == lo || j == hi ) {
                    xrhs(i, j, k) = Real(0.0);
                } 
            }
                        
            vel_star_y(i, j, k) = vel_cont_y(i, j, k) + ( rk[sub] * yrhs(i, j, k) );
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            zrhs(i, j, k) = zrhs(i, j, k) - ( Real(1.5)/dt )*( vel_star_z(i, j, k) - vel_cont_prev_z(i, j, k) ) + ( Real(0.5)/dt )*vel_cont_diff_z(i, j, k);

            vel_star_z(i, j, k) = vel_cont_z(i, j, k) + ( rk[sub] * zrhs(i, j, k) );
        });
#endif
    }
    // ------------------ CONVERT Ucont^{*,l} => Ucart^{*,l} ------------------
    //shift_face_to_center(velCart, velStar);
    //WriteSingleLevelPlotfile("pltIntermediateVel", velCart, {"u-star", "v-star"}, geom, 0, 0);
    cont2cart(velCart, velStar, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
}

void explicit_time_marching (Array<MultiFab, AMREX_SPACEDIM>& momentum_rhs,
                             Array<MultiFab, AMREX_SPACEDIM>& velStar,
                             Array<MultiFab, AMREX_SPACEDIM>& velCont,
                             Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                             Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
                             MultiFab& velCart,
                             Geometry const& geom,
                             int const& Nghost,
                             Vector<int> const& phy_bc_lo,
                             Vector<int> const& phy_bc_hi,
                             int const& n_cell,
                             Real const& dt)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(velCont[0]); mfi.isValid(); ++mfi)
    {
        const Box &xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box &ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box &zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& xrhs = momentum_rhs[0].array(mfi);
        auto const& yrhs = momentum_rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zrhs = momentum_rhs[2].array(mfi);
#endif

        auto const &vel_star_x = velStar[0].array(mfi);
        auto const &vel_star_y = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_star_z = velStar[2].array(mfi);
#endif

        auto const &vel_cont_x = velCont[0].array(mfi);
        auto const &vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_z = velCont[2].array(mfi);
#endif

        auto const &vel_cont_diff_x = velContDiff[0].array(mfi);
        auto const &vel_cont_diff_y = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_diff_z = velContDiff[2].array(mfi);
#endif

        auto const &vel_cont_prev_x = velContPrev[0].array(mfi);
        auto const &vel_cont_prev_y = velContPrev[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_prev_z = velContPrev[2].array(mfi);
#endif

        amrex::ParallelFor(xbx, 
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            xrhs(i, j, k) = xrhs(i, j, k) + ( Real(1.5)/dt )*vel_cont_prev_x(i, j, k) + ( Real(0.5)/dt )*vel_cont_diff_x(i, j, k);

            vel_star_x(i, j, k) = ( dt*xrhs(i, j, k) )/Real(1.5);
        });
        
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            yrhs(i, j, k) = yrhs(i, j, k) - ( Real(1.5)/dt )*vel_cont_prev_y(i, j, k) + ( Real(0.5)/dt )*vel_cont_diff_y(i, j, k);
                        
            vel_star_y(i, j, k) = ( dt*yrhs(i, j, k) )/Real(1.5);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            zrhs(i, j, k) = zrhs(i, j, k) - ( Real(1.5)/dt )*vel_cont_prev_z(i, j, k) + ( Real(0.5)/dt )*vel_cont_diff_z(i, j, k);
                        
            vel_star_z(i, j, k) = ( dt*zrhs(i, j, k) )/Real(1.5);
        });
#endif
    }
    // ------------------ CONVERT Ucont^{*,l} => Ucart^{*,l} ------------------
    cont2cart(velCart, velStar, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
}