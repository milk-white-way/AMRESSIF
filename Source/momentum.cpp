#include <AMReX_MultiFabUtil.H>

#include "momentum.H"

using namespace amrex;

// ==================================== MODULE | MOMENTUM ====================================
// +++++++++++++++++++++++++ Subroutine | Kim and Moine's Runge-Kutta ++++++++++++++++++++++++

// ==================================== MODULE | POISSON =====================================
// Dathi's Module

// ==================================== MODULE | ADVANCE =====================================
void km_runge_kutta_advance (const GpuArray<Real,MAX_RK_ORDER>& rk,
                             int const& sub,
                             Array<MultiFab, AMREX_SPACEDIM>& rhs,
                             Array<MultiFab, AMREX_SPACEDIM>& velHat,
                             Array<MultiFab, AMREX_SPACEDIM>& velHatDiff,
                             Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                             Array<MultiFab, AMREX_SPACEDIM>& velStar,
                             Real const& dt,
                             Vector<int> const& phy_bc_lo,
                             Vector<int> const& phy_bc_hi,
                             int const& n_cell)
{
    BL_PROFILE("advance");
    // =======================================================
    // This example supports both 2D and 3D.  Otherwise,
    // we would not need to use AMREX_D_TERM.
    // AMREX_D_TERM(const Real dxinv = geom.InvCellSize(0);,
    //              const Real dyinv = geom.InvCellSize(1);,
    //              const Real dzinv = geom.InvCellSize(2););
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velHat[0]); mfi.isValid(); ++mfi )
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
        auto const& xhat = velHat[0].array(mfi);
        auto const& yhat = velHat[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zhat = velHat[2].array(mfi);
#endif
        auto const& xhat_diff = velHatDiff[0].array(mfi);
        auto const& yhat_diff = velHatDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zhat_diff = velHatDiff[2].array(mfi);
#endif
        auto const& xcont_diff = velContDiff[0].array(mfi);
        auto const& ycont_diff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zcont_diff = velContDiff[2].array(mfi);
#endif
        auto const& xstar = velStar[0].array(mfi);
        auto const& ystar = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zstar = velStar[2].array(mfi);
#endif

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif
        // auto const& box_id = mfi.LocalIndex();
        // amrex::Print() << "DEBUGGING| in Box ID: " << box_id << "\n";

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){
            // amrex::Print() << "DEBUGGING| X-Runge-Kutta | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
            // Calculate F(\hat{u}_i^k)
            xrhs(i, j, k) = xrhs(i, j, k) - ( Real(1.5)/dt )*xhat_diff(i, j, k) + ( Real(0.5)/dt )*xcont_diff(i, j, k);
            // Update the immediate velocity
            xhat(i, j, k) = xstar(i, j, k) + ( rk[sub] * dt * Real(0.4) * xrhs(i,j,k) );
            // enforce boundary conditions on the fly
            if ( west_wall_bcs != 0 ) {
                if ( i==0 ) { xhat(i, j, k) = amrex::Real(0.0); }
            }
            if ( east_wall_bcs != 0 ) {
                if ( i==(n_cell) ) { xhat(i, j, k) = amrex::Real(0.0); }
            }
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k){
            // amrex::Print() << "DEBUGGING| Y-Runge-Kutta | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
            yrhs(i, j, k) = yrhs(i, j, k) - ( Real(1.5)/dt )*yhat_diff(i, j, k) + ( Real(0.5)/dt )*ycont_diff(i, j, k);
            yhat(i, j, k) = ystar(i, j, k) + ( rk[sub] * dt * Real(0.4) * yrhs(i,j,k) );
            if ( south_wall_bcs != 0 ) {
                if ( j==0 ) { yhat(i, j, k) = amrex::Real(0.0); }
            }
            if ( north_wall_bcs != 0 ) {
                if ( j==(n_cell) ) { yhat(i, j, k) = amrex::Real(0.0); }
            }
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k){
            zrhs(i, j, k) = zrhs(i, j, k) - ( Real(1.5)/dt )*zhat_diff(i, j, k) + ( Real(0.5)/dt )*zcont_diff(i, j, k);
            zhat(i, j, k) = zstar(i, j, k) + ( rk[sub] * dt * Real(0.4) * zrhs(i,j,k) );
            // enforce boundary conditions on the fly
            if ( front_wall_bcs != 0 ) {
                if ( k==0 ) { zhat(i, j, k) = amrex::Real(0.0); }
            }
            if ( back_wall_bcs != 0 ) {
                if ( k==(n_cell) ) { zhat(i, j, k) = amrex::Real(0.0); }
            }
        });
#endif
    }
}
