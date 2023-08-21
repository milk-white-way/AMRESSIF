#include <AMReX_MultiFabUtil.H>

#include "myfunc.H"
#include "mykernel.H"

using namespace amrex;

// ==================================== MODULE | MOMENTUM ====================================
// +++++++++++++++++++++++++ Subroutine | Kim and Moine's Runge-Kutta ++++++++++++++++++++++++

// ==================================== MODULE | POISSON =====================================
// Dathi's Module

// ==================================== MODULE | ADVANCE =====================================
void km_runge_kutta_advance (const GpuArray<Real,MAX_RK_ORDER>& rk,
                             int const& sub,
                             Array<MultiFab, AMREX_SPACEDIM>& rhs,
                             Array<MultiFab, AMREX_SPACEDIM>& velImRK,
                             Array<MultiFab, AMREX_SPACEDIM>& velCont,
                             Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
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
    for ( MFIter mfi(velContDiff[0]); mfi.isValid(); ++mfi )
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
        auto const& ximrk = velImRK[0].array(mfi);
        auto const& yimrk = velImRK[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zimrk = velImRK[2].array(mfi);
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
            // Corection for right-hand-side term
            xrhs(i, j, k) = xrhs(i, j, k) - ( Real(1.5)/dt )*( ximrk(i, j, k) - xcont(i, j, k) ) + ( Real(0.5)/dt )*xdiff(i, j, k);
            // RK4 substep to update the immediate velocity
            ximrk(i, j, k) = xcont(i, j, k) + ( rk[sub] * dt * Real(0.4) * xrhs(i,j,k) );
            // enforce boundary conditions on the fly
            if ( west_wall_bcs != 0 ) {
                if ( i==0 ) { ximrk(i, j, k) = amrex::Real(0.0); }
            }
            if ( east_wall_bcs != 0 ) {
                if ( i==(n_cell) ) { ximrk(i, j, k) = amrex::Real(0.0); }
            }
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k){
            // amrex::Print() << "DEBUGGING| Y-Runge-Kutta | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
            yrhs(i, j, k) = yrhs(i, j, k) - ( Real(1.5)/dt )*( yimrk(i, j, k) - ycont(i, j, k) ) + ( Real(0.5)/dt )*ydiff(i, j, k);

            yimrk(i, j, k) = ycont(i, j, k) + ( rk[sub] * dt * Real(0.4) * yrhs(i,j,k) );
            // enforce boundary conditions on the fly
            if ( south_wall_bcs != 0 ) {
                if ( j==0 ) { yimrk(i, j, k) = amrex::Real(0.0); }
            }
            if ( north_wall_bcs != 0 ) {
                if ( j==(n_cell) ) { yimrk(i, j, k) = amrex::Real(0.0); }
            }
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k){
            zrhs(i, j, k) = zrhs(i, j, k) - ( Real(1.5)/dt )*( zimrk(i, j, k) - zcont(i, j, k) ) + ( Real(0.5)/dt )*zdiff(i, j, k);

            zimrk(i, j, k) = zcont(i, j, k) + ( rk[sub] * dt * Real(0.4) * zrhs(i,j,k) );
            // enforce boundary conditions on the fly
            if ( front_wall_bcs != 0 ) {
                if ( k==0 ) { zimrk(i, j, k) = amrex::Real(0.0); }
            }
            if ( back_wall_bcs != 0 ) {
                if ( k==(n_cell) ) { zimrk(i, j, k) = amrex::Real(0.0); }
            }
        });
#endif
    }
}
