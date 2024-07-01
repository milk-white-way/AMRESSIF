#include <AMReX_MultiFabUtil.H>

#include "fn_rhs.H"

using namespace amrex;

// ==================================== MODULE | MOMENTUM ====================================
void momentum_righthand_side_calc ( Array<MultiFab, AMREX_SPACEDIM>& rhs,
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

        //int const& box_id = mfi.LocalIndex();
        //print_box(box_id);

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            xrhs(i, j, k) = amrex::Real(0.5)*( total_flux(i-1, j, k, 0) + total_flux(i, j, k, 0) );

            // FIXME - check walls
            /*
            if (i == 0 || i == xbx.bigEnd(0)) {
                xrhs(i, j, k) = amrex::Real(0.0);
            }
            */

            //if (i == 1) {
            //    xrhs(i, j, k) = amrex::Real(0.125)*( 3*total_flux(i-1, j, k, 0) + 6*total_flux(i, j, k, 0) - total_flux(i+1, j, k, 0) );
            //} else {
            //    xrhs(i, j, k) = amrex::Real(0.125)*( 3*total_flux(i, j, k, 0) + 6*total_flux(i-1, j, k, 0) - total_flux(i-2, j, k, 0) );
            //}
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            yrhs(i, j, k) = amrex::Real(0.5)*( total_flux(i, j-1, k, 1) + total_flux(i, j, k, 1) );

            // FIXME - check walls
            /*
            if (j == 0 || j == ybx.bigEnd(1)) {
                yrhs(i, j, k) = amrex::Real(0.0);
            }
            */
            //if (j == 1) {
            //    yrhs(i, j, k) = amrex::Real(0.125)*( 3*total_flux(i, j-1, k, 1) + 6*total_flux(i, j, k, 1) - total_flux(i, j+1, k, 1) );
            //} else {
            //    yrhs(i, j, k) = amrex::Real(0.125)*( 3*total_flux(i, j, k, 1) + 6*total_flux(i, j-1, k, 1) - total_flux(i, j-2, k, 1) );
            //}
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            zrhs(i, j, k) = amrex::Real(0.5)*( total_flux(i, j, k-1, 2) + total_flux(i, j, k, 2) );
            // FIXME - check walls
            /*
            if (k == 0 || k == zbx.bigEnd(2)) {
                zrhs(i, j, k) = amrex::Real(0.0);
            }
            */
        });
#endif
    }
}

// ==================================== MODULE | POISSON ====================================
void poisson_righthand_side_calc ( MultiFab& poisson_rhs,
                                   Array<MultiFab, AMREX_SPACEDIM>& velCont,
                                   Geometry const& geom,
                                   Real const& dt )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Calculting the Divergence of V* << velImRK
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(poisson_rhs); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& rhs  = poisson_rhs.array(mfi);

        auto const& vel_cont_x = velCont[0].array(mfi);
        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_z = velCont[2].array(mfi);
#endif
        //Loop for all i,j,k in the local domain
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#if (AMREX_SPACEDIM > 2)
            //compute_flux_divergence_3D(i, j, k, vrhs, xcont, ycont, zcont, dx);
#else
            //compute_flux_divergence_2D(i, j, k, vrhs, xcont, ycont, dx);
            rhs(i, j, k) = ( vel_cont_x(i+1, j, k) - vel_cont_x(i, j, k) )/dx[0] + ( vel_cont_y(i, j+1, k) - vel_cont_y(i, j, k) )/dx[1];
#endif
            });
    } // End of all box loops

    // Scaling the right-hand side to include time-step here
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(poisson_rhs); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& rhs  = poisson_rhs.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            rhs(i, j, k) = (amrex::Real(1.5)/dt) * rhs(i, j, k);
        });
    }
}
