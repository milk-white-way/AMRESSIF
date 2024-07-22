#include <AMReX_MultiFabUtil.H>

#include "fn_rhs.H"

using namespace amrex;

// ==================================== MODULE | MOMENTUM ====================================
void momentum_righthand_side_calc ( MultiFab& fluxTotal,
                                    Array<MultiFab, AMREX_SPACEDIM>& array_grad_p,
                                    Array<MultiFab, AMREX_SPACEDIM>& rhs,
                                    Vector<int> const& phy_bc_lo,
                                    Vector<int> const& phy_bc_hi,
                                    const Geometry& geom )
{
    fluxTotal.FillBoundary(geom.periodicity());
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

        auto const& total_flux = fluxTotal.array(mfi);

        auto const& grad_p_x = array_grad_p[0].array(mfi);
        auto const& grad_p_y = array_grad_p[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& grad_p_z = array_grad_p[2].array(mfi);
#endif

        auto const& xrhs = rhs[0].array(mfi);
        auto const& yrhs = rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zrhs = rhs[2].array(mfi);
#endif

        //int const& box_id = mfi.LocalIndex();
        //print_box(box_id);
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            xrhs(i, j, k) = - grad_p_x(i, j, k) + amrex::Real(0.5)*( total_flux(i-1, j, k, 0) + total_flux(i, j, k, 0) );
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            yrhs(i, j, k) = - grad_p_y(i, j, k) + amrex::Real(0.5)*( total_flux(i, j-1, k, 1) + total_flux(i, j, k, 1) );
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            zrhs(i, j, k) = - grad_p_z(i, j, k) + amrex::Real(0.5)*( total_flux(i, j, k-1, 2) + total_flux(i, j, k, 2) );
        });
#endif
    }

    //shift_face_to_center(fluxTotal, rhs);
    //WriteSingleLevelPlotfile("pltMomentumRHS", fluxTotal, {"rhs-x", "rhs-y"}, geom, 0, 0);
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
        auto const& rhs = poisson_rhs.array(mfi);

        auto const& vel_cont_x = velCont[0].array(mfi);
        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_z = velCont[2].array(mfi);
#endif
        //Loop for all i,j,k in the local domain
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            rhs(i, j, k, 0) = ( vel_cont_x(i+1, j, k) - vel_cont_x(i, j, k) )/dx[0] + ( vel_cont_y(i, j+1, k) - vel_cont_y(i, j, k) )/dx[1]
#if (AMREX_SPACEDIM > 2)
                + ( vel_cont_z(i, j, k+1) - vel_cont_z(i, j, k) )/dx[2];
#else
            ;
#endif
            rhs(i, j, k, 0) = (Real(1.5)/dt) * rhs(i, j, k, 0);
        });
    } // End of all box loops
    Print() << "SOLVING| Poisson  | sum of components of RHS = " << poisson_rhs.sum(0) << '\n';
}
