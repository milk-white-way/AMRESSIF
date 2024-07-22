/**
 * @file fn_init.cpp
 * @author milk-white-way (tam.thien.nguyen@ndsu.edu)
 * @brief 
 * @version 0.3
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <AMReX_MultiFabUtil.H>

#include "fn_init.H"
#include "fn_enforce_wall_bcs.H"
#include "kn_init.H"
#include "utilities.H"

using namespace amrex;
// ================================= MODULE | INITIALIZATION =================================

/**
 * @brief This function initializes the velocity components at face centers and the pressure components at cell centers.
 * 
 * @param userCtx 
 * @param velCont 
 * @param velContPrev 
 * @param velContDiff 
 * @param geom 
 */
void staggered_grid_init (MultiFab& userCtx,
                          Array<MultiFab, AMREX_SPACEDIM>& velCont,
                          Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
                          Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                          MultiFab& velCart,
                          MultiFab& velCartPrev,
                          Geometry const& geom,
                          int const& Nghost,
                          Vector<int> const& phy_bc_lo,
                          Vector<int> const& phy_bc_hi,
                          Real const& time,
                          Real const& dt,
                          int const& n_cell)

{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

// Initialize velocity components at face centers
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
        auto const& vel_cont_x = velCont[0].array(mfi);
        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_z = velCont[2].array(mfi);
#endif

        auto const& vel_cont_prev_x = velContPrev[0].array(mfi);
        auto const& vel_cont_prev_y = velContPrev[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_prev_z = velContPrev[2].array(mfi);
#endif

        auto const& vel_cont_diff_x = velContDiff[0].array(mfi);
        auto const& vel_cont_diff_y = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_diff_z = velContDiff[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
            
            //vel_cont_x(i, j, k) = std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y);
            //vel_cont_prev_x(i, j, k) = std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * (time - dt));
            vel_cont_x(i, j, k) = amrex::Real(0.0);
            vel_cont_prev_x(i, j, k) = amrex::Real(0.0);

            vel_cont_diff_x(i, j, k) = vel_cont_x(i, j, k) - vel_cont_prev_x(i, j, k);
        });
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];

            //vel_cont_y(i, j, k) = - std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y);
            //vel_cont_prev_y(i, j, k) = - std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * (time - dt));
            vel_cont_y(i, j, k) = amrex::Real(0.0);
            vel_cont_prev_y(i, j, k) = amrex::Real(0.0);

            vel_cont_diff_y(i, j, k) = vel_cont_y(i, j, k) - vel_cont_prev_y(i, j, k);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            vel_cont_z(i, j, k) = amrex::Real(1.0);

            vel_cont_prev_z(i, j, k) = amrex::Real(1.0);

            vel_cont_diff_z(i, j, k) = amrex::Real(0.0);
        });
#endif
    }

    /**
     * @brief After all contravariant velocity components are initialized, their cartesian velocity counterparts are interpolated.
     * 
     */
    cont2cart(velCart, velCont, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    //amrex::Print() << "=================================================================== \n";
    cont2cart(velCartPrev, velContPrev, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    //amrex::Print() << "=================================================================== \n";

// Initialize pressure components at celll centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, ctx, dx, prob_lo);
        });
    }
}

void init (MultiFab& userCtx,
           MultiFab& velCart,
           MultiFab& velCartDiff,
           Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
           Geometry const& geom)
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
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, ctx, dx, prob_lo);
        });
    }
    userCtx.FillBoundary(geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart = velCart.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity(i, j, k, vcart, dx, prob_lo);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCartDiff); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart_diff = velCartDiff.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity_difference(i, j, k, vcart_diff);
        });
    }

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
        auto const& xcont_diff = velContDiff[0].array(mfi);
        auto const& ycont_diff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zcont_diff = velContDiff[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            xcont_diff(i, j, k) = Real(0.0);
        });
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            ycont_diff(i, j, k) = Real(0.0);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            zcont_diff(i, j, k) = Real(0.0);
        });
#endif
    }
}
