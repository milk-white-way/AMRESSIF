#include <AMReX_MultiFabUtil.H>

#include "fn_init.H"
#include "kn_init.H"

using namespace amrex;
// ================================= MODULE | INITIALIZATION =================================
/**
 * ... This is the inititialization module ...
 */

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
