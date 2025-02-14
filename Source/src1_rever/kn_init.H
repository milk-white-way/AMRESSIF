#ifndef KERNEL_INITIALIZATION_H_
#define KERNEL_INITIALIZATION_H_

#include <AMReX_FArrayBox.H>

// ================================= KERNEL | INITIALIZATION =================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_userCtx (int i, int j, int k,
                   amrex::Array4<amrex::Real> const& ctx,
                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM > 2)
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif
    // ctx[0] is pressure
    // ctx[1] is intermediate Phi
    //ctx(i, j, k, 0) = Real(0.25) * ( std::cos(Real(4.0) * M_PI * x) + std::cos(Real(4.0) * M_PI * y) );
    ctx(i, j, k, 0) = Real(0.0);
    ctx(i, j, k, 1) = Real(0.0);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_cartesian_velocity (int i, int j, int k,
                              amrex::Array4<amrex::Real> const& vcart,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM > 2)
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif
    // vcart is volume-centered Cartesian velocity
    vcart(i, j, k, 0) = std::sin(Real(2.0) * M_PI * x) * std::cos(Real(2.0) * M_PI * y);
    vcart(i, j, k, 1) = - std::cos(Real(2.0) * M_PI * x) * std::sin(Real(2.0) * M_PI * y);

#if (AMREX_SPACEDIM > 2)
    vcart(i, j, k, 2) = Real(0.0);
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_cartesian_velocity_difference (int i, int j, int k,
                                         amrex::Array4<amrex::Real> const& vcart_diff)
{
    using amrex::Real;

    // vcart_diff live in volume center
    vcart_diff(i, j, k, 0) = Real(0.0);
    vcart_diff(i, j, k, 1) = Real(0.0);
#if (AMREX_SPACEDIM > 2)
    vcart_diff(i ,j k, 2) = Real(0.0);
#endif
}

#endif
