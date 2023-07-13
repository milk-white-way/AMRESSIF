#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_init (int i, int j, int k,
                amrex::Array4<amrex::Real> const& ctx,
                amrex::Array4<amrex::Real> const& ucart,
                amrex::Array4<amrex::Real> const& udiff,
                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];

    ctx(i, j, k, 0) = - 0.25 * ( std::cos(Real(2.0) * M_PI * x) + std::cos(Real(2.0) * M_PI * y) );
    ctx(i, j, k, 1) = 0.0;

    ucart(i, j, k, 0) = - std::cos(Real(2.0) * M_PI * x) * std::sin(Real(2.0) * M_PI * y);
    ucart(i, j, k, 1) = std::sin(Real(2.0) * M_PI * x) * std::cos(Real(2.0) * M_PI * y);

    udiff(i, j, k, 0) = Real(0.0);
    udiff(i, j, k, 1) = Real(0.0);
#if (AMREX_SPACEDIM > 2)
    // Do 3D things
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    ucart(i, j, k, 0) = ucart(i, j, k, 0) * std::sin(Real(2.0) * M_PI * z);
    ucart(i, j, k, 1) = ucart(i, j, k, 1) * std::sin(Real(2.0) * M_PI * z);
    ucart(i, j, k, 2) = Real(0.0);
    udiff(i, j, k, 2) = Real(0.0);
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxx,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dxinv)
{
    // Pressure flux in x-axis
    fluxx(i,j,k) = (phi(i,j,k)-phi(i-1,j,k)) * dxinv;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxy,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dyinv)
{
    // Pressure flux in y-axis
    fluxy(i,j,k) = (phi(i,j,k)-phi(i,j-1,k)) * dyinv;
}


#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxz,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dzinv)
{
    fluxz(i,j,k) = (phi(i,j,k)-phi(i,j,k-1)) * dzinv;
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                              amrex::Array4<amrex::Real const> const& fluxy,
                              amrex::Array4<amrex::Real const> const& fluxz),
                 amrex::Real dt,
                 AMREX_D_DECL(amrex::Real dxinv,
                              amrex::Real dyinv,
                              amrex::Real dzinv))
{
    phinew(i,j,k) = phiold(i,j,k)
        + dt * dxinv * (fluxx(i+1,j  ,k  ) - fluxx(i,j,k))
        + dt * dyinv * (fluxy(i  ,j+1,k  ) - fluxy(i,j,k))
#if (AMREX_SPACEDIM > 2)
        + dt * dzinv * (fluxz(i  ,j  ,k+1) - fluxz(i,j,k));
#else
        ;
#endif
}

#endif
