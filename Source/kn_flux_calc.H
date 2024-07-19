#ifndef MY_FLUX_H_
#define MY_FLUX_H_

#include <AMReX_FArrayBox.H>

// ================================= KERNEL | FLUXES CALCULATION =================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_pressure_gradient_periodic (int i, int j, int k,
                                         amrex::Array4<amrex::Real> const& presgrad_flux,
                                         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                                         amrex::Array4<amrex::Real> const& pressurefield)
{
    using amrex::Real;

    presgrad_flux(i, j, k, 0) = (pressurefield(i+1, j, k, 0) - pressurefield(i-1, j, k, 0))/(Real(2.0)*dx[0]); // partial p / partial x
    presgrad_flux(i, j, k, 1) = (pressurefield(i, j+1, k, 0) - pressurefield(i, j-1, k, 0))/(Real(2.0)*dx[1]); // partial p / partial y
#if (AMREX_SPACEDIM > 2)
    presgrad_flux(i, j, k, 2) = (pressurefield(i, j, k+1, 0) - pressurefield(i, j, k-1, 0))/(Real(2.0)*dx[2]); // partial p / partial z
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_phi_gradient_periodic (int i, int j, int k,
                                    amrex::Array4<amrex::Real> const& grad_phi,
                                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                                    amrex::Array4<amrex::Real> const& phifield)
{
    using amrex::Real;

    grad_phi(i, j, k, 0) = (phifield(i+1, j, k, 1) - phifield(i-1, j, k, 1))/(Real(2.0)*dx[0]); // partial phi / partial x
    grad_phi(i, j, k, 1) = (phifield(i, j+1, k, 1) - phifield(i, j-1, k, 1))/(Real(2.0)*dx[1]); // partial phi / partial y
#if (AMREX_SPACEDIM > 2)
    grad_phi(i, j, k, 2) = (phifield(i, j, k+1, 1) - phifield(i, j, k-1, 1))/(Real(2.0)*dx[2]); // partial phi / partial z
#endif
}

#endif