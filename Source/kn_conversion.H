#ifndef KERNEL_CONVERSION_H_
#define KERNEL_CONVERSION_H_

#include <AMReX_FArrayBox.H>
// ==================================== UTILITY | CONVERSION  ================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void cart2cont_x (int i, int j, int k,
                  amrex::Array4<amrex::Real> const& xcont,
                  amrex::Array4<amrex::Real const> const& vcart)
{
    using amrex::Real;
    xcont(i, j, k) = Real(0.5) * ( vcart(i-1, j, k, 0) + vcart(i, j, k, 0) ) ;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void cart2cont_y (int i, int j, int k,
                  amrex::Array4<amrex::Real> const& ycont,
                  amrex::Array4<amrex::Real const> const& vcart)
{
    using amrex::Real;
    ycont(i, j, k) = Real(0.5) * ( vcart(i, j-1, k, 1) + vcart(i, j, k, 1) ) ;
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void cart2cont_z (int i, int j, int k,
                 amrex::Array4<amrex::Real> const& zcont,
                 amrex::Array4<amrex::Real const> const& vcart)
{
    using amrex:Real;
    zcont(i, j, k) = Real(0.5) * ( vcart(i, j, k-1, 2) + vcart(i, j, k, 2) ) ;
}
#endif

#endif
