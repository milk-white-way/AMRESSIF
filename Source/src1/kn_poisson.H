#include <AMReX_FArrayBox.H>

//-------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_divergence_3D (int i, int j, int k,
                                 amrex::Array4<amrex::Real> const& vrhs,
                                 amrex::Array4<amrex::Real> const& ximrk,
                                 amrex::Array4<amrex::Real> const& yimrk,
                                 amrex::Array4<amrex::Real> const& zimrk,
                                 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx)
{

  // Two-dimensional case
     vrhs(i, j, 0) = (ximrk(i+1, j, k) - ximrk(i, j, k)) / dx[0] + (yimrk(i, j + 1, k) - yimrk(i, j, k))/dx[1];

  // 3-D case
 #if (AMREX_SPACEDIM > 2)
      vrhs(i, j, k) = (ximrk(i+1, j, k) - ximrk(i, j, k)) / dx[0] + (yimrk(i, j + 1, k) - yimrk(i, j, k)) / dx[1] + (zimrk(i, j, k+1) - zimrk(i, j, k)) / dx[2];
 #endif    
}
//----------------------------------------------------------
//-------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_divergence_2D (int i, int j, int k,
                                 amrex::Array4<amrex::Real> const& vrhs,
                                 amrex::Array4<amrex::Real> const& ximrk,
                                 amrex::Array4<amrex::Real> const& yimrk,
                                 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{

  // Two-dimensional case
   vrhs(i, j, 0) = (ximrk(i+1, j, k) - ximrk(i, j, k))/dx[0] + (yimrk(i, j+1, k) - yimrk(i, j, k))/dx[1];
    
}
