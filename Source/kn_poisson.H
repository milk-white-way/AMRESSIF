#include <AMReX_FArrayBox.H>

//-------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_divergence_3D (int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vrhs,
			 amrex::Array4<amrex::Real> const& xCont,
			 amrex::Array4<amrex::Real> const& yCont,
			 amrex::Array4<amrex::Real> const& zCont,
			 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{

  // Two-dimensional case
     vrhs(i, j, 0) = (xCont(i+1, j, k) - xCont(i, j, k)) / dx[0] + (yCont(i, j + 1, k) - yCont(i, j, k))/dx[1];  

  // 3-D case
 #if (AMREX_SPACEDIM > 2)
      vrhs(i, j, k) = (xCont(i+1, j, k) - xCont(i, j, k)) / dx[0] + (yCont(i, j + 1, k) - yCont(i, j, k)) / dx[1] + (zCont(i, j, k+1) - zCont(i, j, k)) / dx[2];  
 #endif    
}
//----------------------------------------------------------
//-------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_divergence_2D (int i, int j, int k,
                         amrex::Array4<amrex::Real> const& vrhs,
			 amrex::Array4<amrex::Real> const& xCont,
			 amrex::Array4<amrex::Real> const& yCont,
			 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{

  // Two-dimensional case
     vrhs(i, j, 0) = (xCont(i+1, j, k) - xCont(i, j, k)) / dx[0] + (yCont(i, j + 1, k) - yCont(i, j, k))/dx[1];  
    
}
