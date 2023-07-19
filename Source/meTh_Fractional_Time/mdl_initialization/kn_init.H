#ifndef KERNEL_INITIALIZATION_H_
#define KERNEL_INITIALIZATION_H_

#include <AMReX_FArrayBox.H>

void init_userCtx (int i, int j, int k,
                   amrex::Array4<amrex::Real> const& ctx,
                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo);

void init_cartesian_velocity (int i, int j, int k,
                              amrex::Array4<amrex::Real> const& vcart,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo);

void init_cartisian_velocity_difference (int i, int j, int k,
                                         amrex::Array4<amrex::Real> const& vcart_diff);

void init_contravariant_velocity_difference (int i, int j, int k,
                                         amrex::Array4<amrex::Real> const& cf_cont_diff);

#endif
