#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

void main_main ();

// void advance (amrex::MultiFab& phi_old,
//               amrex::MultiFab& phi_new,
//               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
//               amrex::Real dt,
//               amrex::Geometry const& geom);

void init_userCTX (amrex::MultiFab& userCtx,
                   amrex::Geometry const& geom);

void init_velocity (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                    amrex::MultiFab& velCart,
                    amrex::Geometry const& geom);

void cartesian_velocity_interpolation (amrex::MultiFab& velCart,
                                       amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                                       amrex::MultiFab& velBcs,
                                       amrex::Geometry const& geom);

void viscous_flux_calc (amrex::MultiFab& fluxViscous,
                        amrex::MultiFab& velCart,
                        amrex::Geometry const& geom,
                        amrex::Real const& ren);

void fill_physical_ghost_cells (amrex::MultiFab& velCart,
                                int const& Nghost,
                                int const& n_cell,
                                amrex::Vector<int> const& bc_lo,
                                amrex::Vector<int> const& bc_hi);
#endif
