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

void init (amrex::MultiFab& userCtx,
           amrex::MultiFab& velCart,
           amrex::MultiFab& velCartDiff,
           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
           amrex::Geometry const& geom);

void cont2cart (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom);

void cart2cont (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom);

void righthand_side_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                          amrex::MultiFab& fluxConvect,
                          amrex::MultiFab& fluxViscous,
                          amrex::MultiFab& fluxPrsGrad,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                          amrex::MultiFab& userCtx,
                          amrex::MultiFab& velCart,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                          amrex::Geometry const& geom,
                          int const& n_cell,
                          amrex::Real const& ren);

void fill_physical_ghost_cells (amrex::MultiFab& velCart,
                                int const& Nghost,
                                int const& n_cell,
                                amrex::Vector<int> const& bc_lo,
                                amrex::Vector<int> const& bc_hi);
#endif
