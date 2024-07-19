#ifndef SUBROUTINE_INITIALIZATION_H_
#define SUBROUTINE_INITIALIZATION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ================================= MODULE | INITIALIZATION =================================
void init (amrex::MultiFab& userCtx,
           amrex::MultiFab& velCart,
           amrex::MultiFab& velCartDiff,
           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
           amrex::Geometry const& geom);

void staggered_grid_init (amrex::MultiFab& userCtx,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContPrev,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
                          amrex::MultiFab& velCart,
                          amrex::MultiFab& velCartPrev,
                          amrex::Geometry const& geom,
                          int const& Nghost,
                          amrex::Vector<int> const& phy_bc_lo,
                          amrex::Vector<int> const& phy_bc_hi,
                          amrex::Real const& time,
                          amrex::Real const& dt,
                          int const& n_cell);
#endif
