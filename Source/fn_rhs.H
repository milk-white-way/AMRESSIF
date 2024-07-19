#ifndef RIGHHAND_SIDE_CALCULATION_H_
#define RIGHHAND_SIDE_CALCULATION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <utilities.H>

// ===== MODULE | MOMENTUM ============//
void momentum_righthand_side_calc (amrex::MultiFab& fluxTotal,
                                   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& array_grad_p,
                                   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& momentum_rhs,
                                   amrex::Vector<int> const& phy_bc_lo,
                                   amrex::Vector<int> const& phy_bc_hi,
                                   const amrex::Geometry& geom);

void poisson_righthand_side_calc (amrex::MultiFab& poisson_rhs,
                                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                                  amrex::Geometry const& geom,
                                  amrex::Real const& dt);

#endif
