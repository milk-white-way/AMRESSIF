#ifndef RIGHHAND_SIDE_CALCULATION_H_
#define RIGHHAND_SIDE_CALCULATION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ===== MODULE | MOMENTUM ============//
void momentum_righthand_side_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& momentum_rhs,
                                   amrex::MultiFab& fluxTotal);

void poisson_righthand_side_calc (amrex::MultiFab& poisson_rhs,
                                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                                  amrex::Geometry const& geom,
                                  amrex::Real const& dt);

#endif
