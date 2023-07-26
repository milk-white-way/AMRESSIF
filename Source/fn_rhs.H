#ifndef SUBROUTINE_RHS_H_
#define SUBROUTINE_RHS_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ==================================== MODULE | MOMENTUM ====================================
// +++++++++++++++++++++++++ Subroutine | Kim and Moine's RHS Term  ++++++++++++++++++++++++
void righthand_side_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                          amrex::MultiFab& fluxTotal);
#endif
