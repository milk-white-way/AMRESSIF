#ifndef SUBROUTINE_ENFORCING_BCS_H_
#define SUBROUTINE_ENFORCING_BCS_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void enforce_boundary_conditions (amrex::MultiFab& inputMultiFab,
                                  std::string const& typeMultiFab,
                                  int const& Nghost,
                                  amrex::Vector<int> const& phy_bc_lo,
                                  amrex::Vector<int> const& phy_bc_hi,
                                  int const& n_cell);
#endif
