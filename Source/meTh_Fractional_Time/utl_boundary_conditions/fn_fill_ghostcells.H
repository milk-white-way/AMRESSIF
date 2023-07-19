#ifndef FUNCTION_FILL_GHOSTCELL_H_
#define FUNCTION_FILL_GHOSTCELL_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

void fill_physical_ghost_cells (amrex::MultiFab& velCart,
                                int const& Nghost,
                                int const& n_cell,
                                amrex::Vector<int> const& bc_lo,
                                amrex::Vector<int> const& bc_hi);
#endif
