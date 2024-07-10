#ifndef SUBROUTINE_ENFORCING_BCS_H_
#define SUBROUTINE_ENFORCING_BCS_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void enforce_wall_bcs_for_cell_centered_flux_on_ghost_cells (amrex::MultiFab& flux,
                                                             amrex::Geometry const& geom,
                                                             int const& Nghost,
                                                             amrex::Vector<int> const& phy_bc_lo,
                                                             amrex::Vector<int> const& phy_bc_hi,
                                                             int const& n_cell);
/**
 * @brief The function is the second step after AMReX "FillBoundary" function fill all the interior communication cells. It enforces the wall boundary conditions on the ghost cells. 
 * 
 * @param velCart 
 * @param geom 
 * @param Nghost 
 * @param phy_bc_lo 
 * @param phy_bc_hi 
 * @param n_cell 
 */
void enforce_wall_bcs_for_cell_centered_velocity_on_ghost_cells (amrex::MultiFab& velCart,
                                                                 amrex::Geometry const& geom,
                                                                 int const& Nghost,
                                                                 amrex::Vector<int> const& phy_bc_lo,
                                                                 amrex::Vector<int> const& phy_bc_hi,
                                                                 int const& n_cell);


void enforce_wall_bcs_for_cell_centered_userCtx_on_ghost_cells (amrex::MultiFab& userCtx,
                                                                amrex::Geometry const& geom,
                                                                int const& Nghost,
                                                                amrex::Vector<int> const& phy_bc_lo,
                                                                amrex::Vector<int> const& phy_bc_hi,
                                                                int const& n_cell);

/**
 * @brief The function is the final step in the process of enforcing wall boundary conditions. It interpolates the components of contravariant velocities on the physical boundaries from the two adjacent cell centered components (one is inside the physical domain and one is in the ghost domain).
 * 
 * @param velCart 
 * @param velCont 
 * @param geom 
 */
void enforce_wall_bcs_for_face_centered_velocity_on_physical_boundaries (amrex::MultiFab& velCart,
                                                                         amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                                                                         amrex::Geometry const& geom,
                                                                         amrex::Vector<int> const& phy_bc_lo,
                                                                         amrex::Vector<int> const& phy_bc_hi);
#endif
