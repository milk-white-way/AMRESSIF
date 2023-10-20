#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <iostream>
#include <fstream>

// ===== UTILITY | CONVERSION  =========//
void cart2cont (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont);

// ==== UTILITY | CONVERSION  ==========//
void cont2cart (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom);

void write_midline_solution (amrex::Real const& midx,
                             amrex::Real const& v_analytic,
                             double const& velx,
                             double const& vely,
                             int const& timestep);

void line_extract (amrex::MultiFab& velCart,
                   int const& n_cell,
                   int const& timestep);

// ===== MODULE | MOMENTUM ============//
void righthand_side_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                          amrex::MultiFab& fluxTotal);
