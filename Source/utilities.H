#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ===== UTILITY | CONVERSION  =========//
void cart2cont (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont);

// ==== UTILITY | CONVERSION  ==========//
void cont2cart (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom);

// ===== MODULE | MOMENTUM ============//
void righthand_side_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                          amrex::MultiFab& fluxTotal);