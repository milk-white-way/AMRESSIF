#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ==================================== UTILITY | CONVERSION  ================================
void cart2cont (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont);
