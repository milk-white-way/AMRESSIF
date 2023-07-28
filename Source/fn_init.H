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

#endif
