#ifndef RHS_H_
#define RHS_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

void righthand_side_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                          amrex::MultiFab& fluxConvect,
                          amrex::MultiFab& fluxViscous,
                          amrex::MultiFab& fluxPrsGrad,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                          amrex::MultiFab& userCtx,
                          amrex::MultiFab& velCart,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                          amrex::Geometry const& geom,
                          int const& n_cell,
                          amrex::Real const& ren);
#endif
