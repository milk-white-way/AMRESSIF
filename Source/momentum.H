//#ifndef MOMENTUM_H_
//#define MOMENTUM_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

amrex::Real Error_Computation( amrex::Array<MultiFab, AMREX_SPACEDIM>  &velImRK,
			       amrex::Array<MultiFab, AMREX_SPACEDIM>  &velImPrev,
			       amrex::Array<MultiFab, AMREX_SPACEDIM>  &velImDiff,
			       amrex::Geometry const &geom);


void momentum_km_runge_kutta (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                              amrex::MultiFab& fluxConvect,
                              amrex::MultiFab& fluxViscous,
                              amrex::MultiFab& fluxPrsGrad,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                              amrex::MultiFab& userCtx,
                              amrex::MultiFab& velCart,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
                              amrex::Real const& dt,
                              amrex::Geometry const& geom,
                              int const& n_cell,
                              amrex::Real const& ren);
