//#ifndef MOMENTUM_H_
//#define MOMENTUM_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#define MAX_RK_ORDER 8

// ==================================== MODULE | MOMENTUM ====================================
// +++++++++++++++++++++++++ Subroutine | Kim and Moine's Runge-Kutta ++++++++++++++++++++++++
// ===================== MOMENTUM | ERROR NORM  =====================
amrex::Real Error_Computation (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velImRK,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velImPrev,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velImDiff,
                               amrex::Geometry const& geom);
/*
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
*/

// ==================================== MODULE | ADVANCE =====================================
void km_runge_kutta_advance ( const amrex::GpuArray<amrex::Real,MAX_RK_ORDER>& rk,
                              int const& sub,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velHat,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velHatDiff,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velStar,
                              amrex::Real const& dt,
                              amrex::Vector<int> const& phy_bc_lo,
                              amrex::Vector<int> const& phy_bc_hi,
                              int const& n_cell );
