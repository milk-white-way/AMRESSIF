#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ================================= DECLARATION MAIN FUNCTION =================================
// void main_main ();

// ================================= MODULE | INITIALIZATION =================================
void init (amrex::MultiFab& userCtx,
           amrex::MultiFab& velCart,
           amrex::MultiFab& velCartDiff,
           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
           amrex::Geometry const& geom);

// ==================================== MODULE | MOMENTUM ====================================
// +++++++++++++++++++++++++ Subroutine | Volume Centered Fluxes  ++++++++++++++++++++++++
void convective_flux_calc (amrex::MultiFab& fluxConvect,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                           amrex::MultiFab& velCart,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                           amrex::Geometry const& geom,
                           int const& n_cell);

void viscous_flux_calc (amrex::MultiFab& fluxViscous,
                        amrex::MultiFab& velCart,
                        amrex::Geometry const& geom,
                        amrex::Real const& ren,
                        int const& n_cell);

void pressure_gradient_calc (amrex::MultiFab& fluxPrsGrad,
                             amrex::MultiFab& userCtx,
                             amrex::Geometry const& geom,
                             int const& n_cell);

// +++++++++++++++++++++++++ Subroutine | Kim and Moine's RHS Term  ++++++++++++++++++++++++
void righthand_side_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                          amrex::MultiFab& fluxConvect,
                          amrex::MultiFab& fluxViscous,
                          amrex::MultiFab& fluxPrsGrad,
                          amrex::MultiFab& fluxTotal,
                          int const& n_cell);

// +++++++++++++++++++++++++ Subroutine | Kim and Moine's Runge-Kutta ++++++++++++++++++++++++
void momentum_km_runge_kutta ( amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                               amrex::MultiFab& fluxTotal,
                               amrex::MultiFab& fluxConvect,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                               amrex::MultiFab& fluxViscous,
                               amrex::MultiFab& fluxPrsGrad,
                               amrex::MultiFab& userCtx,
                               amrex::MultiFab& velCart,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContPrev,
                               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
                               amrex::Vector<amrex::Real>& rk,
                               int const& RungeKuttaOrder,
                               int const& countIter,
                               amrex::Real const& normError,
                               amrex::Geometry const& geom,
                               amrex::Real const& ren,
                               amrex::Real const& dt,
                               int const& n_cell,
                               int const& IterNum,
                               amrex::Real const& Tol);

// ==================================== MODULE | POISSON =====================================
// Dathi's Module

// ==================================== MODULE | ADVANCE =====================================
// void advance (amrex::MultiFab& phi_old,
//               amrex::MultiFab& phi_new,
//               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
//               amrex::Real dt,
//               amrex::Geometry const& geom);


// ==================================== UTILITY | CONVERSION  ================================
void cont2cart (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom);

void cart2cont (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont);

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void manual_filling_ghost_cells (amrex::MultiFab& velCart,
                                int const& Nghost,
                                amrex::Vector<int> const& bc_lo,
                                amrex::Vector<int> const& bc_hi,
                                int const& n_cell);
#endif
