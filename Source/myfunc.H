#ifndef MYFUNC_H_
#define MYFUNC_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>


// ==================================== MODULE | MOMENTUM ====================================
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
void km_runge_kutta_advance ( amrex::Vector<amrex::Real>& rk,
                              int const& sub,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velImRK,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
                              amrex::Real const& dt,
                              amrex::Vector<int> const& phy_bc_lo,
                              amrex::Vector<int> const& phy_bc_hi,
                              int const& n_cell );


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

void momentum_km_runge_kutta ( amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
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

void fill_physical_ghost_cells (amrex::MultiFab& velCart,
                                int const& Nghost,
                                int const& n_cell,
                                amrex::Vector<int> const& bc_lo,
                                amrex::Vector<int> const& bc_hi);



void advance (//amrex::MultiFab& phi_old,
              //amrex::MultiFab& phi_new,
	      amrex::MultiFab& phi_initial,
	      amrex::MultiFab& phi_exact,
	      amrex::MultiFab& rhs_ptr,
              //amrex::Real dt,
              const amrex::Geometry& geom,
              const amrex::BoxArray& grids,
              const amrex::DistributionMapping& dmap,
              const amrex::Vector<amrex::BCRec>& bc);

#endif
