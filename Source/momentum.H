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

// ==================================== MODULE | ADVANCE =====================================
void runge_kutta4_pseudo_time_stepping (const amrex::GpuArray<amrex::Real,MAX_RK_ORDER>& rk,
                                        int const& sub,
                                        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                                        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velStar,
                                        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                                        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
                                        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContPrev,
                                        amrex::MultiFab& velCart,
                                        amrex::Geometry const& geom,
                                        int const& Nghost,
                                        amrex::Vector<int> const& phy_bc_lo,
                                        amrex::Vector<int> const& phy_bc_hi,
                                        int const& n_cell,
                                        amrex::Real const& dt);

void explicit_time_marching (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& rhs,
                             amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velStar,
                             amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                             amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
                             amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContPrev,
                             amrex::MultiFab& velCart,
                             amrex::Geometry const& geom,
                             int const& Nghost,
                             amrex::Vector<int> const& phy_bc_lo,
                             amrex::Vector<int> const& phy_bc_hi,
                             int const& n_cell,
                             amrex::Real const& dt);