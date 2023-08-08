#ifndef SUBROUNTINE_FLUX_CALCULATION_H_
#define SUBROUNTINE_FLUX_CALCULATION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ==================================== MODULE | MOMENTUM ====================================
// +++++++++++++++++++++++++ Subroutine | Volume Centered Fluxes  ++++++++++++++++++++++++
void convective_flux_calc (amrex::MultiFab& fluxConvect,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                           amrex::MultiFab& velCart,
                           amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                           amrex::Vector<int> const& phy_bc_lo,
                           amrex::Vector<int> const& phy_bc_hi,
                           amrex::Geometry const& geom,
                           int const& n_cell);

void viscous_flux_calc (amrex::MultiFab& fluxViscous,
                        amrex::MultiFab& velCart,
                        amrex::Geometry const& geom,
                        amrex::Real const& ren);

void pressure_gradient_calc (amrex::MultiFab& fluxPrsGrad,
                             amrex::MultiFab& userCtx,
                             amrex::Geometry const& geom);

void total_flux_calc (amrex::MultiFab& fluxTotal,
                      amrex::MultiFab& fluxConvect,
                      amrex::MultiFab& fluxViscous,
                      amrex::MultiFab& fluxPrsGrad);


#endif
