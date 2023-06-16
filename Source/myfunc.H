#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

void main_main ();

// void advance (amrex::MultiFab& phi_old,
//               amrex::MultiFab& phi_new,
//               amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
//               amrex::Real dt,
//               amrex::Geometry const& geom);

void init_userCTX (amrex::MultiFab& userCtx, amrex::Geometry const& geom);

void init_velocity (amrex::MultiFab& userCtx, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velocity, amrex::Geometry const& geom);

#endif
