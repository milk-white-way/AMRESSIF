#include "main.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;


void Poisson_Solver (
	      amrex::MultiFab& phi_solution,
	      amrex::MultiFab& rhs_ptr,
              const Geometry& geom,
              const BoxArray& grids,
              const DistributionMapping& dmap,
              const Vector<BCRec>& bc);

void Poisson_RHS(amrex::Geometry const& geom,                       
		 amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
		 amrex::MultiFab& rhs,
		 amrex::Real& dt);

void Set_Phi_To_Zero(amrex::MultiFab& phi);

void Poisson_Update_Solution (
	      amrex::MultiFab& phi_solution,
	      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& grad_phi,
	      MultiFab& userCtx,
	      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
	      const Geometry& geom,
              const BoxArray& grids,
              const DistributionMapping& dmap,
              const Vector<BCRec>& bc);
