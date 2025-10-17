#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>

#ifdef AMREX_USE_HYPRE
#include <AMReX_Hypre.H>
#endif

#include <AMReX_MultiFabUtil.H>

#include <fn_flux_calc.H>
#include <fn_enforce_wall_bcs.H>
#include <utilities.H>

//-------------------------------------------------------
void poisson_advance (amrex::MultiFab& poisson_sol,
                      amrex::MultiFab& poisson_rhs,
                      amrex::Geometry const& geom,
                      amrex::BoxArray const& grids,
                      amrex::DistributionMapping const& dmap,
                      amrex::Vector<amrex::BCRec> const& bc);

void init_phi (amrex::MultiFab& phi);

void update_solution (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& array_grad_p,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& array_grad_phi,
                      amrex::MultiFab& fluxPrsGrad,
                      amrex::MultiFab& cc_grad_phi,
                      amrex::MultiFab& poisson_rhs,
                      amrex::MultiFab& userCtx,
                      amrex::MultiFab& velCart,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velStar,
                      amrex::Geometry const& geom,
					  amrex::Real& time, 
                      amrex::Real const& dt,
                      int const& Nghost,
                      amrex::Vector<int> const& phy_bc_lo,
                      amrex::Vector<int> const& phy_bc_hi,
                      amrex::Vector<amrex::Real> const& inflow_waveform,
                      int const& n_cell,
                      amrex::Real const& ren);
