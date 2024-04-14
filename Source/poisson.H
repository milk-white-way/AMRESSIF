#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

//-------------------------------------------------------
void poisson_advance (amrex::MultiFab& poisson_sol,
                      amrex::MultiFab& poisson_rhs,
                      amrex::Geometry const& geom,
                      amrex::BoxArray const& grids,
                      amrex::DistributionMapping const& dmap,
                      amrex::Vector<amrex::BCRec> const& bc);

void init_phi (amrex::MultiFab& phi);

void update_solution (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &grad_phi,
                      amrex::MultiFab &userCtx,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &velCont,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &velContPrev,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &velContDiff,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &velHat,
                      amrex::Geometry const &geom,
                      amrex::Real const &dt);
