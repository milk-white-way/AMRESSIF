#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include <iostream>
#include <fstream>

// ===================== UTILITY | CONVERSION  =====================
void cart2cont (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont);

void cont2cart (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom);

// ===================== UTILITY | EXTRACT LINE SOLUTION  =====================
void write_midline_solution (amrex::Real const& midx,
                             amrex::Real const& midy,
                             amrex::Real const& mdlu,
                             amrex::Real const& mdlv,
                             amrex::Real const& mdlp,
                             double const& anau,
                             double const& anav,
                             double const& anap,
                             int const& current_step);

// ===================== UTILITY | EXPORT  =====================
void Export_Fluxes (amrex::MultiFab& fluxConvect,
                    amrex::MultiFab& fluxViscous,
                    amrex::MultiFab& fluxPrsGrad,
                    amrex::BoxArray const& ba,
                    amrex::DistributionMapping const& dm,
                    amrex::Geometry const& geom,
                    amrex::Real const& time,
                    int const& timestep);

void Export_Flow_Field (std::string const& nameofFile,
                        amrex::MultiFab& userCtx,
                        amrex::MultiFab& velCart,
                        amrex::BoxArray const& ba,
                        amrex::DistributionMapping const& dm,
                        amrex::Geometry const& geom,
                        amrex::Real const& time,
                        int const& timestep);

void analytic_solution_calc (amrex::MultiFab& analyticSol,
                             amrex::Geometry const& geom,
                             amrex::Real const& time);
