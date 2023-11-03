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
                             amrex::Real const& v_analytic,
                             double const& velx,
                             double const& vely,
                             int const& current_step);

void line_extract (amrex::MultiFab& velCart,
                   int const& n_cell,
                   int const& current_step,
                   amrex::Real const& dt,
                   amrex::Geometry const& geom);

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
