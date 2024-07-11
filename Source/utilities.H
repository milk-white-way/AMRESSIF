#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "fn_enforce_wall_bcs.H"
#include <iostream>
#include <fstream>

// ===================== UTILITY | CONVERSION  =====================
void cont2cart (amrex::MultiFab& velCart,
                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
                amrex::Geometry const& geom,
                int const& Nghost,
                amrex::Vector<int> const& phy_bc_lo,
                amrex::Vector<int> const& phy_bc_hi,
                int const& n_cell);

void shift_face_to_center(amrex::MultiFab& cc_analytical_diff,
                          amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont );

// ===================== UTILITY | EXTRACT LINE SOLUTION  =====================
void write_interp_line_solution (amrex::Real const& interp_sol,
                                 std::string const& filename);

void write_exact_line_solution (amrex::Real const& x,
                                amrex::Real const& y,
                                amrex::Real const& num_sol,
                                amrex::Real const& ana_sol,
                                std::string const& filename);

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

void array_analytical_vel_calc (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& analytical_vel,
                                amrex::Geometry const& geom,
                                amrex::Real const& time);

void cc_analytical_calc (amrex::MultiFab& analytical_sol,
                         amrex::Geometry const& geom,
                         amrex::Real const& time);

void SumAbsStag(const std::array<amrex::MultiFab, AMREX_SPACEDIM>& m1,
                amrex::Vector<amrex::Real>& sum);

void StagL2Norm(const std::array<amrex::MultiFab, AMREX_SPACEDIM>& m1,
		const int& comp,
                amrex::Vector<amrex::Real>& inner_prod);

void StagInnerProd(const std::array<amrex::MultiFab, AMREX_SPACEDIM>& m1,
                   const int& comp1,
                   std::array<amrex::MultiFab, AMREX_SPACEDIM>& mscr,
                   amrex::Vector<amrex::Real>& prod_val);
