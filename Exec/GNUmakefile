# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../../../amrex

DEBUG        = FALSE
USE_MPI      = FALSE
MPI_THREAD_MULTIPLE = FALSE  # amrex.async_out=1 with more than 64 processes requires MPI_THREAD_MULTIPLE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 2

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

Pdirs 	:= Base Boundary LinearSolvers/MLMG
Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
