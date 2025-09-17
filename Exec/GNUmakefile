# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../../../amrex

DEBUG        = FALSE
USE_MPI      = TRUE
MPI_THREAD_MULTIPLE = FALSE  # amrex.async_out=1 with more than 64 processes requires MPI_THREAD_MULTIPLE
USE_OMP      = FALSE
COMP         = llvm
DIM          = 3

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

#include ../Source/src1/Make.package
#VPATH_LOCATIONS  += ../Source/src1
#INCLUDE_LOCATIONS += ../Source/src1

include ../Source/src1_rever/Make.package
VPATH_LOCATIONS  += ../Source/src1_rever
INCLUDE_LOCATIONS += ../Source/src1_rever

Pdirs 	:= Base Boundary LinearSolvers
Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
