"""ABACUS atomic units at the ASE boundary.

The native values come from source/source_base/constants.h. ABACUS uses
them for printed Angstrom coordinates, eV energies, and eV/Angstrom forces;
they differ slightly from ASE's CODATA constants. Keep native/i-PI packets in
atomic units and translate only at this Python interface.
"""

from ase import units as ase_units

# ModuleBase constants in source/source_base/constants.h.
BOHR_TO_ANGSTROM = 0.5291770
RYDBERG_TO_EV = 13.605698
HARTREE_TO_EV = 2.0 * RYDBERG_TO_EV
HARTREE_TO_JOULE = 4.35974394e-18

# ASE IPIProtocol encodes Angstrom using ase.units.Bohr. Scaling a wire-only
# Atoms copy by this ratio makes the raw packets use ABACUS atomic units.
ASE_TO_ABACUS_LENGTH = ase_units.Bohr / BOHR_TO_ANGSTROM
ASE_TO_ABACUS_ENERGY = HARTREE_TO_EV / ase_units.Hartree
ASE_TO_ABACUS_FORCE = ASE_TO_ABACUS_ENERGY * ASE_TO_ABACUS_LENGTH

# output_log.cpp::print_stress uses RYDBERG_SI / BOHR_RADIUS_SI**3 * 1e-8
# to print kbar. Invert that conversion, then use the same eV/Angstrom units
# as energy and forces. The 1e-22 combines kbar -> Pa and m**3 -> Angstrom**3.
# A direct ASE GPa conversion would mix ABACUS's older joule and eV constants.
KBAR_TO_EV_PER_ANGSTROM3 = HARTREE_TO_EV / HARTREE_TO_JOULE * 1.0e-22
