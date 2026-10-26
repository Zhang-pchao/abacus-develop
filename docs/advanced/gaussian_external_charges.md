# Neutral periodic Gaussian external charges

This experimental CPU PW PBE interface represents a fixed environment by
physical Gaussian charges in a fixed orthorhombic 3D cell. It is independent of
the existing uniform electric field and isolated-cell correction interfaces.
It does not implement QM/DPA subtraction inside ABACUS; the external coupling
owner supplies its low-level and reference terms separately.

Set `gaussian_charge_file` to an immutable file and `gaussian_charge_kmax` to
an explicitly converged reciprocal cube half-width. The file has this format:

```text
QMDPA_GAUSSIAN_FIELD_V1
2
11  0.4  1.0  3.0  1.0  2.0
12 -0.4  1.0  8.0  7.0  6.0
```

Each row is a unique positive integer identity, charge in e, Gaussian standard
deviation in Bohr, and Cartesian x/y/z in Bohr. Width is a physical field
parameter. The environment must sum to zero within 1e-12 e, and the QM electron
count must match its pseudopotential valence charge. No neutralizing background
is silently introduced. Empty environment and all-zero charges are valid
zero-field controls.

Use CPU, basis_type pw, calculation scf, nspin 1, precision double,
smearing_method fixed, dft_functional pbe, symmetry -1, gamma_only false and
norm-conserving pseudopotentials. Stress, other external fields, implicit
solvent, isolated-cell corrections and modified functionals are rejected.
Fixed occupations exclude a general metallic-material qualification.

For every signed nonzero reciprocal vector in the complete integer cube,
B(G)=sum_j q_j exp(-sigma_j^2 G^2/2) exp(-i G.R_j). In atomic units,
phi(r)=4*pi/V sum_G B(G)exp(iG.r)/G^2. The electron potential is -phi;
the complete cross energy is sum_I Zval_I phi(R_I)-integral n(r)phi(r)dr.
Every signed mode appears once. Density cutoff and FFT grid must contain the
entire cube; missing modes fail closed. Converge K independently of SCF and
plane-wave convergence. The fixed finite cube defines the tested field.

ABACUS retains its periodic internal QM Coulomb/Ewald/self conventions. The
external term owns QM-environment and all their images, with no half factor.
Environment-environment and Gaussian self energy are excluded. G=0 is zero,
with a 3D tin-foil mean-zero gauge, no slab/dipole/open-boundary correction.

The dynamic potential enters the band energy and is removed by the normal
potential subtraction. `E_gaussian_field` adds its complete electronic plus
ionic cross energy once. Ionic forces include this field and retain the QM
net reaction. The running log contains a full-precision
`QMDPA_GAUSSIAN_FIELD_V1_RESULT` block with ionic/electronic energies in Ry and
environment forces in Ry/Bohr, keyed by input identity. The coupled driver
must use these environment derivatives rather than independently centering
QM forces. SCF convergence and complete results are required before use.

Software status and exact qualification belong to the external release record.
This interface alone does not establish material accuracy, charged-cell support,
field convergence, long dynamics, stress or performance.
