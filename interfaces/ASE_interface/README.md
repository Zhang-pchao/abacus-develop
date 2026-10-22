# abacuslite

## Introduction

abacuslite is a lightweight plugin for ABACUS (Atomic-orbital Based Ab-initio Computation at UStc), implementing the ASE (Atomic Simulation Environment) calculator interface.

### Key Features

- **Lightweight Design**: Implemented as a plugin, no need to modify ASE core code
- **Version Compatibility**: Supports ASE versions satisfying the package requirement `ase>=3.22`
- **ASE Integration**: Uses ASE as the running platform, making ABACUS a callable calculator within it
- **Function Support**: Provides SCF-based energy, force, and stress evaluations through ASE. ASE can use these evaluations for relaxation, molecular dynamics, NEB, band-structure, and density-of-states workflows.
- **Socket Support**: `AbacusSocketIO` provides fixed-cell i-PI socket calculations, with energy always available and forces/stress enabled independently when requested.

## Installation

Install the plugin from the ASE interface directory:

```bash
cd interfaces/ASE_interface
pip install .
```

## Usage Examples

Please refer to the example scripts in the `examples` folder. Recommended learning path:

1. **scf.py** - Basic SCF calculation example
2. **relax.py** - Atomic position relaxation calculation
3. **cellrelax.py** - Cell parameter relaxation calculation
4. **bandstructure.py** - Band structure calculation
5. **dos.py** - Density of states calculation
6. **md.py** - Molecular dynamics simulation
7. **constraintmd.py** - Constrained molecular dynamics simulation
8. **metadynamics.py** - Metadynamics simulation
9. **neb.py** - Nudged Elastic Band (NEB) calculation
10. **soc.py** - Noncollinear spin-orbit coupling calculation
11. **socketio.py** - Fixed-cell ASE optimization with `AbacusSocketIO`, running ABACUS as an i-PI socket client

The regular `Abacus` calculator runs one ABACUS calculation for each ASE property evaluation. ASE controls the relaxation, molecular-dynamics, and other workflow steps. The socket calculator reuses one ABACUS process for position updates, while the cell and electronic-structure settings remain fixed for that calculator instance.

## Units

ASE inputs use Angstrom, and calculator results use eV, eV/Angstrom, and
eV/Angstrom cubed (ASE stress sign). The interface uses the native constants in
source/source_base/constants.h (BOHR_TO_A and Ry_to_eV) so file and socket
calculations refer to the same geometry and energy convention. These constants
differ slightly from ASE's CODATA values.

AbacusSocketIO leaves the native i-PI protocol in atomic units. It rescales an
internal copy before ASE serializes coordinates and the cell, then normalizes
ASE-decoded energy, forces, and virial. The user's atoms and the initial STRU
retain their physical Angstrom values. Stress uses the original cell volume.
The file readers invert ABACUS's own kbar conversion before converting to this
same energy-per-volume convention, including the native joule/eV difference.
read_stru returns raw STRU fields; running-log and MD-dump trajectory readers
already consume the lattice constant printed in Angstrom.

Focused boundary tests (no ABACUS executable needed):

    python -m unittest discover -s tests -p 'test_units.py' -v

## Authors

- Yuyang Ji
- Zhenxiong Shen
- Yike Huang
- Zhaoqing Liu

## Acknowledgments

Thanks to the ABACUS development team for their support and contributions.

## License

The applicable license terms are provided in the repository [LICENSE](../../LICENSE).

## Contact

If you have any questions or suggestions, please contact us through:

- GitHub: [deepmodeling/abacus-develop](https://github.com/deepmodeling/abacus-develop)
