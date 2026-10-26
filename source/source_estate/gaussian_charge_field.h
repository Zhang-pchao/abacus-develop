#ifndef GAUSSIAN_CHARGE_FIELD_H
#define GAUSSIAN_CHARGE_FIELD_H

#include "source_base/vector3.h"

#include <complex>
#include <iosfwd>
#include <vector>

namespace elecstate
{

// Positions and Gaussian standard deviations are in Bohr; charge is in e.
struct GaussianExternalCharge
{
    int id;
    double charge;
    double sigma;
    ModuleBase::Vector3<double> position;
};

// Each signed reciprocal vector occurs once, including its negative partner.
// coefficient = 8*pi/(volume*|G|^2), in Ry/e^2. G=0 is excluded.
struct GaussianFieldMode
{
    ModuleBase::Vector3<double> vector;
    double coefficient;
};

struct GaussianFieldResult
{
    double ionic_energy = 0.0;
    double electronic_energy = 0.0;
    std::vector<std::complex<double>> electron_potential;
    std::vector<ModuleBase::Vector3<double>> ionic_force;
    std::vector<ModuleBase::Vector3<double>> environment_force;
};

// Strict format: QMDPA_GAUSSIAN_FIELD_V1, count, then id q sigma x y z.
// All numerical fields use atomic units. Requires neutral external charges.
std::vector<GaussianExternalCharge> read_gaussian_external_charges(std::istream& input);

// integrated_density[G] = integral_cell n(r) exp(-iG.r) dr. The complete
// result is the QM-environment cross term only: no environment self or pair
// energy, no QM-QM Coulomb, and no compensating background. Forces are Ry/Bohr.
// A rank may supply a subset of modes; callers reduce the extensive results.
GaussianFieldResult evaluate_gaussian_charge_field(
    const std::vector<GaussianFieldMode>& modes,
    const std::vector<GaussianExternalCharge>& environment,
    const std::vector<ModuleBase::Vector3<double>>& ion_positions,
    const std::vector<double>& ion_charges,
    const std::vector<std::complex<double>>& integrated_density);

} // namespace elecstate

#endif
