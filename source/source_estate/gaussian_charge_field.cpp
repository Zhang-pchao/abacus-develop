#include "gaussian_charge_field.h"

#include <cmath>
#include <istream>
#include <set>
#include <stdexcept>
#include <string>

namespace
{
using Vector = ModuleBase::Vector3<double>;

bool finite_vector(const Vector& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validate_environment(const std::vector<elecstate::GaussianExternalCharge>& charges)
{
    std::set<int> ids;
    double total = 0.0;
    for (const auto& charge : charges)
    {
        if (charge.id <= 0 || !ids.insert(charge.id).second || !std::isfinite(charge.charge)
            || !std::isfinite(charge.sigma) || charge.sigma <= 0.0 || !finite_vector(charge.position))
        {
            throw std::invalid_argument("Invalid Gaussian external charge identity, value, width or position");
        }
        total += charge.charge;
    }
    if (!std::isfinite(total) || std::abs(total) > 1.0e-12)
    {
        throw std::invalid_argument("Gaussian environment must be neutral; no compensating background is applied");
    }
}
} // namespace

namespace elecstate
{
std::vector<GaussianExternalCharge> read_gaussian_external_charges(std::istream& input)
{
    std::string header;
    long long count = 0;
    if (!(input >> header >> count) || header != "QMDPA_GAUSSIAN_FIELD_V1" || count < 0 || count > 1000000)
    {
        throw std::invalid_argument("Expected QMDPA_GAUSSIAN_FIELD_V1 and a bounded nonnegative charge count");
    }
    std::vector<GaussianExternalCharge> charges;
    for (long long row = 0; row < count; ++row)
    {
        GaussianExternalCharge charge;
        if (!(input >> charge.id >> charge.charge >> charge.sigma
                    >> charge.position.x >> charge.position.y >> charge.position.z))
        {
            throw std::invalid_argument("Malformed Gaussian external charge row: expected id q sigma x y z");
        }
        charges.push_back(charge);
    }
    std::string extra;
    if (input >> extra)
    {
        throw std::invalid_argument("Unexpected trailing Gaussian external charge data");
    }
    if (!input.eof())
    {
        throw std::invalid_argument("Gaussian external charge input could not be read completely");
    }
    validate_environment(charges);
    return charges;
}

GaussianFieldResult evaluate_gaussian_charge_field(
    const std::vector<GaussianFieldMode>& modes,
    const std::vector<GaussianExternalCharge>& environment,
    const std::vector<Vector>& ion_positions,
    const std::vector<double>& ion_charges,
    const std::vector<std::complex<double>>& integrated_density)
{
    validate_environment(environment);
    if (ion_positions.size() != ion_charges.size() || modes.size() != integrated_density.size())
    {
        throw std::invalid_argument("Gaussian field ion or density array sizes differ");
    }
    for (std::size_t ion = 0; ion < ion_positions.size(); ++ion)
    {
        if (!finite_vector(ion_positions[ion]) || !std::isfinite(ion_charges[ion]))
        {
            throw std::invalid_argument("Gaussian field ionic input is not finite");
        }
    }
    GaussianFieldResult result;
    result.electron_potential.resize(modes.size());
    result.ionic_force.resize(ion_positions.size());
    result.environment_force.resize(environment.size());
    std::vector<std::complex<double>> source(environment.size());
    const std::complex<double> imaginary(0.0, 1.0);
    for (std::size_t mode = 0; mode < modes.size(); ++mode)
    {
        const Vector& g = modes[mode].vector;
        const double g2 = g * g;
        const double coefficient = modes[mode].coefficient;
        const std::complex<double> density = integrated_density[mode];
        if (!finite_vector(g) || !std::isfinite(g2) || g2 <= 0.0
            || !std::isfinite(coefficient) || coefficient <= 0.0
            || !std::isfinite(density.real()) || !std::isfinite(density.imag()))
        {
            throw std::invalid_argument("Gaussian field requires finite nonzero modes and density");
        }
        std::complex<double> total_source = 0.0;
        for (std::size_t j = 0; j < environment.size(); ++j)
        {
            const auto& charge = environment[j];
            const double exponent = -0.5 * charge.sigma * charge.sigma * g2;
            const double phase = g * charge.position;
            const std::complex<double> argument = -imaginary * phase;
            source[j] = charge.charge * std::exp(exponent) * std::exp(argument);
            total_source += source[j];
        }
        std::complex<double> ionic_density = 0.0;
        for (std::size_t ion = 0; ion < ion_positions.size(); ++ion)
        {
            const double phase = g * ion_positions[ion];
            const std::complex<double> argument = -imaginary * phase;
            const std::complex<double> factor = ion_charges[ion] * std::exp(argument);
            ionic_density += factor;
            const std::complex<double> interaction = std::conj(factor) * total_source;
            result.ionic_force[ion] += g * (coefficient * interaction.imag());
        }
        const std::complex<double> ionic_interaction = std::conj(ionic_density) * total_source;
        const std::complex<double> electronic_interaction = std::conj(density) * total_source;
        result.ionic_energy += coefficient * ionic_interaction.real();
        result.electronic_energy -= coefficient * electronic_interaction.real();
        result.electron_potential[mode] = -coefficient * total_source;
        const std::complex<double> total_qm_density = ionic_density - density;
        for (std::size_t j = 0; j < environment.size(); ++j)
        {
            const std::complex<double> interaction = std::conj(total_qm_density) * source[j];
            result.environment_force[j] -= g * (coefficient * interaction.imag());
        }
    }
    return result;
}
} // namespace elecstate
