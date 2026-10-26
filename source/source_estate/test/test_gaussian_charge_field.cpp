#include "source_estate/gaussian_charge_field.h"
#include "source_base/constants.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace
{
using Vector = ModuleBase::Vector3<double>;
using elecstate::GaussianExternalCharge;
using elecstate::GaussianFieldMode;
using elecstate::GaussianFieldResult;

struct Fixture
{
    double length = 12.0;
    std::vector<GaussianFieldMode> modes;
    std::vector<GaussianExternalCharge> environment;
    std::vector<Vector> positions;
    std::vector<double> charges;
    std::vector<std::complex<double>> density;

    Fixture()
    {
        const double scale = 2.0 * ModuleBase::PI / length;
        const double volume = length * length * length;
        for (int x = -3; x <= 3; ++x)
        {
            for (int y = -3; y <= 3; ++y)
            {
                for (int z = -3; z <= 3; ++z)
                {
                    if (x == 0 && y == 0 && z == 0) { continue; }
                    const double gx = scale * x;
                    const double gy = scale * y;
                    const double gz = scale * z;
                    const Vector g(gx, gy, gz);
                    const double g2 = g * g;
                    const double coefficient = 8.0 * ModuleBase::PI / (volume * g2);
                    modes.push_back({g, coefficient});
                    const Vector center(1.2, 0.7, -0.6);
                    const double phase = -(g * center);
                    const std::complex<double> argument(0.0, phase);
                    const double envelope = -0.7 * g2;
                    density.push_back(2.0 * std::exp(envelope) * std::exp(argument));
                }
            }
        }
        environment = {{19, 0.4, 0.8, Vector(3.2, -1.3, 0.8)},
                       {8, -0.4, 1.1, Vector(-2.2, 1.8, 0.2)}};
        positions = {Vector(-0.7, 0.3, 0.1), Vector(0.7, -0.2, -0.1)};
        charges = {1.0, 1.0};
    }

    GaussianFieldResult evaluate() const
    {
        return elecstate::evaluate_gaussian_charge_field(modes, environment, positions, charges, density);
    }

    double energy() const
    {
        const auto result = evaluate();
        return result.ionic_energy + result.electronic_energy;
    }
};

double& coordinate(Vector& value, const int axis)
{
    if (axis == 0) { return value.x; }
    if (axis == 1) { return value.y; }
    return value.z;
}

void compare_forces(const GaussianFieldResult& a, const GaussianFieldResult& b, const double tolerance)
{
    ASSERT_EQ(a.ionic_force.size(), b.ionic_force.size());
    ASSERT_EQ(a.environment_force.size(), b.environment_force.size());
    for (std::size_t i = 0; i < a.ionic_force.size(); ++i)
    {
        EXPECT_NEAR(a.ionic_force[i].x, b.ionic_force[i].x, tolerance);
        EXPECT_NEAR(a.ionic_force[i].y, b.ionic_force[i].y, tolerance);
        EXPECT_NEAR(a.ionic_force[i].z, b.ionic_force[i].z, tolerance);
    }
    for (std::size_t j = 0; j < a.environment_force.size(); ++j)
    {
        EXPECT_NEAR(a.environment_force[j].x, b.environment_force[j].x, tolerance);
        EXPECT_NEAR(a.environment_force[j].y, b.environment_force[j].y, tolerance);
        EXPECT_NEAR(a.environment_force[j].z, b.environment_force[j].z, tolerance);
    }
}
} // namespace

TEST(GaussianChargeField, AllIonicAndEnvironmentCoordinateDerivatives)
{
    Fixture fixture;
    const auto analytic = fixture.evaluate();
    for (const double step : {1.0e-5, 5.0e-6})
    {
        for (int set = 0; set < 2; ++set)
        {
            const std::size_t count = set == 0 ? fixture.positions.size() : fixture.environment.size();
            for (std::size_t atom = 0; atom < count; ++atom)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    auto plus = fixture;
                    auto minus = fixture;
                    Vector& positive = set == 0 ? plus.positions[atom] : plus.environment[atom].position;
                    Vector& negative = set == 0 ? minus.positions[atom] : minus.environment[atom].position;
                    coordinate(positive, axis) += step;
                    coordinate(negative, axis) -= step;
                    const double numerical = -(plus.energy() - minus.energy()) / (2.0 * step);
                    Vector force = set == 0 ? analytic.ionic_force[atom] : analytic.environment_force[atom];
                    const double expected = coordinate(force, axis);
                    EXPECT_NEAR(numerical, expected, 2.0e-8);
                }
            }
        }
    }
}

TEST(GaussianChargeField, DensityFunctionalDerivative)
{
    Fixture fixture;
    const auto value = fixture.evaluate();
    const double step = 1.0e-5;
    for (const std::size_t index : {std::size_t(0), std::size_t(101), std::size_t(341)})
    {
        for (int component = 0; component < 2; ++component)
        {
            auto plus = fixture;
            auto minus = fixture;
            const std::complex<double> delta = component == 0
                ? std::complex<double>(step, 0.0) : std::complex<double>(0.0, step);
            plus.density[index] += delta;
            minus.density[index] -= delta;
            const double numerical = (plus.energy() - minus.energy()) / (2.0 * step);
            const double expected = component == 0 ? value.electron_potential[index].real()
                                                   : value.electron_potential[index].imag();
            EXPECT_NEAR(numerical, expected, 2.0e-10);
        }
    }
}

TEST(GaussianChargeField, JointTranslationAndIndividualImages)
{
    Fixture fixture;
    const auto base = fixture.evaluate();
    const Vector shift(1.5, -0.9, 2.1);
    auto moved = fixture;
    for (auto& position : moved.positions) { position += shift; }
    for (auto& charge : moved.environment) { charge.position += shift; }
    for (std::size_t i = 0; i < moved.modes.size(); ++i)
    {
        const double phase = -(moved.modes[i].vector * shift);
        const std::complex<double> argument(0.0, phase);
        moved.density[i] *= std::exp(argument);
    }
    const auto translated = moved.evaluate();
    EXPECT_NEAR(base.ionic_energy, translated.ionic_energy, 2.0e-14);
    EXPECT_NEAR(base.electronic_energy, translated.electronic_energy, 2.0e-14);
    compare_forces(base, translated, 2.0e-14);
    moved = fixture;
    moved.environment[0].position.x += fixture.length;
    moved.positions[1].z -= fixture.length;
    const auto images = moved.evaluate();
    EXPECT_NEAR(base.ionic_energy, images.ionic_energy, 2.0e-14);
    EXPECT_NEAR(base.electronic_energy, images.electronic_energy, 2.0e-14);
    compare_forces(base, images, 2.0e-14);
}

TEST(GaussianChargeField, IndependentCosineCrossEnergyAndReaction)
{
    Fixture fixture;
    std::fill(fixture.density.begin(), fixture.density.end(), std::complex<double>(0.0));
    const auto value = fixture.evaluate();
    double independent = 0.0;
    for (const auto& mode : fixture.modes)
    {
        const double g2 = mode.vector * mode.vector;
        for (std::size_t ion = 0; ion < fixture.positions.size(); ++ion)
        {
            for (const auto& charge : fixture.environment)
            {
                const Vector displacement = fixture.positions[ion] - charge.position;
                const double phase = mode.vector * displacement;
                const double envelope = -0.5 * charge.sigma * charge.sigma * g2;
                independent += mode.coefficient * fixture.charges[ion] * charge.charge
                             * std::exp(envelope) * std::cos(phase);
            }
        }
    }
    EXPECT_NEAR(value.ionic_energy, independent, 2.0e-14);
    Vector sum;
    for (const auto& force : value.ionic_force) { sum += force; }
    for (const auto& force : value.environment_force) { sum += force; }
    EXPECT_NEAR(sum.x, 0.0, 2.0e-14);
    EXPECT_NEAR(sum.y, 0.0, 2.0e-14);
    EXPECT_NEAR(sum.z, 0.0, 2.0e-14);
}

TEST(GaussianChargeField, ZeroFieldAndPermutation)
{
    Fixture fixture;
    const auto baseline = fixture.evaluate();
    std::reverse(fixture.environment.begin(), fixture.environment.end());
    auto permuted = fixture.evaluate();
    std::reverse(permuted.environment_force.begin(), permuted.environment_force.end());
    EXPECT_DOUBLE_EQ(baseline.ionic_energy, permuted.ionic_energy);
    EXPECT_DOUBLE_EQ(baseline.electronic_energy, permuted.electronic_energy);
    compare_forces(baseline, permuted, 1.0e-15);
    for (auto& charge : fixture.environment) { charge.charge = 0.0; }
    const auto zero = fixture.evaluate();
    EXPECT_DOUBLE_EQ(zero.ionic_energy, 0.0);
    EXPECT_DOUBLE_EQ(zero.electronic_energy, 0.0);
    for (const auto& potential : zero.electron_potential)
    {
        EXPECT_DOUBLE_EQ(potential.real(), 0.0);
        EXPECT_DOUBLE_EQ(potential.imag(), 0.0);
    }
    fixture.environment.clear();
    const auto empty = fixture.evaluate();
    EXPECT_DOUBLE_EQ(empty.ionic_energy, 0.0);
    EXPECT_DOUBLE_EQ(empty.electronic_energy, 0.0);
    EXPECT_TRUE(empty.environment_force.empty());
}

TEST(GaussianChargeField, StrictInputAndUnsupportedChargeRejection)
{
    std::istringstream good("QMDPA_GAUSSIAN_FIELD_V1 2\n7 .3 .8 1 2 3\n4 -.3 .7 -1 -2 -3\n");
    const auto parsed = elecstate::read_gaussian_external_charges(good);
    ASSERT_EQ(parsed.size(), 2u);
    EXPECT_EQ(parsed[1].id, 4);
    EXPECT_DOUBLE_EQ(parsed[0].sigma, 0.8);
    for (const std::string& input : {
         "bad 0", "QMDPA_GAUSSIAN_FIELD_V1 -1", "QMDPA_GAUSSIAN_FIELD_V1 1 1 0 1 0 0",
         "QMDPA_GAUSSIAN_FIELD_V1 1 1 .2 1 0 0 0", "QMDPA_GAUSSIAN_FIELD_V1 1 1 0 0 0 0 0",
         "QMDPA_GAUSSIAN_FIELD_V1 2 1 .2 1 0 0 0 1 -.2 1 0 0 0",
         "QMDPA_GAUSSIAN_FIELD_V1 0 trailing", "QMDPA_GAUSSIAN_FIELD_V1 1 1 0 nan 0 0 0"})
    {
        std::istringstream stream(input);
        EXPECT_THROW(elecstate::read_gaussian_external_charges(stream), std::invalid_argument);
    }
    Fixture fixture;
    fixture.environment[0].charge += 0.1;
    EXPECT_THROW(fixture.evaluate(), std::invalid_argument);
    fixture = Fixture();
    fixture.modes[0].vector = Vector();
    EXPECT_THROW(fixture.evaluate(), std::invalid_argument);
    fixture = Fixture();
    fixture.density.pop_back();
    EXPECT_THROW(fixture.evaluate(), std::invalid_argument);
}
