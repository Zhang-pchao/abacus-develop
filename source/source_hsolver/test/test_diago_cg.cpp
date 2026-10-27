#include "source_hsolver/diago_cg.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

namespace
{
const int dimension = 4;
const int bands = 2;

template <typename Real>
class DiagonalOperator final : public hsolver::HSOperator<std::complex<Real>>
{
    using Complex = std::complex<Real>;

  public:
    explicit DiagonalOperator(const bool generalized) : generalized_(generalized) {}
    void update_k(const int ik) override {}

    void hpsi(const Complex* x, Complex* hx, const int ld, const int nvec) const override
    {
        ASSERT_EQ(ld, dimension);
        for (int band = 0; band < nvec; ++band)
        {
            for (int row = 0; row < ld; ++row)
            {
                const Real overlap = generalized_ ? Real(row + 1) : Real(1);
                hx[band * ld + row] = Real(row + 1) * overlap * x[band * ld + row];
            }
        }
    }

    void spsi(const Complex* x, Complex* sx, const int ld, const int nvec) const override
    {
        ASSERT_EQ(ld, dimension);
        for (int band = 0; band < nvec; ++band)
        {
            for (int row = 0; row < ld; ++row)
            {
                const Real overlap = generalized_ ? Real(row + 1) : Real(1);
                sx[band * ld + row] = overlap * x[band * ld + row];
            }
        }
    }

  private:
    const bool generalized_;
};

template <typename Real>
void solve_and_check(const Real independent_component, const bool generalized, const double tolerance)
{
    using Complex = std::complex<Real>;
    std::vector<Complex> vectors(dimension * bands, Complex(0));
    vectors[0] = Complex(Real(0.6), Real(0.8));
    vectors[dimension] = vectors[0];
    vectors[dimension + 1] = Complex(Real(0.8), Real(-0.6)) * independent_component;
    std::vector<Real> eigenvalues(bands, Real(0));
    const std::vector<Real> precondition(dimension, Real(1));
    const std::vector<double> thresholds(bands, tolerance * 0.01);
    const hsolver::diag_comm_info comm(0, 1);
    const Real threshold = Real(tolerance * 0.01);
    hsolver::DiagoCG<Complex> solver("pw", "scf", false, comm, threshold, 100);
    DiagonalOperator<Real> op(generalized);
    Complex* vector_data = vectors.data();
    Real* eigenvalue_data = eigenvalues.data();
    const Real* precondition_data = precondition.data();
    solver.diag(op, dimension, bands, dimension, vector_data, eigenvalue_data, thresholds, precondition_data);
    for (int band = 0; band < bands; ++band)
    {
        EXPECT_NEAR(eigenvalues[band], band + 1.0, tolerance);
        for (int other = 0; other < bands; ++other)
        {
            Complex overlap(0);
            for (int row = 0; row < dimension; ++row)
            {
                const Real weight = generalized ? Real(row + 1) : Real(1);
                overlap += std::conj(vectors[band * dimension + row]) * weight * vectors[other * dimension + row];
            }
            const double expected = band == other ? 1.0 : 0.0;
            EXPECT_NEAR(std::real(overlap), expected, tolerance);
            EXPECT_NEAR(std::imag(overlap), 0.0, tolerance);
        }
        double residual_squared = 0.0;
        for (int row = 0; row < dimension; ++row)
        {
            const Real weight = generalized ? Real(row + 1) : Real(1);
            const Complex residual = weight * (Real(row + 1) - eigenvalues[band]) * vectors[band * dimension + row];
            residual_squared += std::norm(residual);
        }
        const double residual_norm = std::sqrt(residual_squared);
        EXPECT_NEAR(residual_norm, 0.0, tolerance);
    }
}

TEST(DiagoCGStability, IndependentComplexPair)
{
    solve_and_check<double>(1.0, false, 1e-12);
}

TEST(DiagoCGStability, NearlyParallelComplexDoublePair)
{
    solve_and_check<double>(1e-9, false, 1e-12);
}

TEST(DiagoCGStability, NearlyParallelComplexFloatPair)
{
    solve_and_check<float>(1e-4f, false, 1e-5);
}

TEST(DiagoCGStability, NearlyParallelGeneralizedPair)
{
    solve_and_check<double>(1e-9, true, 1e-12);
}

TEST(DiagoCGStabilityDeathTest, ExactlyDependentPairStillFails)
{
    EXPECT_EXIT(solve_and_check<double>(0.0, false, 1e-12), testing::ExitedWithCode(1), "");
}

class SpectralOperator final : public hsolver::HSOperator<std::complex<double>>
{
  public:
    explicit SpectralOperator(const bool generalized) : generalized_(generalized) {}
    static double energy(const int i) { return -20.0 + 0.2 * i + 0.005 * i * i; }
    double overlap(const int i) const { return generalized_ ? 1.0 + 0.01 * i : 1.0; }
    void update_k(const int ik) override {}
    void hpsi(const std::complex<double>* x, std::complex<double>* hx, const int ld, const int nvec) const override
    {
        for (int j = 0; j < nvec; ++j)
        {
            for (int i = 0; i < ld; ++i)
            {
                hx[j * ld + i] = overlap(i) * energy(i) * x[j * ld + i];
            }
        }
    }
    void spsi(const std::complex<double>* x, std::complex<double>* sx, const int ld, const int nvec) const override
    {
        for (int j = 0; j < nvec; ++j)
        {
            for (int i = 0; i < ld; ++i)
            {
                sx[j * ld + i] = overlap(i) * x[j * ld + i];
            }
        }
    }
  private:
    const bool generalized_;
};

void check_repeated_tight_solve(const double threshold, const bool generalized)
{
    using Complex = std::complex<double>;
    const int size = 64;
    const int count = 16;
    const double pi = std::acos(-1.0);
    const double amplitude = 1.0 / std::sqrt(size);
    std::vector<Complex> vectors(size * count);
    for (int j = 0; j < count; ++j)
    {
        for (int i = 0; i < size; ++i)
        {
            const double angle = 2 * pi * i * j / size;
            vectors[j * size + i] = std::polar(amplitude, angle);
        }
    }
    std::vector<double> eigenvalues(count, 0.0);
    const std::vector<double> thresholds(count, threshold);
    std::vector<double> precondition(size);
    for (int i = 0; i < size; ++i) { precondition[i] = 1.0 + 0.2 * i; }
    const hsolver::diag_comm_info comm(0, 1);
    hsolver::DiagoCG<Complex> solver("pw", "scf", false, comm, threshold, 100);
    SpectralOperator op(generalized);
    Complex* vector_data = vectors.data();
    double* eigenvalue_data = eigenvalues.data();
    const double* precondition_data = precondition.data();
    for (int cycle = 0; cycle < 4; ++cycle)
    {
        solver.diag(op, size, count, size, vector_data, eigenvalue_data, thresholds, precondition_data);
        for (int j = 0; j < count; ++j)
        {
            const double expected_energy = SpectralOperator::energy(j);
            EXPECT_NEAR(eigenvalues[j], expected_energy, 1e-10);
            double residual_squared = 0;
            for (int i = 0; i < size; ++i)
            {
                const Complex residual = op.overlap(i) * (SpectralOperator::energy(i) - eigenvalues[j])
                                         * vectors[j * size + i];
                residual_squared += std::norm(residual);
            }
            const double residual_norm = std::sqrt(residual_squared);
            EXPECT_LT(residual_norm, 1e-6);
            for (int k = 0; k < count; ++k)
            {
                Complex overlap(0);
                for (int i = 0; i < size; ++i)
                {
                    overlap += std::conj(vectors[j * size + i]) * op.overlap(i) * vectors[k * size + i];
                }
                const double expected_overlap = j == k ? 1.0 : 0.0;
                const Complex overlap_error = overlap - expected_overlap;
                const double error = std::abs(overlap_error);
                EXPECT_LT(error, 1e-12);
            }
        }
    }
}

TEST(DiagoCGStability, TightRepeatedComplexSpectrum)
{
    check_repeated_tight_solve(1e-16, false);
}

TEST(DiagoCGStability, SubUlpEnergyThresholdPreservesEigenvectors)
{
    check_repeated_tight_solve(1e-20, false);
}

TEST(DiagoCGStability, TightRepeatedGeneralizedSpectrum)
{
    check_repeated_tight_solve(1e-16, true);
}
} // namespace
