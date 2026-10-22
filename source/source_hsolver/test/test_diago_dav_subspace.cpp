#include "source_hsolver/diago_dav_subspace.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using Complex = std::complex<double>;
const int dimension = 12;
const int bands = 2;
const int vector_size = dimension * bands;

// The same diagonal H, S=I fixture used by test_diago_david.cpp, restricted to
// one serial complex-double block so it also builds with ENABLE_LCAO=OFF.
class DiagonalOperator final : public hsolver::HSOperator<Complex>
{
  public:
    void update_k(const int ik) override {}

    void hpsi(const Complex* x, Complex* hx, const int ld, const int nvec) const override
    {
        ASSERT_EQ(ld, dimension);
        for (int band = 0; band < nvec; ++band)
        {
            for (int row = 0; row < dimension; ++row)
            {
                hx[band * ld + row] = (row + 1.0) * x[band * ld + row];
            }
        }
    }

    void spsi(const Complex* x, Complex* sx, const int ld, const int nvec) const override
    {
        ASSERT_EQ(ld, dimension);
        const int count = ld * nvec;
        std::copy_n(x, count, sx);
    }
};

std::vector<Complex> near_eigenvectors(const double perturbation)
{
    std::vector<Complex> vectors(vector_size, Complex(0.0));
    vectors[0] = 1.0;
    vectors[2] = Complex(0.0, perturbation);
    vectors[dimension + 1] = 1.0;
    const double second_perturbation = 2.0 * perturbation;
    vectors[dimension + 3] = Complex(0.0, second_perturbation);
    return vectors;
}

int solve(std::vector<Complex>& vectors, std::vector<double>& eigenvalues)
{
    const std::vector<double> precondition(dimension, 1.0);
    const std::vector<double> thresholds(bands, 1e-13);
    const hsolver::diag_comm_info comm = {0, 1};
    DiagonalOperator op;
    hsolver::Diago_DavSubspace<Complex> solver(precondition, bands, dimension, 3, 1e-13, 300, comm, 0, 1);
    Complex* vector_data = vectors.data();
    double* eigenvalue_data = eigenvalues.data();
    return solver.diag(op, vector_data, dimension, eigenvalue_data, thresholds, false);
}

double residual_norm(const std::vector<Complex>& vectors, const int band, const double eigenvalue)
{
    double norm_squared = 0.0;
    for (int row = 0; row < dimension; ++row)
    {
        const Complex residual = (row + 1.0 - eigenvalue) * vectors[band * dimension + row];
        norm_squared += std::norm(residual);
    }
    return std::sqrt(norm_squared);
}

double diagnostic_value(const std::string& output, const std::string& key)
{
    const std::size_t position = output.find(key);
    EXPECT_NE(position, std::string::npos);
    if (position == std::string::npos)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::string remaining = output.substr(position + key.size());
    std::istringstream value_stream(remaining);
    double value = std::numeric_limits<double>::quiet_NaN();
    value_stream >> value;
    const bool parsed = !value_stream.fail();
    EXPECT_TRUE(parsed);
    return value;
}

// These death tests characterize the current failure, not desired solver
// convergence. A numerical fix must replace them with solution assertions.
TEST(DavSubspaceResidualDeathTest, NearEigenspaceReportsSmallPositiveResidual)
{
    std::vector<Complex> vectors = near_eigenvectors(1e-13);
    std::vector<double> eigenvalues(bands, 0.0);
    const double initial_residual = residual_norm(vectors, 0, 1.0);
    ASSERT_GT(initial_residual, 0.0);
    ASSERT_LT(initial_residual, 1e-12);

    testing::internal::CaptureStdout();
    EXPECT_EXIT(solve(vectors, eigenvalues), testing::ExitedWithCode(1), "");
    const std::string output = testing::internal::GetCapturedStdout();
    const double norm = diagnostic_value(output, "L2_norm=");
    const double threshold = diagnostic_value(output, "threshold=");
    const double band = diagnostic_value(output, "band_index=");
    const double residual_index = diagnostic_value(output, "residual_index=");
    const double nbase = diagnostic_value(output, "nbase=");
    const double eigenvalue = diagnostic_value(output, "eigenvalue=");
    EXPECT_GT(norm, 0.0);
    EXPECT_LE(norm, threshold);
    EXPECT_DOUBLE_EQ(threshold, 1e-12);
    EXPECT_DOUBLE_EQ(band, 0.0);
    EXPECT_DOUBLE_EQ(residual_index, 0.0);
    EXPECT_DOUBLE_EQ(nbase, 2.0);
    EXPECT_NEAR(eigenvalue, 1.0, 1e-14);
}

TEST(DavSubspaceResidualDeathTest, ExactEigenspaceReachesZeroNormBoundary)
{
    std::vector<Complex> vectors = near_eigenvectors(0.0);
    std::vector<double> eigenvalues(bands, 0.0);
    const double initial_residual = residual_norm(vectors, 0, 1.0);
    ASSERT_DOUBLE_EQ(initial_residual, 0.0);

#ifdef NDEBUG
    // Release reaches the guard after normalize_op has divided by zero.
    testing::internal::CaptureStdout();
    EXPECT_EXIT(solve(vectors, eigenvalues), testing::ExitedWithCode(1), "");
    const std::string output = testing::internal::GetCapturedStdout();
    const double norm = diagnostic_value(output, "L2_norm=");
    EXPECT_DOUBLE_EQ(norm, 0.0);
#else
    // Debug fails earlier at normalize_op's existing positive-norm assertion.
    EXPECT_DEATH(solve(vectors, eigenvalues), "psi_m_norm > 0.0");
#endif
}

TEST(DavSubspaceResidualTest, OrdinaryGuessRecoversEigenpairs)
{
    std::vector<Complex> vectors(vector_size);
    for (int band = 0; band < bands; ++band)
    {
        for (int row = 0; row < dimension; ++row)
        {
            const double real_angle = (row + 1) * (band + 1) * 0.37;
            const double offset_angle = (row + 2) * 0.51;
            const double imag_angle = (row + 1) * (band + 2) * 0.29;
            const double real = std::sin(real_angle) + 0.2 * std::cos(offset_angle);
            const double imag = 0.3 * std::cos(imag_angle);
            vectors[band * dimension + row] = Complex(real, imag);
        }
    }
    std::vector<double> eigenvalues(bands, 0.0);
    const int iterations = solve(vectors, eigenvalues);
    EXPECT_GT(iterations, 0);
    for (int band = 0; band < bands; ++band)
    {
        const double expected = band + 1.0;
        const double residual = residual_norm(vectors, band, eigenvalues[band]);
        double norm_squared = 0.0;
        for (int row = 0; row < dimension; ++row)
        {
            norm_squared += std::norm(vectors[band * dimension + row]);
        }
        EXPECT_NEAR(eigenvalues[band], expected, 1e-9);
        EXPECT_LT(residual, 2e-6);
        EXPECT_NEAR(norm_squared, 1.0, 1e-10);
    }
    Complex overlap = 0.0;
    for (int row = 0; row < dimension; ++row)
    {
        overlap += std::conj(vectors[row]) * vectors[dimension + row];
    }
    const double overlap_magnitude = std::abs(overlap);
    EXPECT_LT(overlap_magnitude, 1e-10);
}
} // namespace
