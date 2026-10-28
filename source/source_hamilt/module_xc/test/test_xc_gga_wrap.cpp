#include "gtest/gtest.h"
#include "../xc_functional.h"
#include <cmath>

namespace
{
struct KernelReference
{
    int kind;
    int flag;
    double rho;
    double gradient_squared;
    double energy;
    double density_derivative;
    double twice_gradient_derivative;
};
}

TEST(XCGgaWrap, PbeKernelAnalyticLimits)
{
    // Independent 80-digit energy definitions differentiated with mpmath.
    // Decimal constants match the built-in functional; derivatives are not
    // obtained from the production potential implementation.
    const KernelReference references[] = {
        {0, 0, 0.088, 0.0, 0.0, 0.0, -0.21638557885350662},
        {0, 0, 0.088, 8.1e-12, -8.763615943236587e-13, 1.327820597359958e-11, -0.216385578837189},
        {0, 1, 0.088, 0.0, 0.0, 0.0, -0.21638557885350662},
        {0, 1, 0.088, 8.1e-12, -8.763615943353631e-13, 1.3278205974131602e-11, -0.21638557884296897},
        {0, 2, 0.088, 0.0, 0.0, 0.0, -0.1216967966140405},
        {0, 2, 0.088, 8.1e-12, -4.928720262764124e-13, 7.467757973568322e-12, -0.1216967966088792},
        {1, 0, 0.088, 0.0, 0.0, 0.0, 0.21638557885350643},
        {1, 0, 0.088, 8.1e-12, 8.763615942163473e-13, -1.3278205969253462e-11, 0.21638557878419598},
        {1, 1, 0.088, 0.0, 0.0, 0.0, 0.1491765255409829},
        {1, 1, 0.088, 8.1e-12, 6.041649283742742e-13, -9.154014064508151e-12, 0.14917652550804142},
        {0, 0, 1.7, 0.0, 0.0, 0.0, -0.004174520979089177},
        {0, 0, 1.7, 8.1e-12, -1.690680996531093e-14, 1.3260243110047417e-14, -0.00417452097908906},
        {0, 1, 1.7, 0.0, 0.0, 0.0, -0.004174520979089177},
        {0, 1, 1.7, 8.1e-12, -1.6906809965311013e-14, 1.3260243110047615e-14, -0.004174520979089102},
        {0, 2, 1.7, 0.0, 0.0, 0.0, -0.002347780444727304},
        {0, 2, 1.7, 8.1e-12, -9.508510801145505e-15, 7.457655530310082e-15, -0.0023477804447272666},
        {1, 0, 1.7, 0.0, 0.0, 0.0, 0.004174520979089174},
        {1, 0, 1.7, 8.1e-12, 1.6906809965308448e-14, -1.326024311004213e-14, 0.004174520979087838},
        {1, 1, 1.7, 0.0, 0.0, 0.0, 0.002877920694890956},
        {1, 1, 1.7, 8.1e-12, 1.1655578814307088e-14, -9.14163044259203e-15, 0.0028779206948903215},
    };
    for (const auto& reference : references)
    {
        SCOPED_TRACE(::testing::Message() << reference.kind << ":" << reference.flag
                     << ":" << reference.rho << ":" << reference.gradient_squared);
        double energy = 0.0;
        double v1 = 0.0;
        double v2 = 0.0;
        if (reference.kind == 0)
        {
            XC_Functional::pbex(reference.rho, reference.gradient_squared, reference.flag, energy, v1, v2);
        }
        else
        {
            XC_Functional::pbec(reference.rho, reference.gradient_squared, reference.flag, energy, v1, v2);
        }
        EXPECT_TRUE(std::isfinite(energy));
        EXPECT_TRUE(std::isfinite(v1));
        EXPECT_TRUE(std::isfinite(v2));
        EXPECT_NEAR(energy, reference.energy, 1.e-35 + 3.e-10 * std::abs(reference.energy));
        EXPECT_NEAR(v1, reference.density_derivative, 1.e-35 + 3.e-10 * std::abs(reference.density_derivative));
        EXPECT_NEAR(v2, reference.twice_gradient_derivative,
                    1.e-35 + 3.e-10 * std::abs(reference.twice_gradient_derivative));
    }
}

TEST(XCGgaWrap, PbeFamilySmallGradientDerivative)
{
    for (const auto* name : {"PBE", "REVPBE", "PBESOL", "PBE0"})
    {
        XC_Functional::ScopedType selection(name);
        const double rho = 0.088;
        const double gradient = 1.e-10;
        const double step = 1.e-13;
        double energy = 0.0;
        double v1 = 0.0;
        double v2 = 0.0;
        double plus = 0.0;
        double minus = 0.0;
        double unused_v1 = 0.0;
        double unused_v2 = 0.0;
        XC_Functional::gcxc(rho, gradient, energy, v1, v2);
        XC_Functional::gcxc(rho, gradient + step, plus, unused_v1, unused_v2);
        XC_Functional::gcxc(rho, gradient - step, minus, unused_v1, unused_v2);
        // v2 = 2*d(E)/d(|grad rho|^2), across the former abrupt cutoff.
        EXPECT_NEAR((plus - minus) / step, v2, 1.e-11 + 1.e-7 * std::abs(v2));
    }
}

TEST(XCGgaWrap, PbesolLowDensityLargeGradientDerivative)
{
    XC_Functional::ScopedType selection("PBESOL");
    double energy = 0.0;
    double v1 = 0.0;
    double v2 = 0.0;
    XC_Functional::gcxc(1.01e-6, 0.087, energy, v1, v2);
    // Independent high-precision energy derivative, including both components.
    EXPECT_NEAR(v2, -4.070675506958874e-20, 1.2212036520881989e-29);
}

TEST(XCGgaWrap, PreserveOtherFunctionalAndDensityGuards)
{
    for (const auto* name : {"BLYP", "BP", "PW91", "WC"})
    {
        XC_Functional::ScopedType selection(name);
        for (double gradient : {0.0, 8.1e-12})
        {
            double energy = 1.0;
            double v1 = 1.0;
            double v2 = 1.0;
            XC_Functional::gcxc(1.7, gradient, energy, v1, v2);
            EXPECT_DOUBLE_EQ(energy, 0.0);
            EXPECT_DOUBLE_EQ(v1, 0.0);
            EXPECT_DOUBLE_EQ(v2, 0.0);
        }
    }
    for (const auto* name : {"PBE", "REVPBE", "PBESOL", "PBE0"})
    {
        XC_Functional::ScopedType selection(name);
        for (double rho : {-1.0, 0.0, 1.e-10, 1.e-6})
        {
            double energy = 1.0;
            double v1 = 1.0;
            double v2 = 1.0;
            XC_Functional::gcxc(rho, 1.7, energy, v1, v2);
            EXPECT_DOUBLE_EQ(energy, 0.0);
            EXPECT_DOUBLE_EQ(v1, 0.0);
            EXPECT_DOUBLE_EQ(v2, 0.0);
        }
    }
}
