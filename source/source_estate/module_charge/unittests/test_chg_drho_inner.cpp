#include "gtest/gtest.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_charge/chg_drho.h"
#include "source_estate/module_charge/chg_drho_detail.h"
#include "source_estate/module_charge/chg_mix_cfg.h"

#include <cmath>
#include <complex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/************************************************
 *  unit test of module_charge/chg_drho_inner.cpp
 ***********************************************/

/**
 * - Tested Functions:
 *   - inner_product_recip_rho (module_charge::detail):
 *     Coulomb-metric reciprocal inner product of the charge residual
 *     - nspin == 1: sum over G!=0 of conj(rho1)*rho2 / gg, weighted by fac
 *   - inner_product_recip_hartree:
 *     Hartree-like reciprocal inner product used in charge mixing
 *     - nspin == 1: same Coulomb sum as inner_product_recip_rho
 *
 * Both are tested with a single nonzero G component (ig=1, assuming ig0=0)
 * so the analytic value is a single term.
 */

namespace
{

MixingConfig make_cfg(int nspin)
{
    MixingConfig cfg{
        "plain",  // mixing_mode
        0.8,      // mixing_beta
        4,        // mixing_ndim
        0.0,      // mixing_gg0
        false,    // mixing_tau
        1.6,      // mixing_beta_mag
        0.0,      // mixing_gg0_mag
        0.1,      // mixing_gg0_min
        -10.0,    // mixing_angle
        false,    // mixing_dmr
        nspin,    // nspin
        2,        // scf_thr_type
        false,    // double_grid
        false,    // gamma_only_pw
        false,    // domag
        false,    // domag_z
        100       // scf_nmax
    };
    return cfg;
}

} // namespace

class ChgDrhoInnerTest : public ::testing::Test
{
  protected:
    ModulePW::PW_Basis pw_basis;

    void SetUp() override
    {
        pw_basis.initgrids(4, ModuleBase::Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1), 20);
        pw_basis.initparameters(false, 20);
        pw_basis.setuptransform();
        pw_basis.collect_local_pw();
    }
};

TEST_F(ChgDrhoInnerTest, InnerProductRecipRhoNspin1SingleG)
{
    const int nspin = 1;
    MixingConfig cfg = make_cfg(nspin);

    // place a single nonzero component at ig=1 (ig0 == 0 is the G=0 vector)
    const int ig = 1;
    std::vector<std::complex<double>> rho1(nspin * pw_basis.npw, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> rho2(nspin * pw_basis.npw, std::complex<double>(0.0, 0.0));
    rho1[ig] = std::complex<double>(2.0, 1.0);
    rho2[ig] = std::complex<double>(3.0, -1.0);

    const double omega = 1.0;
    const double tpiba = 1.0;
    const double inner = module_charge::detail::inner_product_recip_rho(
        rho1.data(), rho2.data(), pw_basis, cfg, omega, tpiba);

    const double fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (tpiba * tpiba);
    const double gg = pw_basis.gg[ig];
    // (conj(2+i) * (3-i)).real() = (2-i)*(3-i) = 6 -2i -3i + i^2 = 5 -5i, real = 5
    const double overlap = (std::conj(rho1[ig]) * rho2[ig]).real();
    const double ref = fac * overlap / gg * omega * 0.5;

    EXPECT_NEAR(inner, ref, 1e-8);
}

TEST_F(ChgDrhoInnerTest, InnerProductRecipHartreeNspin1SingleG)
{
    const int nspin = 1;
    MixingConfig cfg = make_cfg(nspin);

    const int ig = 1;
    std::vector<std::complex<double>> rhog1(nspin * pw_basis.npw, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> rhog2(nspin * pw_basis.npw, std::complex<double>(0.0, 0.0));
    rhog1[ig] = std::complex<double>(1.0, 0.0);
    rhog2[ig] = std::complex<double>(2.0, 0.0);

    const double omega = 1.0;
    const double tpiba = 1.0;
    const double inner = module_charge::inner_product_recip_hartree(
        rhog1.data(), rhog2.data(), pw_basis, cfg, omega, tpiba);

    const double fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (tpiba * tpiba);
    const double gg = pw_basis.gg[ig];
    const double overlap = (std::conj(rhog1[ig]) * rhog2[ig]).real();
    const double ref = fac * overlap / gg * omega * 0.5;

    EXPECT_NEAR(inner, ref, 1e-8);
}

TEST_F(ChgDrhoInnerTest, InnerProductRecipRhoNspin1ZeroInput)
{
    const int nspin = 1;
    MixingConfig cfg = make_cfg(nspin);

    std::vector<std::complex<double>> rho1(nspin * pw_basis.npw, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> rho2(nspin * pw_basis.npw, std::complex<double>(0.0, 0.0));

    const double inner = module_charge::detail::inner_product_recip_rho(
        rho1.data(), rho2.data(), pw_basis, cfg, 1.0, 1.0);

    EXPECT_NEAR(inner, 0.0, 1e-12);
}

namespace
{
#ifdef _OPENMP
class OpenMPSettingsGuard
{
  public:
    OpenMPSettingsGuard()
    {
        threads_ = omp_get_max_threads();
        dynamic_ = omp_get_dynamic();
        omp_set_dynamic(0);
    }
    ~OpenMPSettingsGuard()
    {
        omp_set_num_threads(threads_);
        omp_set_dynamic(dynamic_);
    }

  private:
    int threads_;
    int dynamic_;
};
#endif
} // namespace

class ChgDrhoInnerParallelTest : public ChgDrhoInnerTest
{
  protected:
    void SetUp() override
    {
        pw_basis.initgrids(16, ModuleBase::Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1), 20);
        pw_basis.initparameters(false, 20);
        pw_basis.setuptransform();
        pw_basis.collect_local_pw();
    }
};

TEST_F(ChgDrhoInnerParallelTest, ReproducibleNonmagneticInnerProducts)
{
    ASSERT_GT(pw_basis.npw, 4096);
#ifdef _OPENMP
    OpenMPSettingsGuard restore;
    const std::vector<int> team_sizes{1, 2, 8};
#else
    const std::vector<int> team_sizes{1};
#endif
    for (const int nspin : {1, 4})
    {
        const MixingConfig cfg = make_cfg(nspin);
        // Nonzero imaginary parts and unequal operands exercise the full real
        // complex overlap. A self product also checks the residual norm path.
        std::vector<std::complex<double>> rho1(nspin * pw_basis.npw);
        std::vector<std::complex<double>> rho2(nspin * pw_basis.npw);
        for (int ig = 0; ig < nspin * pw_basis.npw; ++ig)
        {
            rho1[ig] = std::complex<double>(1.0 / (1 + ig % 23), -0.3 / (1 + ig % 17));
            rho2[ig] = std::complex<double>(-0.7 / (1 + ig % 19), 0.2 / (1 + ig % 29));
        }
        for (const bool self_product : {false, true})
        {
            SCOPED_TRACE(nspin);
            SCOPED_TRACE(self_product);
            const auto* second = self_product ? rho1.data() : rho2.data();
            long double reference = 0.0L;
            for (int ig = 0; ig < pw_basis.npw; ++ig)
            {
                if (ig == pw_basis.ig_gge0)
                {
                    continue;
                }
                const long double real_product = static_cast<long double>(rho1[ig].real()) * second[ig].real();
                const long double imag_product = static_cast<long double>(rho1[ig].imag()) * second[ig].imag();
                reference += (real_product + imag_product) / pw_basis.gg[ig];
            }
            reference *= ModuleBase::e2 * ModuleBase::FOUR_PI * 0.5L;
            const double expected = static_cast<double>(reference);
            const double tolerance = std::abs(expected) * 1e-12;
            double first = 0.0;
            for (const int team_size : team_sizes)
            {
#ifdef _OPENMP
                omp_set_num_threads(team_size);
                int actual_team = 0;
#pragma omp parallel
                {
#pragma omp single
                    actual_team = omp_get_num_threads();
                }
                ASSERT_EQ(actual_team, team_size);
#endif
                SCOPED_TRACE(team_size);
                for (int repeat = 0; repeat < 16; ++repeat)
                {
                    const double rho = module_charge::detail::inner_product_recip_rho(
                        rho1.data(), second, pw_basis, cfg, 1.0, 1.0);
                    const double hartree = module_charge::inner_product_recip_hartree(
                        rho1.data(), second, pw_basis, cfg, 1.0, 1.0);
                    if (team_size == 1 && repeat == 0)
                    {
                        first = rho;
                    }
                    EXPECT_NEAR(rho, expected, tolerance);
                    EXPECT_EQ(rho, first);
                    EXPECT_EQ(hartree, first);
                }
            }
        }
    }
}
