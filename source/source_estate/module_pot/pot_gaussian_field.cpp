#include "pot_gaussian_field.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_cell/cell_tools.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <stdexcept>

namespace
{
void reduce_forces(std::vector<ModuleBase::Vector3<double>>& forces)
{
    for (auto& force : forces)
    {
        double values[3] = {force.x, force.y, force.z};
        Parallel_Reduce::reduce_pool(values, 3);
        force.x = values[0];
        force.y = values[1];
        force.z = values[2];
    }
}
} // namespace

namespace elecstate
{
PotGaussianField::PotGaussianField(const ModulePW::PW_Basis* basis,
                                   const std::string& filename,
                                   const int kmax,
                                   const double electron_count,
                                   std::ostream& running_log)
    : kmax_(kmax), electron_count_(electron_count)
{
    this->rho_basis_ = basis;
    this->dynamic_mode = true;
    if (basis == nullptr || kmax < 1 || kmax > 32 || !std::isfinite(electron_count) || electron_count < 0.0)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Invalid Gaussian field basis, reciprocal limit or electron count");
    }
    char full_path[PATH_MAX];
    const char* resolved = ::realpath(filename.c_str(), full_path);
    if (resolved == nullptr)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Cannot resolve Gaussian external charge file: " + filename);
    }
    std::ifstream input(full_path);
    try
    {
        environment_ = read_gaussian_external_charges(input);
    }
    catch (const std::exception& error)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", error.what());
    }
    std::cout << "Read Gaussian external charges: " << full_path << '\n';
    running_log << "Read Gaussian external charges: " << full_path << '\n';
}

void PotGaussianField::prepare_ions(const UnitCell& cell,
                                    std::vector<ModuleBase::Vector3<double>>& positions,
                                    std::vector<double>& charges) const
{
    const auto& lattice = cell.latvec;
    const double off_diagonal = std::abs(lattice.e12) + std::abs(lattice.e13) + std::abs(lattice.e21)
                              + std::abs(lattice.e23) + std::abs(lattice.e31) + std::abs(lattice.e32);
    if (off_diagonal > 1.0e-12 || lattice.e11 <= 0.0 || lattice.e22 <= 0.0 || lattice.e33 <= 0.0
        || !std::isfinite(cell.omega) || cell.omega <= 0.0)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Gaussian field currently requires a positive orthorhombic fixed cell");
    }
    for (int type = 0; type < cell.ntype; ++type)
    {
        if (cell.atoms[type].ncpp.tvanp)
        {
            ModuleBase::WARNING_QUIT("PotGaussianField", "Gaussian field requires norm-conserving pseudopotentials");
        }
    }
    const auto atoms = unitcell::get_atom_data(cell.atoms, cell.ntype, cell.lat0);
    double ionic_charge = 0.0;
    for (const auto& atom : atoms)
    {
        positions.push_back(atom.position);
        charges.push_back(atom.valence_charge);
        ionic_charge += atom.valence_charge;
    }
    if (electron_count_ != 0.0 && std::abs(ionic_charge - electron_count_) > 1.0e-10)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Periodic QM must be neutral; no compensating background is applied");
    }
}

void PotGaussianField::prepare_modes(const UnitCell& cell,
                                     std::vector<GaussianFieldMode>& modes,
                                     std::vector<int>& indices) const
{
    const auto& basis = *this->rho_basis_;
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        const auto& index = basis.gdirect[ig];
        if (std::abs(index.x) > kmax_ + 1.0e-8 || std::abs(index.y) > kmax_ + 1.0e-8
            || std::abs(index.z) > kmax_ + 1.0e-8 || basis.gg[ig] < 1.0e-16)
        {
            continue;
        }
        const auto vector = basis.gcar[ig] * basis.tpiba;
        const double square = vector * vector;
        const double coefficient = 8.0 * ModuleBase::PI / (cell.omega * square);
        modes.push_back({vector, coefficient});
        indices.push_back(ig);
    }
    int mode_count = static_cast<int>(modes.size());
    Parallel_Reduce::reduce_pool(&mode_count, 1);
    const int side = 2 * kmax_ + 1;
    if (mode_count != side * side * side - 1)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Density basis omits Gaussian field modes; increase density cutoff or grid");
    }
}

void PotGaussianField::cal_v_eff(const Charge* charge, const UnitCell* cell, ModuleBase::matrix& potential)
{
    ModuleBase::timer::start("PotGaussianField", "cal_v_eff");
    const auto& basis = *this->rho_basis_;
    if (charge == nullptr || cell == nullptr || charge->nspin != 1 || basis.gamma_only)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Gaussian field requires nspin 1 and complete signed reciprocal storage");
    }
    std::vector<ModuleBase::Vector3<double>> positions;
    std::vector<double> charges;
    this->prepare_ions(*cell, positions, charges);
    double ionic_charge = 0.0;
    for (const double value : charges) { ionic_charge += value; }
    double integrated = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir) { integrated += charge->rho[0][ir]; }
    integrated *= cell->omega / basis.nxyz;
    Parallel_Reduce::reduce_pool(integrated);
    if (!std::isfinite(integrated) || std::abs(integrated - ionic_charge) > 1.0e-6)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Periodic QM density does not integrate to its neutral electron count");
    }
    std::vector<GaussianFieldMode> modes;
    std::vector<int> indices;
    this->prepare_modes(*cell, modes, indices);
    std::vector<std::complex<double>> density(basis.npw);
    auto* transform = const_cast<ModulePW::PW_Basis*>(this->rho_basis_);
    transform->real2recip(charge->rho[0], density.data());
    std::vector<std::complex<double>> selected_density;
    for (const int index : indices) { selected_density.push_back(density[index] * cell->omega); }
    try
    {
        result_ = evaluate_gaussian_charge_field(modes, environment_, positions, charges, selected_density);
    }
    catch (const std::exception& error)
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", error.what());
    }
    Parallel_Reduce::reduce_pool(result_.ionic_energy);
    Parallel_Reduce::reduce_pool(result_.electronic_energy);
    reduce_forces(result_.ionic_force);
    reduce_forces(result_.environment_force);
    std::vector<std::complex<double>> reciprocal(basis.npw, 0.0);
    for (std::size_t i = 0; i < indices.size(); ++i) { reciprocal[indices[i]] = result_.electron_potential[i]; }
    std::vector<double> real_potential(basis.nrxx);
    transform->recip2real(reciprocal.data(), real_potential.data());
    for (int ir = 0; ir < basis.nrxx; ++ir) { potential(0, ir) += real_potential[ir]; }
    ModuleBase::timer::end("PotGaussianField", "cal_v_eff");
}

double PotGaussianField::get_energy() const
{
    return result_.ionic_energy + result_.electronic_energy;
}

void PotGaussianField::add_force(const UnitCell& cell, ModuleBase::matrix& force) const
{
    if (result_.ionic_force.size() != static_cast<std::size_t>(cell.nat))
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Gaussian field ionic force is unavailable");
    }
    for (int i = 0; i < cell.nat; ++i)
    {
        force(i, 0) += result_.ionic_force[i].x;
        force(i, 1) += result_.ionic_force[i].y;
        force(i, 2) += result_.ionic_force[i].z;
    }
}

void PotGaussianField::write_result(std::ostream& output) const
{
    if (result_.environment_force.size() != environment_.size())
    {
        ModuleBase::WARNING_QUIT("PotGaussianField", "Gaussian environment force is unavailable");
    }
    const auto flags = output.flags();
    const auto precision = output.precision();
    output << std::scientific << std::setprecision(16);
    output << "QMDPA_GAUSSIAN_FIELD_V1_RESULT\n";
    output << "reciprocal_kmax " << kmax_ << '\n';
    output << "ionic_energy_Ry " << result_.ionic_energy << '\n';
    output << "electronic_energy_Ry " << result_.electronic_energy << '\n';
    output << "environment_count " << environment_.size() << '\n';
    output << "environment_forces_Ry_per_Bohr\n";
    for (std::size_t j = 0; j < environment_.size(); ++j)
    {
        const auto& force = result_.environment_force[j];
        output << environment_[j].id << ' ' << force.x << ' ' << force.y << ' ' << force.z << '\n';
    }
    output << "END_QMDPA_GAUSSIAN_FIELD_V1_RESULT\n";
    output.flags(flags);
    output.precision(precision);
}
} // namespace elecstate
