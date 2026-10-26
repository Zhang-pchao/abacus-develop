#ifndef POT_GAUSSIAN_FIELD_H
#define POT_GAUSSIAN_FIELD_H

#include "pot_base.h"
#include "source_estate/gaussian_charge_field.h"

#include <iosfwd>
#include <string>

namespace elecstate
{
// Fixed-cell, neutral Gaussian environment for periodic CPU PW ground states.
// Added as a dynamic component so the complete cross energy is restored once
// after the normal density-potential subtraction from the band energy.
class PotGaussianField : public PotBase
{
  public:
    PotGaussianField(const ModulePW::PW_Basis* basis,
                     const std::string& filename,
                     int kmax,
                     double electron_count,
                     std::ostream& running_log);
    void cal_v_eff(const Charge* charge, const UnitCell* cell, ModuleBase::matrix& potential) override;
    double get_energy() const override;
    void add_force(const UnitCell& cell, ModuleBase::matrix& force) const;
    void write_result(std::ostream& output) const;

  private:
    void prepare_ions(const UnitCell& cell,
                      std::vector<ModuleBase::Vector3<double>>& positions,
                      std::vector<double>& charges) const;
    void prepare_modes(const UnitCell& cell,
                       std::vector<GaussianFieldMode>& modes,
                       std::vector<int>& indices) const;
    const int kmax_;
    const double electron_count_;
    std::vector<GaussianExternalCharge> environment_;
    GaussianFieldResult result_;
};
} // namespace elecstate

#endif
