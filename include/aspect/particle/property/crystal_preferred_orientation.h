/*
 Copyright (C) 2022 - 2024 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file LICENSE.  If not see
 <http://www.gnu.org/licenses/>.
 */

#ifndef _aspect_particle_property_cpo_h
#define _aspect_particle_property_cpo_h

#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>
#include <array>
#include <aspect/material_model/rheology/diffusion_creep.h>
#include <aspect/material_model/rheology/dislocation_creep.h>
#include <aspect/material_model/rheology/visco_plastic.h>
#include <aspect/material_model/utilities.h>
#include <aspect/material_model/interface.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <boost/random.hpp>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      /**
       * @brief The type of deformation used by the CPO code.
       *
       * passive: Only to be used with the spin tensor CPO Derivative algorithm.
       * olivine_a_fabric to olivine_e_fabric: Only to be used with the D-Rex CPO Derivative algorithm.
       *  Sets the deformation type of the mineral to a Olivine A-E Fabric, which influences the relative strength of the slip planes. See table 1 in Fraters and Billen (2021).
       * enstatite: Only to be used with the D-Rex CPO Derivative algorithm. Sets the deformation type of the mineral to a enstatite Fabric, which influences the relative strength of the slip planes.
       */
      enum class DeformationType
      {
        passive, olivine_a_fabric, olivine_b_fabric, olivine_c_fabric, olivine_d_fabric, olivine_e_fabric, enstatite, clinopyroxene, olivine_d_0kl
      };


      /**
       * @brief The type of deformation selector used by the CPO code.
       *
       * The selector is a input parameter and it can either set a deformation type directly or determine the deformation type through an algorithm.
       * The deformation type selector is used to determine/select the deformation type. It can be a fixed deformation type, for example,
       * by setting it to olivine_a_fabric, or it can be dynamically chosen, which is what the olivine_karato_2008 option does.
       *
       * passive: Only to be used with the spin tensor CPO Derivative algorithm.
       * olivine_a_fabric to olivine_e_fabric: Only to be used with the D-Rex CPO Derivative algorithm.
       *  Sets the deformation type of the mineral to a Olivine A-E Fabric, which influences the relative strength of the slip planes. See table 1 in Fraters and Billen (2021).
       * enstatite: Only to be used with the D-Rex CPO Derivative algorithm. Sets the deformation type of the mineral to a enstatite Fabric, which influences the relative strength of the slip planes.
       * olivine_karato_2008: Only to be used with the D-Rex CPO Derivative algorithm. Sets the deformation type of the mineral to a olivine fabric based on the table in Karato 2008.
       */
      enum class DeformationTypeSelector
      {
        passive, olivine_a_fabric, olivine_b_fabric, olivine_c_fabric, olivine_d_fabric, olivine_e_fabric, enstatite, olivine_karato_2008, clinopyroxene, olivine_d_0kl
      };

      /**
       * @brief The type of Advection method used to advect the CPO properties.
       */
      enum class AdvectionMethod
      {
        forward_euler, backward_euler, exponential_update
      };

      /**
       * @brief The algorithm used to compute the derivatives of the grain size and rotation matrix used in the advection.
       *
       * spin_tensor: Rotates the CPO properties soly with the rotation of the particle itself.
       * drex_2004: Rotates the CPO properties based on the D-Rex 2004 algorithm.
       */
      enum class CPODerivativeAlgorithm
      {
        spin_tensor, drex_2004, drexpp
      };

      /**
       * @brief An enum used to determine how the initial grain sizes and orientations are set for all particles
       *
       * uniform_grains_and_random_uniform_rotations: all particles are set to a uniform grain-size of 1/n_grains
       * world_builder: all particle grain-sizes and orientations are set by the world builder.
       */
      enum class CPOInitialGrainsModel
      {
        uniform_grains_and_random_uniform_rotations, world_builder, pre_existing_fabric, normal_distribution_grainsize_and_random_initial_orientations
      };

      /**
       * The plugin manages and computes the evolution of Lattice/Crystal Preferred Orientations (LPO/CPO)
       * on particles. Each ASPECT particle represents many grains. Each grain is assigned a size and a orientation
       * matrix. This allows tracking the LPO evolution with kinematic polycrystal CPO evolution models such
       * as D-Rex (Kaminski and Ribe, 2001; Kaminski et al., 2004).
       *
       * This plugin stores M minerals and for each mineral it stores N grains.
       * The total memory per particle is M * (13 + N * 30) doubles.
       *
       * Memory layout:
       *   mineral block stride = n_grains * 30 + 13
       *   grain block start    = 13
       *   grain stride         = 30
       *
       * Mineral-level slots [0..12] — one per mineral:
       *   [0]  deformation type          => 0  + mineral_i * (n_grains * 30 + 13)
       *   [1]  volume fraction mineral   => 1  + mineral_i * (n_grains * 30 + 13)
       *   [2]  bulk rx grain size        => 2  + mineral_i * (n_grains * 30 + 13)
       *   [3]  mean grain size           => 3  + mineral_i * (n_grains * 30 + 13)
       *   [4]  mean mechanism ratio      => 4  + mineral_i * (n_grains * 30 + 13)
       *   [5]  n_shrink_dead             => 5  + mineral_i * (n_grains * 30 + 13)
       *   [6]  n_curv_dead               => 6  + mineral_i * (n_grains * 30 + 13)
       *   [7]  n_floor                   => 7  + mineral_i * (n_grains * 30 + 13)
       *   [8]  n_ceiling                 => 8  + mineral_i * (n_grains * 30 + 13)
       *   [9]  n_rx_skip                 => 9  + mineral_i * (n_grains * 30 + 13)
       *   [10] n_alive                   => 10 + mineral_i * (n_grains * 30 + 13)
       *   [11] kappa (2*M*gamma)         => 11 + mineral_i * (n_grains * 30 + 13)
       *   [12] Wk (kinematic vorticity)  => 12 + mineral_i * (n_grains * 30 + 13)
       *
       * Grain-level slots [+0..+29] — absolute offset = 13 + relative + grain_i*30 + mineral_i*(n_grains*30+13):
       *   [+0]     volume fraction (grain size d)
       *   [+1..+9] rotation matrix (9 doubles, Tensor<2,3>)
       *   [+10]    grain status
       *   [+11]    strain rate ratio
       *   [+12]    rx fraction
       *   [+13]    active slip system
       *   [+14]    strain rate
       *   [+15]    strain energy
       *   [+16]    surface energy
       *   [+17]    differential stress
       *   [+18]    grain size change
       *   [+19]    dislocation density
       *   [+20]    strain accumulated
       *   [+21]    J (stiffness ratio)
       *   [+22]    tau (relaxation timescale)
       *   [+23]    A_drive (total frozen drive)
       *   [+24]    theta (misorientation angle this step)
       *   [+25..+27] rotation_axis (Tensor<1,3>, 3 doubles)
       *   [+28]    theta_dot (instantaneous spin rate)
       *   [+29]    psi (fabric angle)
       *
       * Last used data entry is n_minerals * (n_grains * 30 + 13).
       *
       * @ingroup ParticleProperties
       */
      template <int dim>
      class CrystalPreferredOrientation : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * Constructor
           */
          CrystalPreferredOrientation() = default;

          /**
           * Initialization function. This function is called once at the
           * beginning of the program after parse_parameters is run.
           */
          void
          initialize () override;

          /**
           * Initialization function. This function is called once at the
           * creation of every particle for every property to initialize its
           * value.
           *
           * @param [in] position The current particle position.
           * @param [in,out] particle_properties The properties of the particle
           * that is initialized within the call of this function. The purpose
           * of this function should be to extend this vector by a number of
           * properties.
           */
          void
          initialize_one_particle_property (const Point<dim> &position,
                                            std::vector<double> &particle_properties) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::update_particle_properties()
           */
          void
          update_particle_properties (const ParticleUpdateInputs<dim> &inputs,
                                      typename ParticleHandler<dim>::particle_iterator_range &particles) const override;

          /**
           * This implementation tells the particle manager that
           * we need to update particle properties every time step.
           */
          UpdateTimeFlags
          need_update () const override;

          /**
           * The CPO of late particles is initialized by interpolating from existing particles.
           */
          InitializationModeForLateParticles
          late_initialization_mode () const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::get_update_flags()
           */
          UpdateFlags
          get_update_flags (const unsigned int component) const override;

          /**
           * Set up the information about the names and number of components
           * this property requires.
           *
           * @return A vector that contains pairs of the property names and the
           * number of components this property plugin defines.
           */
          std::vector<std::pair<std::string, unsigned int>>
          get_property_information() const override;

          /**
           * @brief Computes the volume fraction and grain orientation derivatives of all the grains of a mineral.
           */
          std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
          compute_derivatives(const unsigned int cpo_index,
                              const ArrayView<double> &data,
                              const unsigned int mineral_i,
                              const SymmetricTensor<2,3> &strain_rate_3d,
                              const Tensor<2,3> &velocity_gradient_tensor,
                              const Point<dim> &position,
                              const typename DoFHandler<dim>::active_cell_iterator &cell,
                              const double temperature,
                              const double pressure,
                              const Tensor<1,dim> &velocity,
                              const std::vector<double> &compositions,
                              const SymmetricTensor<2,dim> &strain_rate,
                              const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                              const double water_content) const;

          /**
           * @brief Computes the CPO derivatives with the D-Rex 2004 algorithm.
           */
          std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
          compute_derivatives_drex_2004(const DeformationType deformation_type,
                                        const unsigned int cpo_index,
                                        const ArrayView<double> &data,
                                        const unsigned int mineral_i,
                                        const SymmetricTensor<2,3> &strain_rate_3d,
                                        const Tensor<2,3> &velocity_gradient_tensor,
                                        const std::array<double,4> ref_resolved_shear_stress,
                                        const bool prevent_nondimensionalization = false) const;

          /**
           * @brief The dynamic recrystallization module called in drexpp.
           */
          void
          recrystalize_grains(const unsigned int cpo_index,
                              const ArrayView<double> &data,
                              const unsigned int mineral_i,
                              const std::vector<double> &recrystalized_fraction,
                              const std::vector<Tensor<1,3>> &sgr_rotation_axis,
                              std::vector<double> &piezometer,
                              std::vector<bool> &rx_now) const;

          /**
           * @brief Computes the CPO derivatives with the D-Rex++ algorithm.
           */
          std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
          compute_derivatives_drexpp(const unsigned int cpo_index,
                                     const ArrayView<double> &data,
                                     const unsigned int mineral_i,
                                     const SymmetricTensor<2,3> &strain_rate_3d,
                                     const Tensor<2,3> &velocity_gradient_tensor,
                                     const std::array<double,4> ref_resolved_shear_stress,
                                     const double diffusion_pre_strainrate,
                                     const SymmetricTensor<2,dim> &deviatoric_stress,
                                     const double diffusion_grain_size_exponent,
                                     const double dislocation_factor,
                                     const double dislocation_stress_exponent,
                                     const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                                     const double temperature,
                                     const double pressure) const;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm) override;

          /**
           * Return the number of grains per particle
           */
          unsigned int
          get_number_of_grains() const;

          /**
           * Return the number of minerals per particle
           */
          unsigned int
          get_number_of_minerals() const;

          /**
           * @brief Determines the deformation type from the deformation type selector.
           */
          DeformationType
          determine_deformation_type(const DeformationTypeSelector deformation_type_selector,
                                     const Point<dim> &position,
                                     const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     const double temperature,
                                     const double pressure,
                                     const Tensor<1,dim> &velocity,
                                     const std::vector<double> &compositions,
                                     const SymmetricTensor<2,dim> &strain_rate,
                                     const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                                     const double water_content) const;

          /**
           * @brief Computes the deformation type given the stress and water content according to the
           * table in Karato 2008.
           */
          DeformationType
          determine_deformation_type_karato_2008(const double stress,
                                                 const double water_content) const;

          /**
           * @brief Computes the reference resolved shear stress (RRSS) based on the selected deformation type.
           */
          std::array<double,4>
          reference_resolved_shear_stress_from_deformation_type(DeformationType deformation_type,
                                                                double max_value = 1e60) const;

          // ═══════════════════════════════════════════════════════════════════
          // Getter / Setter functions
          // All use: cpo_data_position + offset + grain_i * 30 + mineral_i * (n_grains * 30 + 13)
          // Mineral-level offsets: 0..12 (no grain_i term)
          // Grain-level offsets:   13 + relative_offset (0..29) + grain_i * 30
          // ═══════════════════════════════════════════════════════════════════

          // ── Mineral-level ──────────────────────────────────────────────────

          /**
           * @brief Returns the deformation type for a mineral.
           */
          inline
          DeformationType get_deformation_type(const unsigned int cpo_data_position,
                                               const ArrayView<double> &data,
                                               const unsigned int mineral_i) const
          {
            return static_cast<DeformationType>(data[cpo_data_position + 0 + mineral_i * (n_grains * 30 + 13)]);
          }

          /**
           * @brief Sets the deformation type for a mineral.
           */
          inline
          void set_deformation_type(const unsigned int cpo_data_position,
                                    const ArrayView<double> &data,
                                    const unsigned int mineral_i,
                                    const DeformationType deformation_type) const
          {
            data[cpo_data_position + 0 + mineral_i * (n_grains * 30 + 13)] = static_cast<double>(deformation_type);
          }

          /**
           * @brief Returns the volume fraction of a mineral.
           */
          inline
          double get_volume_fraction_mineral(const unsigned int cpo_data_position,
                                             const ArrayView<double> &data,
                                             const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 1 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the volume fraction of a mineral.
           */
          inline
          void set_volume_fraction_mineral(const unsigned int cpo_data_position,
                                           const ArrayView<double> &data,
                                           const unsigned int mineral_i,
                                           const double volume_fraction_mineral) const
          {
            data[cpo_data_position + 1 + mineral_i * (n_grains * 30 + 13)] = volume_fraction_mineral;
          }

          /**
           * @brief Returns the bulk recrystallization grain size for a mineral.
           */
          inline
          double get_bulk_recrystalization_grain_size_mineral(const unsigned int cpo_data_position,
                                                              const ArrayView<double> &data,
                                                              const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 2 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the bulk recrystallization grain size for a mineral.
           */
          inline
          void set_bulk_recrystalization_grain_size_mineral(const unsigned int cpo_data_position,
                                                            const ArrayView<double> &data,
                                                            const unsigned int mineral_i,
                                                            const double recrystalization_size) const
          {
            data[cpo_data_position + 2 + mineral_i * (n_grains * 30 + 13)] = recrystalization_size;
          }

          /**
           * @brief Returns the harmonic mean grain size for a mineral.
           */
          inline
          double get_mean_grain_size_mineral(const unsigned int cpo_data_position,
                                             const ArrayView<double> &data,
                                             const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 3 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the harmonic mean grain size for a mineral.
           */
          inline
          void set_mean_grain_size_mineral(const unsigned int cpo_data_position,
                                           const ArrayView<double> &data,
                                           const unsigned int mineral_i,
                                           const double mean_grain_size) const
          {
            data[cpo_data_position + 3 + mineral_i * (n_grains * 30 + 13)] = mean_grain_size;
          }

          /**
           * @brief Returns the area-weighted mean mechanism ratio (diffusion fraction) for a mineral.
           */
          inline
          double get_mean_mechanism_ratio_mineral(const unsigned int cpo_data_position,
                                                  const ArrayView<double> &data,
                                                  const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 4 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the area-weighted mean mechanism ratio for a mineral.
           */
          inline
          void set_mean_mechanism_ratio_mineral(const unsigned int cpo_data_position,
                                                const ArrayView<double> &data,
                                                const unsigned int mineral_i,
                                                const double mean_mechanism_ratio_mineral) const
          {
            data[cpo_data_position + 4 + mineral_i * (n_grains * 30 + 13)] = mean_mechanism_ratio_mineral;
          }

          /**
           * @brief Returns the number of grains killed by strain collapse (A <= 0) this step.
           */
          inline
          double get_n_shrink_dead(const unsigned int cpo_data_position,
                                   const ArrayView<const double> &data,
                                   const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 5 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the number of grains killed by strain collapse (A <= 0) this step.
           */
          inline
          void set_n_shrink_dead(const unsigned int cpo_data_position,
                                 const ArrayView<double> &data,
                                 const unsigned int mineral_i,
                                 const double n_shrink_dead) const
          {
            data[cpo_data_position + 5 + mineral_i * (n_grains * 30 + 13)] = n_shrink_dead;
          }

          /**
           * @brief Returns the number of grains killed by curvature collapse (disc < 0) this step.
           */
          inline
          double get_n_curv_dead(const unsigned int cpo_data_position,
                                 const ArrayView<const double> &data,
                                 const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 6 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the number of grains killed by curvature collapse (disc < 0) this step.
           */
          inline
          void set_n_curv_dead(const unsigned int cpo_data_position,
                               const ArrayView<double> &data,
                               const unsigned int mineral_i,
                               const double n_curv_dead) const
          {
            data[cpo_data_position + 6 + mineral_i * (n_grains * 30 + 13)] = n_curv_dead;
          }

          /**
           * @brief Returns the number of grains retired by the sub-resolution floor this step.
           */
          inline
          double get_n_floor(const unsigned int cpo_data_position,
                             const ArrayView<const double> &data,
                             const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 7 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the number of grains retired by the sub-resolution floor this step.
           */
          inline
          void set_n_floor(const unsigned int cpo_data_position,
                           const ArrayView<double> &data,
                           const unsigned int mineral_i,
                           const double n_floor) const
          {
            data[cpo_data_position + 7 + mineral_i * (n_grains * 30 + 13)] = n_floor;
          }

          /**
           * @brief Returns the number of grains capped at the physical ceiling this step.
           */
          inline
          double get_n_ceiling(const unsigned int cpo_data_position,
                               const ArrayView<const double> &data,
                               const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 8 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the number of grains capped at the physical ceiling this step.
           */
          inline
          void set_n_ceiling(const unsigned int cpo_data_position,
                             const ArrayView<double> &data,
                             const unsigned int mineral_i,
                             const double n_ceiling) const
          {
            data[cpo_data_position + 8 + mineral_i * (n_grains * 30 + 13)] = n_ceiling;
          }

          /**
           * @brief Returns the number of freshly nucleated grains skipped by GBM this step.
           */
          inline
          double get_n_rx_skip(const unsigned int cpo_data_position,
                               const ArrayView<const double> &data,
                               const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 9 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the number of freshly nucleated grains skipped by GBM this step.
           */
          inline
          void set_n_rx_skip(const unsigned int cpo_data_position,
                             const ArrayView<double> &data,
                             const unsigned int mineral_i,
                             const double n_rx_skip) const
          {
            data[cpo_data_position + 9 + mineral_i * (n_grains * 30 + 13)] = n_rx_skip;
          }

          /**
           * @brief Returns the number of grains that survived this step.
           */
          inline
          double get_n_alive(const unsigned int cpo_data_position,
                             const ArrayView<const double> &data,
                             const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 10 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the number of grains that survived this step.
           */
          inline
          void set_n_alive(const unsigned int cpo_data_position,
                           const ArrayView<double> &data,
                           const unsigned int mineral_i,
                           const double n_alive) const
          {
            data[cpo_data_position + 10 + mineral_i * (n_grains * 30 + 13)] = n_alive;
          }

          /**
           * @brief Returns the curvature coefficient kappa = 2*M*gamma for this mineral (same for all grains).
           */
          inline
          double get_kappa(const unsigned int cpo_data_position,
                           const ArrayView<const double> &data,
                           const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 11 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the curvature coefficient kappa = 2*M*gamma for this mineral.
           */
          inline
          void set_kappa(const unsigned int cpo_data_position,
                         const ArrayView<double> &data,
                         const unsigned int mineral_i,
                         const double kappa) const
          {
            data[cpo_data_position + 11 + mineral_i * (n_grains * 30 + 13)] = kappa;
          }

          /**
           * @brief Returns the kinematic vorticity number Wk = ||W||_F / ||D||_F for this mineral.
           */
          inline
          double get_Wk(const unsigned int cpo_data_position,
                        const ArrayView<const double> &data,
                        const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 12 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the kinematic vorticity number Wk for this mineral.
           */
          inline
          void set_Wk(const unsigned int cpo_data_position,
                      const ArrayView<double> &data,
                      const unsigned int mineral_i,
                      const double Wk) const
          {
            data[cpo_data_position + 12 + mineral_i * (n_grains * 30 + 13)] = Wk;
          }
          
          /**
           * @brief Returns the kinematic vorticity number Wk = ||W||_F / ||D||_F for this mineral.
           */
          inline
          double get_sauter_mean(const unsigned int cpo_data_position,
                        const ArrayView<const double> &data,
                        const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 12 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the kinematic vorticity number Wk for this mineral.
           */
          inline
          void set_sauter_mean(const unsigned int cpo_data_position,
                      const ArrayView<double> &data,
                      const unsigned int mineral_i,
                      const double sauter_mean) const
          {
            data[cpo_data_position + 12 + mineral_i * (n_grains * 30 + 13)] = sauter_mean;
          }

          /**
           * @brief Returns the kinematic vorticity number Wk = ||W||_F / ||D||_F for this mineral.
           */
          inline
          double get_mean_dislocation_density(const unsigned int cpo_data_position,
                        const ArrayView<const double> &data,
                        const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 12 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the kinematic vorticity number Wk for this mineral.
           */
          inline
          void set_mean_dislocation_density(const unsigned int cpo_data_position,
                      const ArrayView<double> &data,
                      const unsigned int mineral_i,
                      const double mean_dislocation_density) const
          {
            data[cpo_data_position + 12 + mineral_i * (n_grains * 30 + 13)] = mean_dislocation_density;
          }

          // ── Grain-level ────────────────────────────────────────────────────
          // absolute offset = cpo_data_position + 13 + relative_offset + grain_i * 30
          //                   + mineral_i * (n_grains * 30 + 13)

          /**
           * @brief Returns the grain size d for a grain (stored in volume_fraction slot by convention).
           */
          inline
          double get_volume_fractions_grains(const unsigned int cpo_data_position,
                                             const ArrayView<const double> &data,
                                             const unsigned int mineral_i,
                                             const unsigned int grain_i) const
          {
            return data[cpo_data_position + 13 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the grain size d for a grain.
           */
          inline
          void set_volume_fractions_grains(const unsigned int cpo_data_position,
                                           const ArrayView<double> &data,
                                           const unsigned int mineral_i,
                                           const unsigned int grain_i,
                                           const double volume_fractions_grains) const
          {
            data[cpo_data_position + 13 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = volume_fractions_grains;
          }

          /**
           * @brief Gets the rotation matrix for a grain (9 consecutive doubles, Tensor<2,3>).
           */
          inline
          Tensor<2,3> get_rotation_matrix_grains(const unsigned int cpo_data_position,
                                                  const ArrayView<const double> &data,
                                                  const unsigned int mineral_i,
                                                  const unsigned int grain_i) const
          {
            Tensor<2,3> rotation_matrix;
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                rotation_matrix[index] = data[cpo_data_position + 14 + grain_i * 30 + mineral_i * (n_grains * 30 + 13) + i];
              }
            return rotation_matrix;
          }

          /**
           * @brief Sets the rotation matrix for a grain.
           */
          inline
          void set_rotation_matrix_grains(const unsigned int cpo_data_position,
                                          const ArrayView<double> &data,
                                          const unsigned int mineral_i,
                                          const unsigned int grain_i,
                                          const Tensor<2,3> &rotation_matrix) const
          {
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                data[cpo_data_position + 14 + grain_i * 30 + mineral_i * (n_grains * 30 + 13) + i] = rotation_matrix[index];
              }
          }

          /**
           * @brief Returns the grain status.
           * -2 = empty buffer slot
           * -1 = dead, available for reuse
           *  0 = normal active grain
           *  1-4 = freshly nucleated this step via rx path 1-4
           */
          inline
          int get_grain_status(const unsigned int cpo_data_position,
                               const ArrayView<const double> &data,
                               const unsigned int mineral_i,
                               const unsigned int grain_i) const
          {
            return data[cpo_data_position + 23 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the grain status.
           */
          inline
          void set_grain_status(const unsigned int cpo_data_position,
                                const ArrayView<double> &data,
                                const unsigned int mineral_i,
                                const unsigned int grain_i,
                                const int grain_status) const
          {
            data[cpo_data_position + 23 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = grain_status;
          }

          /**
           * @brief Returns the strain rate ratio (diffusion fraction) for a grain.
           */
          inline
          double get_strain_rate_ratio(const unsigned int cpo_data_position,
                                       const ArrayView<const double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i) const
          {
            return data[cpo_data_position + 24 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the strain rate ratio for a grain.
           */
          inline
          void set_strain_rate_ratio(const unsigned int cpo_data_position,
                                     const ArrayView<double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i,
                                     const double strain_rate_ratio) const
          {
            data[cpo_data_position + 24 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = strain_rate_ratio;
          }

          /**
           * @brief Returns the Avrami recrystallized fraction for a grain.
           */
          inline
          double get_rx_fractions(const unsigned int cpo_data_position,
                                  const ArrayView<const double> &data,
                                  const unsigned int mineral_i,
                                  const unsigned int grain_i) const
          {
            return data[cpo_data_position + 25 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the Avrami recrystallized fraction for a grain.
           */
          inline
          void set_rx_fractions(const unsigned int cpo_data_position,
                                const ArrayView<double> &data,
                                const unsigned int mineral_i,
                                const unsigned int grain_i,
                                const double rx_fractions) const
          {
            data[cpo_data_position + 25 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = rx_fractions;
          }

          /**
           * @brief Returns the index of the most active slip system for a grain (0-3).
           */
          inline
          int get_active_slip_system(const unsigned int cpo_data_position,
                                     const ArrayView<const double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i) const
          {
            return data[cpo_data_position + 26 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the index of the most active slip system for a grain.
           */
          inline
          void set_active_slip_system(const unsigned int cpo_data_position,
                                      const ArrayView<double> &data,
                                      const unsigned int mineral_i,
                                      const unsigned int grain_i,
                                      const int active_slip_system) const
          {
            data[cpo_data_position + 26 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = active_slip_system;
          }

          /**
           * @brief Returns the dislocation creep slip strain rate for a grain.
           */
          inline
          double get_strain_rate(const unsigned int cpo_data_position,
                                 const ArrayView<const double> &data,
                                 const unsigned int mineral_i,
                                 const unsigned int grain_i) const
          {
            return data[cpo_data_position + 27 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the dislocation creep slip strain rate for a grain.
           */
          inline
          void set_strain_rate(const unsigned int cpo_data_position,
                               const ArrayView<double> &data,
                               const unsigned int mineral_i,
                               const unsigned int grain_i,
                               const double strain_rate) const
          {
            data[cpo_data_position + 27 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = strain_rate;
          }

          /**
           * @brief Returns the strain energy driving force Fstrain = mu*b^2*(rho_bar - rho_i) for a grain.
           */
          inline
          double get_strain_energy(const unsigned int cpo_data_position,
                                   const ArrayView<const double> &data,
                                   const unsigned int mineral_i,
                                   const unsigned int grain_i) const
          {
            return data[cpo_data_position + 28 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the strain energy driving force for a grain.
           */
          inline
          void set_strain_energy(const unsigned int cpo_data_position,
                                 const ArrayView<double> &data,
                                 const unsigned int mineral_i,
                                 const unsigned int grain_i,
                                 const double strain_energy) const
          {
            data[cpo_data_position + 28 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = strain_energy;
          }

          /**
           * @brief Returns the surface energy driving force Fsurface = 2*gamma*(1/R_bar - 1/R_i) for a grain.
           */
          inline
          double get_surface_energy(const unsigned int cpo_data_position,
                                    const ArrayView<const double> &data,
                                    const unsigned int mineral_i,
                                    const unsigned int grain_i) const
          {
            return data[cpo_data_position + 29 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the surface energy driving force for a grain.
           */
          inline
          void set_surface_energy(const unsigned int cpo_data_position,
                                  const ArrayView<double> &data,
                                  const unsigned int mineral_i,
                                  const unsigned int grain_i,
                                  const double surface_energy) const
          {
            data[cpo_data_position + 29 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = surface_energy;
          }

          /**
           * @brief Returns the differential stress sigma = 1.5*mu*b*sqrt(rho_i) for a grain.
           */
          inline
          double get_differential_stress(const unsigned int cpo_data_position,
                                         const ArrayView<const double> &data,
                                         const unsigned int mineral_i,
                                         const unsigned int grain_i) const
          {
            return data[cpo_data_position + 30 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the differential stress for a grain.
           */
          inline
          void set_differential_stress(const unsigned int cpo_data_position,
                                       const ArrayView<double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i,
                                       const double differential_stress) const
          {
            data[cpo_data_position + 30 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = differential_stress;
          }

          /**
           * @brief Returns the grain size change d_new - d_n this step for a grain.
           */
          inline
          double get_grain_size_change(const unsigned int cpo_data_position,
                                       const ArrayView<const double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i) const
          {
            return data[cpo_data_position + 31 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the grain size change for a grain.
           */
          inline
          void set_grain_size_change(const unsigned int cpo_data_position,
                                     const ArrayView<double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i,
                                     const double grain_size_change) const
          {
            data[cpo_data_position + 31 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = grain_size_change;
          }

          /**
           * @brief Returns the analytical dislocation density rho_i for a grain.
           */
          inline
          double get_dislocation_density(const unsigned int cpo_data_position,
                                         const ArrayView<const double> &data,
                                         const unsigned int mineral_i,
                                         const unsigned int grain_i) const
          {
            return data[cpo_data_position + 32 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the dislocation density for a grain.
           */
          inline
          void set_dislocation_density(const unsigned int cpo_data_position,
                                       const ArrayView<double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i,
                                       const double dislocation_density) const
          {
            data[cpo_data_position + 32 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = dislocation_density;
          }

          /**
           * @brief Returns the accumulated strain since last rx event for a grain.
           */
          inline
          double get_strain_accumulated(const unsigned int cpo_data_position,
                                        const ArrayView<const double> &data,
                                        const unsigned int mineral_i,
                                        const unsigned int grain_i) const
          {
            return data[cpo_data_position + 33 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the accumulated strain for a grain.
           */
          inline
          void set_strain_accumulated(const unsigned int cpo_data_position,
                                      const ArrayView<double> &data,
                                      const unsigned int mineral_i,
                                      const unsigned int grain_i,
                                      const double strain_accumulated) const
          {
            data[cpo_data_position + 33 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = strain_accumulated;
          }

          // ── Diagnostic grain slots [+21..+29] ──────────────────────────────

          /**
           * @brief Returns the stiffness ratio J = kappa*dt/d_n^2.
           * J < 1: ETD1 branch used. J >= 1: BE quadratic branch used.
           */
          inline
          double get_J(const unsigned int cpo_data_position,
                       const ArrayView<const double> &data,
                       const unsigned int mineral_i,
                       const unsigned int grain_i) const
          {
            return data[cpo_data_position + 34 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the stiffness ratio J for a grain.
           */
          inline
          void set_J(const unsigned int cpo_data_position,
                     const ArrayView<double> &data,
                     const unsigned int mineral_i,
                     const unsigned int grain_i,
                     const double J) const
          {
            data[cpo_data_position + 34 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = J;
          }

          /**
           * @brief Returns the local grain relaxation timescale tau = d_n / |R_dn|.
           */
          inline
          double get_tau(const unsigned int cpo_data_position,
                         const ArrayView<const double> &data,
                         const unsigned int mineral_i,
                         const unsigned int grain_i) const
          {
            return data[cpo_data_position + 35 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the local grain relaxation timescale for a grain.
           */
          inline
          void set_tau(const unsigned int cpo_data_position,
                       const ArrayView<double> &data,
                       const unsigned int mineral_i,
                       const unsigned int grain_i,
                       const double tau) const
          {
            data[cpo_data_position + 35 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = tau;
          }

          /**
           * @brief Returns the total frozen drive A = R_dn + kappa/d_n for a grain.
           */
          inline
          double get_A_drive(const unsigned int cpo_data_position,
                             const ArrayView<const double> &data,
                             const unsigned int mineral_i,
                             const unsigned int grain_i) const
          {
            return data[cpo_data_position + 36 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the total frozen drive A for a grain.
           */
          inline
          void set_A_drive(const unsigned int cpo_data_position,
                           const ArrayView<double> &data,
                           const unsigned int mineral_i,
                           const unsigned int grain_i,
                           const double A_drive) const
          {
            data[cpo_data_position + 36 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = A_drive;
          }

          /**
           * @brief Returns the misorientation angle theta between R(t_n) and R(t_{n+1}) for a grain.
           * theta = arccos((tr(delta_R) - 1) / 2) where delta_R = R_new * R_old^T.
           */
          inline
          double get_theta(const unsigned int cpo_data_position,
                           const ArrayView<const double> &data,
                           const unsigned int mineral_i,
                           const unsigned int grain_i) const
          {
            return data[cpo_data_position + 37 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the misorientation angle theta for a grain.
           */
          inline
          void set_theta(const unsigned int cpo_data_position,
                         const ArrayView<double> &data,
                         const unsigned int mineral_i,
                         const unsigned int grain_i,
                         const double theta) const
          {
            data[cpo_data_position + 37 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = theta;
          }

          /**
           * @brief Returns the rotation axis unit vector n_hat for a grain (Tensor<1,3>, 3 consecutive doubles).
           * Axis around which the grain rotated by theta this step.
           */
          inline
          Tensor<1,3> get_rotation_axis(const unsigned int cpo_data_position,
                                         const ArrayView<const double> &data,
                                         const unsigned int mineral_i,
                                         const unsigned int grain_i) const
          {
            Tensor<1,3> axis;
            for (unsigned int i = 0; i < 3; ++i)
              axis[i] = data[cpo_data_position + 38 + grain_i * 30 + mineral_i * (n_grains * 30 + 13) + i];
            return axis;
          }

          /**
           * @brief Sets the rotation axis unit vector for a grain.
           */
          inline
          void set_rotation_axis(const unsigned int cpo_data_position,
                                  const ArrayView<double> &data,
                                  const unsigned int mineral_i,
                                  const unsigned int grain_i,
                                  const Tensor<1,3> &rotation_axis) const
          {
            for (unsigned int i = 0; i < 3; ++i)
              data[cpo_data_position + 38 + grain_i * 30 + mineral_i * (n_grains * 30 + 13) + i] = rotation_axis[i];
          }

          /**
           * @brief Returns the instantaneous spin rate theta_dot = ||Omega||_F / sqrt(2) for a grain.
           */
          inline
          double get_theta_dot(const unsigned int cpo_data_position,
                               const ArrayView<const double> &data,
                               const unsigned int mineral_i,
                               const unsigned int grain_i) const
          {
            return data[cpo_data_position + 41 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the instantaneous spin rate for a grain.
           */
          inline
          void set_theta_dot(const unsigned int cpo_data_position,
                             const ArrayView<double> &data,
                             const unsigned int mineral_i,
                             const unsigned int grain_i,
                             const double theta_dot) const
          {
            data[cpo_data_position + 41 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = theta_dot;
          }

          /**
           * @brief Returns the fabric angle psi = arccos(|a_[100] . e_shear|) for a grain.
           * e_shear is the eigenvector of D = (L+L^T)/2 for the largest positive eigenvalue,
           * computed at runtime from the velocity gradient — not stored.
           */
          inline
          double get_psi(const unsigned int cpo_data_position,
                         const ArrayView<const double> &data,
                         const unsigned int mineral_i,
                         const unsigned int grain_i) const
          {
            return data[cpo_data_position + 42 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the fabric angle psi for a grain.
           */
          inline
          void set_psi(const unsigned int cpo_data_position,
                       const ArrayView<double> &data,
                       const unsigned int mineral_i,
                       const unsigned int grain_i,
                       const double psi) const
          {
            data[cpo_data_position + 42 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = psi;
          }
          
          /**
           * @brief Returns the fabric angle psi = arccos(|a_[100] . e_shear|) for a grain.
           * e_shear is the eigenvector of D = (L+L^T)/2 for the largest positive eigenvalue,
           * computed at runtime from the velocity gradient — not stored.
           */
          inline
          double get_post_rx_grainsize(const unsigned int cpo_data_position,
                                       const ArrayView<const double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i) const
          {
            return data[cpo_data_position + 42 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)];
          }

          /**
           * @brief Sets the fabric angle psi for a grain.
           */
          inline
          void set_post_rx_grainsize(const unsigned int cpo_data_position,
                                     const ArrayView<double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i,
                                     const double post_rx_grainsize) const
          {
            data[cpo_data_position + 42 + grain_i * 30 + mineral_i * (n_grains * 30 + 13)] = post_rx_grainsize;
          }

        private:
          /**
           * Computes a random rotation matrix.
           */
          void
          compute_random_rotation_matrix(Tensor<2,3> &rotation_matrix) const;

          /**
           * @brief Updates the volume fractions and rotation matrices with a Forward Euler scheme.
           */
          double
          advect_forward_euler(const unsigned int cpo_data_position,
                               const ArrayView<double> &data,
                               const unsigned int mineral_i,
                               const double dt,
                               const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const;

          /**
           * @brief Updates the volume fractions and rotation matrices with a Backward Euler scheme.
           */
          double
          advect_backward_euler(const unsigned int cpo_data_position,
                                const ArrayView<double> &data,
                                const unsigned int mineral_i,
                                const double dt,
                                const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const;

          /**
           * @brief Updates the volume fractions and rotation matrices with the ETD1/BE switching scheme.
           * Passes e_shear (principal stretching direction from L) for computing the fabric angle psi.
           */
          double
          advect_exponential_update(const unsigned int cpo_data_position,
                                    const ArrayView<double> &data,
                                    const unsigned int mineral_i,
                                    const double dt,
                                    const Tensor<1,3> &e_shear,
                                    const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const;

          /**
           * Computes and returns the volume fraction and grain orientation derivatives such that
           * the grains stay the same size and the orientations rotating passively with the particle.
           */
          std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
          compute_derivatives_spin_tensor(const Tensor<2,3> &velocity_gradient_tensor) const;

          /**
           * Random number generator used for initialization of particles
           */
          mutable boost::mt19937 random_number_generator;
          unsigned int random_number_seed;

          unsigned int n_grains;

          unsigned int n_minerals;

          /**
           * The index of the water composition.
           */
          unsigned int water_index;

          /**
           * A vector containing the deformation type selectors provided by the user.
           */
          std::vector<DeformationTypeSelector> deformation_type_selector;

          /**
           * Store the volume fraction for each mineral.
           */
          std::vector<double> volume_fractions_minerals;

          /**
           * Advection method for particle properties
           */
          AdvectionMethod advection_method;

          /**
           * What algorithm to use to compute the derivatives
           */
          CPODerivativeAlgorithm cpo_derivative_algorithm;

          /**
           * This value determines the tolerance used for the Backward Euler and
           * Crank-Nicolson iterations.
           */
          double property_advection_tolerance;

          /**
           * This value determines the the maximum number of iterations used for the
           * Backward Euler and Crank-Nicolson iterations.
           */
          unsigned int property_advection_max_iterations;

          /**
           * @name D-Rex variables
           */
          /** @{ */
          double stress_exponent;
          double nucleation_efficiency;
          double exponent_p;
          double threshold_GBS;
          double mobility;
          CPOInitialGrainsModel initial_grains_model;
          std::vector<double> CPX_RRSS;
          std::vector<double> OlivineD_RRSS;
          /** @} */

          double n_grains_init;
          double n_grains_buffer;
          double initial_grain_size;
          double initial_dislocation_density;

          std::string input_orientation_file;

          double normal_distribution_mean;
          double normal_distribution_standard_deviation;

          double lognormal_distribution_mean;
          double lognormal_distribution_standard_deviation;

          double uniform_distribution_max;
          double uniform_distribution_min;

          std::vector<double> drexpp_exponent_p;
          std::vector<double> drexpp_stress_exponent;
          std::vector<double> drexpp_mobility;
          double interfacial_energy;
          double avrami_slope_input;
          double max_dispersion;

          std::shared_ptr<MaterialModel::Rheology::DiffusionCreep<dim>> rheology_diff;
          std::shared_ptr<MaterialModel::Rheology::DislocationCreep<dim>> rheology_disl;
          std::shared_ptr<MaterialModel::Rheology::ViscoPlastic<dim>> rheology_vipl;
          double min_strain_rate;
          std::vector<double> thermal_diffusivities;
          bool define_conductivities;
          std::vector<double> thermal_conductivities;
          std::shared_ptr<MaterialModel::MaterialUtilities::PhaseFunction<dim>> phase_function;

      };
    }
  }
}

#endif