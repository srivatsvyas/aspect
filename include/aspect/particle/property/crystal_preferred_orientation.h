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
        forward_euler, backward_euler
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
       * as D-Rex (Kaminski and Ribe, 2001; ́Kaminski et al., 2004).
       *
       * This plugin stores M minerals and for each mineral it stores N grains. The total amount of memory stored is 5
       * doubles per mineral plus 24 doubles per grain, resulting in a total memory of M * (5 + N * 24) . The layout of
       * the data stored is the following (note that for this plugin the following dims are always 3):
       * 1. M minerals times
       *    1.1  Mineral deformation type   -> 1 double, at location
       *                                      => 0 + mineral_i * (n_grains * 24 + 5)
       *    2.1. Mineral volume fraction    -> 1 double, at location
       *                                      => 1 + mineral_i * (n_grains * 24 + 5)
       *    2.2. Mineral bulk recrystallization grain size    -> 1 double, at location
       *                                      => 2 + mineral_i * (n_grains * 24 + 5)
       *    2.3. Mineral mean grain size    -> 1 double, at location
       *                                      => 3 + mineral_i * (n_grains * 24 + 5)
       *    2.4. Mineral mean mechanism ratio    -> 1 double, at location
       *                                      => 4 + mineral_i * (n_grains * 24 + 5)
       *    2.5. N grains times:
       *         2.1. volume fraction grain       -> 1 double, at location:
       *                                          => 5 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.2. rotation_matrix grain       -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
       *                                          => 6 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.3. grain status                -> 1 integer, at location:
       *                                          => 15 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.4. strain rate ratio           -> 1 double, at location:
       *                                          => 16 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.5. viscosity ratio             -> 1 double, at location:
       *                                          => 17 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.6. grain boundary velocity     -> 1 double, at location:
       *                                          => 18 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.7. recrystallization fraction  -> 1 double, at location:
       *                                          => 19 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.8. active slip system          -> 1 integer, at location:
       *                                          => 20 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.9. strain rate                 -> 1 double, at location:
       *                                          => 21 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.10. strain energy              -> 1 double, at location:
       *                                          => 22 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.11. surface energy             -> 1 double, at location:
       *                                          => 23 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.12. differential stress        -> 1 double, at location:
       *                                          => 24 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.13. grain size change          -> 1 double, at location:
       *                                          => 25 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.14. dislocation density        -> 1 double, at location:
       *                                          => 26 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.15. strain accumulated          -> 1 double, at location:
       *                                          => 27 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       *         2.16. energy ratio                -> 1 double, at location:
       *                                          => 28 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)
       * Last used data entry is n_minerals * (n_grains * 24 + 5).
       *
       * We store the same number of grains for all minerals (e.g. olivine and enstatite
       * grains), although their volume fractions may not be the same. This is because we need a minimum number
       * of grains per particle to perform reliable statistics on it. This minimum should be the same for all
       * minerals.
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
           *
           * @param cpo_index The location where the CPO data starts in the data array.
           * @param data The data array containing the CPO data.
           * @param mineral_i The mineral for which to compute the derivatives for.
           * @param strain_rate_3d The 3D strain rate at the location where the derivative is requested.
           * @param velocity_gradient_tensor The velocity gradient tensor at the location where the derivative is requested.
           * @param position The location for which the derivative is requested.
           * @param cell The cell that contains the point @p position.
           * @param temperature The temperature at the location where the derivative is requested.
           * @param pressure The pressure at the location where the derivative is requested.
           * @param velocity The veloicty at the location where the derivative is requested.
           * @param compositions The compositios at the location where the derivative is requested.
           * @param strain_rate The strain-rate at the location where the derivative is requested.
           * @param deviatoric_strain_rate The deviatoric strain-rate at the location where the derivative is requested.
           * @param water_content The water content at the location where the derivative is requested.
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
           *
           * @param cpo_index The location where the CPO data starts in the data array.
           * @param data The data array containing the CPO data.
           * @param mineral_i The mineral for which to compute the derivatives for.
           * @param strain_rate_3d The 3D strain rate
           * @param velocity_gradient_tensor The velocity gradient tensor
           * @param ref_resolved_shear_stress Represent one value per slip plane.
           * The planes are ordered from weakest to strongest with relative values,
           * where the inactive plane is infinity strong. So it is a measure of strength
           * on each slip plane.
           * @param deformation_type Represent one of the deformation types
           * @param prevent_nondimensionalization Prevent nondimensializing values internally.
           * Only for unit testing purposes.
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
          * @brief The dynamic recrystallization module called in drexpp. Takes care of calculating
          * the number of recrystallization and handles the assignment of the newly nucleated grains to 
          * slots in the memory array
          *
          * @param cpo_index The location where the CPO data starts in the data array.
          * @param data The particle data vector.
          * @param mineral_i The mineral for which to apply recrystallization.
          * @param recrystalized_fraction Per-grain fraction of volume consumed by
          *        recrystallization this time step.
          * @param sgr_rotation_axis Per-grain subgrain rotation axes used to set the
          *        orientation of newly nucleated recrystallized grains.
          * @param piezometer Per-mineral piezometrically predicted grain size used to
          *        set the size of newly recrystallized grains.
          * @param rx_now Per-grain flag indicating whether a grain was newly nucleated in this time step.
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
           *
           * @param cpo_index The location where the CPO data starts in the data array.
           * @param data The data array containing the CPO data.
           * @param mineral_i The mineral for which to compute the derivatives.
           * @param strain_rate_3d The 3D strain rate tensor.
           * @param velocity_gradient_tensor The velocity gradient tensor.
           * @param ref_resolved_shear_stress Represent one value per slip plane.
           * The planes are ordered from weakest to strongest with relative values,
           * where the inactive plane is infinity strong. So it is a measure of strength
           * on each slip plane.
           * @param diffusion_pre_strainrate Pre-exponential factor for the diffusion creep
           *        strain rate.
           * @param deviatoric_stress The deviatoric stress tensor used to compute the
           *        grain boundary driving force for grain boundary migration.
           * @param diffusion_grain_size_exponent Grain size exponent in the diffusion
           *        creep flow law.
           * @param dislocation_factor Pre-exponential scaling factor for the dislocation
           *        creep strain rate.
           * @param dislocation_stress_exponent Stress exponent in the dislocation creep
           *        flow law.
           * @param deviatoric_strain_rate The deviatoric strain rate tensor.
           * @param growth_rate Normal grain growth rate constant.
           * @param temperature Temperature at the particle location,
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
                                     const double growth_rate,
                                     const double temperature) const;

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
           * If the provided @p deformation_type_selector is a specific deformation type,
           * the function will simply return the corresponding deformation type. However,
           * if the @p deformation_type_selector is an algorithm to determine the current
           * deformation type (e.g. based on measured lab data or analytical models), then
           * the function computes the appropriate deformation type at the given conditions
           * and returns the compute deformation type.
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
           *
           * The inactive plane should theoretically be infinitely strong, but this is nummerically not desirable,
           * so an optional max_value can be set to indicate an inactive plane.
           *
           * It is currently designed to return the relative strength of the slip planes for olivine, which are are 4,
           * but this could be generalized.
           */
          std::array<double,4>
          reference_resolved_shear_stress_from_deformation_type(DeformationType deformation_type,
                                                                double max_value = 1e60) const;

          /**
           * @brief Returns the value in the data array representing the deformation type.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the deformation type for.
           */
          inline
          DeformationType get_deformation_type(const unsigned int cpo_data_position,
                                               const ArrayView<double> &data,
                                               const unsigned int mineral_i) const
          {
            return static_cast<DeformationType>(data[cpo_data_position + 0 + mineral_i * (n_grains * 24 + 5)]);
          }

          /**
           * @brief Sets the value in the data array representing the deformation type.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value deformation type for.
           * @param deformation_type The value of the deformation type to set.
           */
          inline
          void set_deformation_type(const unsigned int cpo_data_position,
                                    const ArrayView<double> &data,
                                    const unsigned int mineral_i,
                                    const DeformationType deformation_type) const
          {
            data[cpo_data_position + 0 + mineral_i * (n_grains * 24 + 5)] = static_cast<double>(deformation_type);
          }

          /**
           * @brief Returns the value in the data array representing the volume fraction of a mineral.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the volume fraction of a mineral for.
           */
          inline
          double get_volume_fraction_mineral(const unsigned int cpo_data_position,
                                             const ArrayView<double> &data,
                                             const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 1 + mineral_i *(n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the volume fraction of a mineral.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the volume fraction of a mineral for.
           * @param volume_fraction_mineral The value of the volume fraction of a mineral to set.
           */
          inline
          void set_volume_fraction_mineral(const unsigned int cpo_data_position,
                                           const ArrayView<double> &data,
                                           const unsigned int mineral_i,
                                           const double volume_fraction_mineral) const
          {
            data[cpo_data_position + 1 + mineral_i *(n_grains * 24 + 5)] = volume_fraction_mineral;
          }

          /**
           * @brief Returns the value in the data array representing the recrystalization grain size predicted for the mineral phase.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the recrystallization grain size of a mineral for.
           */
          inline
          double get_bulk_recrystalization_grain_size_mineral(const unsigned int cpo_data_position,
                                                              const ArrayView<double> &data,
                                                              const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 2 + mineral_i *(n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing recrystalization grain size predicted for the mineral phase.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the recrystalization grain size of a mineral for.
           * @param recrystalization_size The value of the recrystalization grain size predicted for the mineral phase.
           */
          inline
          void set_bulk_recrystalization_grain_size_mineral(const unsigned int cpo_data_position,
                                                            const ArrayView<double> &data,
                                                            const unsigned int mineral_i,
                                                            const double recrystalization_size) const
          {
            data[cpo_data_position + 2 + mineral_i *(n_grains * 24 + 5)] = recrystalization_size;
          }

          /**
           * @brief Returns the mean grain size for the mineral phase.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the mean grain size of a mineral for.
           */
          inline
          double get_mean_grain_size_mineral(const unsigned int cpo_data_position,
                                             const ArrayView<double> &data,
                                             const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 3 + mineral_i *(n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing mean grain size predicted for the mineral phase.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the mean grain size of a mineral for.
           * @param mean_grain_size The value of the mean grain size predicted for the mineral phase.
           */
          inline
          void set_mean_grain_size_mineral(const unsigned int cpo_data_position,
                                           const ArrayView<double> &data,
                                           const unsigned int mineral_i,
                                           const double mean_grain_size) const
          {
            data[cpo_data_position + 3 + mineral_i *(n_grains * 24 + 5)] = mean_grain_size;
          }

          /**
           * @brief Returns the mean mechanism ratio for the mineral phase.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the mean mechanism ratio of a mineral for.
           */
          inline
          double get_mean_mechanism_ratio_mineral(const unsigned int cpo_data_position,
                                                  const ArrayView<double> &data,
                                                  const unsigned int mineral_i) const
          {
            return data[cpo_data_position + 4 + mineral_i *(n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing mean mechanism ratio predicted for the mineral phase.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the mean mechanism ratio of a mineral for.
           * @param mean_mechanism_ratio_mineral The value of the mean mechanism ratio predicted for the mineral phase.
           */
          inline
          void set_mean_mechanism_ratio_mineral(const unsigned int cpo_data_position,
                                                const ArrayView<double> &data,
                                                const unsigned int mineral_i,
                                                const double mean_mechanism_ratio_mineral) const
          {
            data[cpo_data_position + 4 + mineral_i *(n_grains * 24 + 5)] = mean_mechanism_ratio_mineral;
          }

          /**
           * @brief Returns the value in the data array representing the volume fraction of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the volume fraction of a grain for.
           * @param grain_i The grain to get the value of the volume fraction of.
           */
          inline
          double get_volume_fractions_grains(const unsigned int cpo_data_position,
                                             const ArrayView<const double> &data,
                                             const unsigned int mineral_i,
                                             const unsigned int grain_i) const
          {
            return data[cpo_data_position + 5 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the volume fraction of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the volume fraction of a grain for.
           * @param grain_i The grain to set the value of the volume fraction of.
           * @param volume_fractions_grains The value of the volume fraction of a grain to set.
           */
          inline
          void set_volume_fractions_grains(const unsigned int cpo_data_position,
                                           const ArrayView<double> &data,
                                           const unsigned int mineral_i,
                                           const unsigned int grain_i,
                                           const double volume_fractions_grains) const
          {
            data[cpo_data_position + 5 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = volume_fractions_grains;
          }

          /**
           * @brief Gets the rotation matrix for a grain in a mineral.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the rotation matrix of a grain for.
           * @param grain_i The grain to get the value of the rotation matrix of.
           * @return Tensor<2,3> The rotation matrix of a grain in a mineral
           */
          inline
          Tensor<2,3> get_rotation_matrix_grains(const unsigned int cpo_data_position,
                                                 const ArrayView<const double> &data,
                                                 const unsigned int mineral_i,
                                                 const unsigned int grain_i) const
          {
            Tensor<2,3> rotation_matrix;
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                rotation_matrix[index] = data[cpo_data_position + 6 + grain_i * 24 + mineral_i * (n_grains * 24 + 5) + i];
              }
            return rotation_matrix;
          }

          /**
           * @brief Sets the rotation matrix for a grain in a mineral.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the rotation matrix of a grain for.
           * @param grain_i The grain to set the value of the rotation matrix of.
           * @param rotation_matrix The rotation matrix to set for the grain in the mineral.
           */
          inline
          void set_rotation_matrix_grains(const unsigned int cpo_data_position,
                                          const ArrayView<double> &data,
                                          const unsigned int mineral_i,
                                          const unsigned int grain_i,
                                          const Tensor<2,3> &rotation_matrix) const
          {
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                data[cpo_data_position + 6 + grain_i * 24 + mineral_i * (n_grains * 24 + 5) + i] = rotation_matrix[index];
              }
          }

          /**
           * @brief Returns the value in the data array representing the status of a grain (ie. parent, recrystallized, buffer, uninitialized).
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the grain status of a grain for.
           * @param grain_i The grain to get the value of the grain status of.
           */
          inline
          int get_grain_status(const unsigned int cpo_data_position,
                               const ArrayView<const double> &data,
                               const unsigned int mineral_i,
                               const unsigned int grain_i) const
          {
            return data[cpo_data_position + 15 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the grain status of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the grain statusof a grain for.
           * @param grain_i The grain to set the value of the grain status of.
           * @param grain_status The value of the grain status of a grain to set.
           */
          inline
          void set_grain_status(const unsigned int cpo_data_position,
                                const ArrayView<double> &data,
                                const unsigned int mineral_i,
                                const unsigned int grain_i,
                                const int grain_status) const
          {
            data[cpo_data_position + 15 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = grain_status;
          }

          /**
           * @brief Returns the value in the data array representing the ratio of the strain rates.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the strain rates of a grain for.
           * @param grain_i The grain to get the value of the strain rates of.
           */
          inline
          double get_strain_rate_ratio(const unsigned int cpo_data_position,
                                       const ArrayView<const double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i) const
          {
            return data[cpo_data_position + 16 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the strain rates of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the strain rates of a grain for.
           * @param grain_i The grain to set the value of the strain rates of.
           * @param strain_rate_ratio The value of the strain rates of a grain to set.
           */
          inline
          void set_strain_rate_ratio(const unsigned int cpo_data_position,
                                     const ArrayView<double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i,
                                     const double strain_rate_ratio) const
          {
            data[cpo_data_position + 16 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = strain_rate_ratio;
          }


          /**
           * @brief Returns the value in the data array representing the viscosity ratio.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the viscosity ratio of a grain for.
           * @param grain_i The grain to get the value of the viscosity ratio of.
           */
          inline
          double get_viscosity_ratio(const unsigned int cpo_data_position,
                                     const ArrayView<const double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i) const
          {
            return data[cpo_data_position + 17 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the viscosity ratio of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the viscosity ratio of a grain for.
           * @param grain_i The grain to set the value of the viscosity ratio of.
           * @param viscosity_ratio The value of the viscosity ratio of a grain to set.
           */
          inline
          void set_viscosity_ratio(const unsigned int cpo_data_position,
                                   const ArrayView<double> &data,
                                   const unsigned int mineral_i,
                                   const unsigned int grain_i,
                                   const double viscosity_ratio) const
          {
            data[cpo_data_position + 17 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = viscosity_ratio;
          }


          /**
           * @brief Returns the value in the data array representing the grain boundary velocity.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the grain boundary velocity of a grain for.
           * @param grain_i The grain to get the value of the grain boundary velocity of.
           */
          inline
          double get_grain_boundary_velocity(const unsigned int cpo_data_position,
                                             const ArrayView<const double> &data,
                                             const unsigned int mineral_i,
                                             const unsigned int grain_i) const
          {
            return data[cpo_data_position + 18 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the grain boundary velocity of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the grain boundary velocity of a grain for.
           * @param grain_i The grain to set the value of the grain boundary velocity of.
           * @param grain_boundary_velocity The value of the grain boundary velocity of a grain to set.
           */
          inline
          void set_grain_boundary_velocity(const unsigned int cpo_data_position,
                                           const ArrayView<double> &data,
                                           const unsigned int mineral_i,
                                           const unsigned int grain_i,
                                           const double grain_boundary_velocity) const
          {
            data[cpo_data_position + 18 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = grain_boundary_velocity;
          }

          /**
           * @brief Returns the value in the data array representing the recrystallized fraction of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the recrystallized fraction of a grain for.
           * @param grain_i The grain to get the value of the recrystallized fraction of.
           */
          inline
          double get_rx_fractions(const unsigned int cpo_data_position,
                                  const ArrayView<const double> &data,
                                  const unsigned int mineral_i,
                                  const unsigned int grain_i) const
          {
            return data[cpo_data_position + 19 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the recrystallized fraction of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the recrystallized fraction of a grain for.
           * @param grain_i The grain to set the value of the recrystallized fraction of.
           * @param rx_fractions The value of the recrystallized fraction of a grain to set.
           */
          inline
          void set_rx_fractions(const unsigned int cpo_data_position,
                                const ArrayView<double> &data,
                                const unsigned int mineral_i,
                                const unsigned int grain_i,
                                const double rx_fractions) const
          {
            data[cpo_data_position + 19 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = rx_fractions;
          }

          // After this, all properties stored to the cpo array is purely for the purpose of visualizing the evolution of different model parameters for verification and not needed for running the model
          /**
          * @brief Returns the value in the data array representing the weakest/most active slip system of the grain.
          *
          * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
          * @param data The particle data vector.
          * @param mineral_i The mineral to get the value of the most active slip system of a grain for.
          * @param grain_i The grain to get the value of the most active slip system of.
          */
          inline
          int get_active_slip_system(const unsigned int cpo_data_position,
                                     const ArrayView<const double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i) const
          {
            return data[cpo_data_position + 20 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the most active slip system of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the most active slip system of a grain for.
           * @param grain_i The grain to set the value of the most active slip system of.
           * @param active_slip_system The value of the most active slip system of a grain to set.
           */
          inline
          void set_active_slip_system(const unsigned int cpo_data_position,
                                      const ArrayView<double> &data,
                                      const unsigned int mineral_i,
                                      const unsigned int grain_i,
                                      const int active_slip_system) const
          {
            data[cpo_data_position + 20 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = active_slip_system;
          }

          /**
          * @brief Returns the value in the data array representing the strain rate arising from glide along slip plane of a grain.
          *
          * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
          * @param data The particle data vector.
          * @param mineral_i The mineral to get the value of the strain rate arising from glide along slip plane of a grain for.
          * @param grain_i The grain to get the value of the strain rate arising from glide along slip plane of.
          */
          inline
          double get_strain_rate(const unsigned int cpo_data_position,
                                 const ArrayView<const double> &data,
                                 const unsigned int mineral_i,
                                 const unsigned int grain_i) const
          {
            return data[cpo_data_position + 21 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the strain rate arising from glide along slip plane of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the strain rate arising from glide along slip plane of a grain for.
           * @param grain_i The grain to set the value of the strain rate arising from glide along slip plane of.
           * @param strain_rate The value of the strain rate arising from glide along slip plane to set.
           */
          inline
          void set_strain_rate(const unsigned int cpo_data_position,
                               const ArrayView<double> &data,
                               const unsigned int mineral_i,
                               const unsigned int grain_i,
                               const double strain_rate) const
          {
            data[cpo_data_position + 21 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = strain_rate;
          }


          /**
          * @brief Returns the value in the data array representing the strain energy due to stored dislocation density of a grain.
          *
          * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
          * @param data The particle data vector.
          * @param mineral_i The mineral to get the value of the strain energy of a grain for.
          * @param grain_i The grain to get the value of the strain energy of.
          */

          inline
          double get_strain_energy(const unsigned int cpo_data_position,
                                   const ArrayView<const double> &data,
                                   const unsigned int mineral_i,
                                   const unsigned int grain_i) const
          {
            return data[cpo_data_position + 22 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the strain energy due to stored dislocation density of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the strain energy of a grain for.
           * @param grain_i The grain to set the value of the strain energy of.
           * @param strain_energy The value of the strain energy to set.
           */
          inline
          void set_strain_energy(const unsigned int cpo_data_position,
                                 const ArrayView<double> &data,
                                 const unsigned int mineral_i,
                                 const unsigned int grain_i,
                                 const double strain_energy) const
          {
            data[cpo_data_position + 22 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = strain_energy;
          }

          /**
          * @brief Returns the value in the data array representing the surface energy of a grain.
          *
          * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
          * @param data The particle data vector.
          * @param mineral_i The mineral to get the value of the surface energy of a grain for.
          * @param grain_i The grain to get the value of the surface energy of.
          */
          inline
          double get_surface_energy(const unsigned int cpo_data_position,
                                    const ArrayView<const double> &data,
                                    const unsigned int mineral_i,
                                    const unsigned int grain_i) const
          {
            return data[cpo_data_position + 23 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the surface energy of a grain.
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the surface energy of a grain for.
           * @param grain_i The grain to set the value of the surface energy of.
           * @param surface_energy The value of the surface energy of a grain to set.
           */
          inline
          void set_surface_energy(const unsigned int cpo_data_position,
                                  const ArrayView<double> &data,
                                  const unsigned int mineral_i,
                                  const unsigned int grain_i,
                                  const double surface_energy) const
          {
            data[cpo_data_position + 23 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = surface_energy;
          }

          /**
          * @brief Returns the value in the data array representing the differential stress of a grain.
          *
          * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
          * @param data The particle data vector.
          * @param mineral_i The mineral to get the value of the differential stress of a grain for.
          * @param grain_i The grain to get the value of the differential stress of.
          */
          inline
          double get_differential_stress(const unsigned int cpo_data_position,
                                         const ArrayView<const double> &data,
                                         const unsigned int mineral_i,
                                         const unsigned int grain_i) const
          {
            return data[cpo_data_position + 24 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the differential stress of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the differential stress of a grain for.
           * @param grain_i The grain to set the value of the differential stress of.
           * @param differential_stress The value of the differential stress of a grain to set.
           */
          inline
          void set_differential_stress(const unsigned int cpo_data_position,
                                       const ArrayView<double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i,
                                       const double differential_stress) const
          {
            data[cpo_data_position + 24 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = differential_stress;
          }

          /**
           * @brief Returns the value in the data array representing the grain size change of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the grain size change of a grain for.
           * @param grain_i The grain to get the value of the grain size change of.
           */
          inline
          double get_grain_size_change(const unsigned int cpo_data_position,
                                       const ArrayView<const double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i) const
          {
            return data[cpo_data_position + 25 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the grain size change of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the grain size change of a grain for.
           * @param grain_i The grain to set the value of the grain size change of.
           * @param grain_size_change The value of the grain size change of a grain to set.
           */
          inline
          void set_grain_size_change(const unsigned int cpo_data_position,
                                     const ArrayView<double> &data,
                                     const unsigned int mineral_i,
                                     const unsigned int grain_i,
                                     const double grain_size_change) const
          {
            data[cpo_data_position + 25 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = grain_size_change;
          }

          /**
           * @brief Returns the value in the data array representing the dislocation density of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the dislocation density of a grain for.
           * @param grain_i The grain to get the value of the dislocation density of.
           */
          inline
          double get_dislocation_density(const unsigned int cpo_data_position,
                                         const ArrayView<const double> &data,
                                         const unsigned int mineral_i,
                                         const unsigned int grain_i) const
          {
            return data[cpo_data_position + 26 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the dislocation density of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the dislocation density of a grain for.
           * @param grain_i The grain to set the value of the dislocation density of.
           * @param dislocation_density The value of the dislocation density of a grain to set.
           */
          inline
          void set_dislocation_density(const unsigned int cpo_data_position,
                                       const ArrayView<double> &data,
                                       const unsigned int mineral_i,
                                       const unsigned int grain_i,
                                       const double dislocation_density) const
          {
            data[cpo_data_position + 26 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = dislocation_density;
          }

          /**
           * @brief Returns the value in the data array representing the strain accumulation of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the strain accumulation of a grain for.
           * @param grain_i The grain to get the value of the strain accumulation of.
           */
          inline
          double get_strain_accumulated(const unsigned int cpo_data_position,
                                        const ArrayView<const double> &data,
                                        const unsigned int mineral_i,
                                        const unsigned int grain_i) const
          {
            return data[cpo_data_position + 27 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the strain accumulation of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the strain accumulation of a grain for.
           * @param grain_i The grain to set the value of the strain accumulation of.
           * @param strain_accumulation The value of the strain accumulation of a grain to set.
           */
          inline
          void set_strain_accumulated(const unsigned int cpo_data_position,
                                      const ArrayView<double> &data,
                                      const unsigned int mineral_i,
                                      const unsigned int grain_i,
                                      const double strain_accumulated) const
          {
            data[cpo_data_position + 27 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = strain_accumulated;
          }

          /**
           * @brief Returns the value in the data array representing the energy ratio of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to get the value of the energy ratio of a grain for.
           * @param grain_i The grain to get the value of the energy ratio of.
           */
          inline
          double get_energy_ratio(const unsigned int cpo_data_position,
                                  const ArrayView<const double> &data,
                                  const unsigned int mineral_i,
                                  const unsigned int grain_i) const
          {
            return data[cpo_data_position + 28 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)];
          }

          /**
           * @brief Sets the value in the data array representing the energy ratio of a grain.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i The mineral to set the value of the energy ratio of a grain for.
           * @param grain_i The grain to set the value of the energy ratio of.
           * @param energy_ratio The value of the energy ratio of a grain to set.
           */
          inline
          void set_energy_ratio(const unsigned int cpo_data_position,
                                const ArrayView<double> &data,
                                const unsigned int mineral_i,
                                const unsigned int grain_i,
                                const double energy_ratio) const
          {
            data[cpo_data_position + 28 + grain_i * 24 + mineral_i * (n_grains * 24 + 5)] = energy_ratio;
          }

        private:
          /**
           * Computes a random rotation matrix.
           */
          void
          compute_random_rotation_matrix(Tensor<2,3> &rotation_matrix) const;

          /**
           * @brief Updates the volume fractions and rotation matrices with a Forward Euler scheme.
           *
           * Updates the volume fractions and rotation matrices with a Forward Euler scheme:
           * $x_t = x_{t-1} + dt * x_{t-1} * \frac{dx_t}{dt}$. The function returns the sum of
           * the new volume fractions.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i Which mineral to advect for.
           * @param dt The time step used for the advection step
           * @param derivatives A pair containing the derivatives for the volume fractions and
           * orientations respectively.
           * @return double The sum of all volume fractions.
           */
          double
          advect_forward_euler(const unsigned int cpo_data_position,
                               const ArrayView<double> &data,
                               const unsigned int mineral_i,
                               const double dt,
                               const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const;

          /**
           * @brief Updates the volume fractions and rotation matrices with a Backward Euler scheme.
           *
           * Updates the volume fractions and rotation matrices with a Backward Euler scheme:
           * $x_t = x_{t-1} + dt * x_{t-1} * \frac{dx_t}{dt}$. The function returns the sum of
           * the new volume fractions.
           *
           * @param cpo_data_position The starting index/position of the cpo data in the particle data vector.
           * @param data The particle data vector.
           * @param mineral_i Which mineral to advect for.
           * @param dt The time step used for the advection step
           * @param derivatives A pair containing the derivatives for the volume fractions and
           * orientations respectively.
           * @return double The sum of all volume fractions.
           */
          double
          advect_backward_euler(const unsigned int cpo_data_position,
                                const ArrayView<double> &data,
                                const unsigned int mineral_i,
                                const double dt,
                                const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const;


          /**
           * Computes and returns the volume fraction and grain orientation derivatives such that
           * the grains stay the same size and the orientations rotating passively with the particle.
           *
           * @param velocity_gradient_tensor is the velocity gradient tensor at the location of the particle.
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
           * Should be one of the following: "Olivine: Karato 2008", "Olivine: A-fabric",
           * "Olivine: B-fabric", "Olivine: C-fabric", "Olivine: D-fabric", "Olivine: E-fabric",
           * "Enstatite" or "Passive".
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
          /**
           * Stress exponent
           */
          double stress_exponent;

          /**
           * efficiency of nucleation parameter.
           * lambda_m in equation 8 of Kaminski et al. (2004, Geophys. J. Int)
           */
          double nucleation_efficiency;

          /**
           * An exponent described in equation 10 of Kaminski and Ribe (2001, EPSL)
           */
          double exponent_p;

          /**
           * The Dimensionless Grain Boundary Sliding (GBS) threshold.
           * This is a grain size threshold below which grain deform by GBS and
           * become strain-free grains.
           */
          double threshold_GBS;

          /**
           * Dimensionless grain boundary mobility as described by equation 14
           * in Kaminski and Ribe (2001, EPSL).
           */
          double mobility;

          /**
           * Sets which type of initial grain model is used to create the gain sizes and orientations
           */
          CPOInitialGrainsModel initial_grains_model;

          /**
           * Clinopyroxene Reference Resolved Shear Stress (RRSS);
           * The RRSS notation is defined in
           * Kaminski etal., 2004 and see Fraters and Billen 2021 for details;
           * Activated when 'Clinopyroxene' is set as a Mineral type in the input file.
           *
           * RRSS value for CPX does not exist in DRex formulation,
           * but Bascou etal., 2002 JSG using VPSC provides a reference, which is
           * supported by experimental work from Zhang et al., 2006 EPSL
           * that agree with the three main slip systems
          */
          std::vector<double> CPX_RRSS;

          /**
           * Alternative Reference Resolved Shear Stress (RRSS) for;
           * Olivine D type fabric slip systems with {0kl} slip plane;
           * Activated when 'Olivine: D-fabric_0kl' is set as a Mineral type in the input file
           *
           * The RRSS notation is defined in Kaminski etal., 2004 and
           * see Fraters and Billen 2021 for details;
           * Main slip systems from Karato 2008 and Bystricky et al., 2001
          */
          std::vector<double> OlivineD_RRSS;

          /** @} */
          double n_grains_init;
          double n_grains_buffer;
          double initial_grain_size;

          /**
          * Parameters required to initialize the grain size and orientations for options other than "Random Orientations"
          */

          std::string input_orientation_file; // File containing orientations that be used to initialize the distribution with.

          // For initializing the grain size with a normal distribution
          double normal_distribution_mean;
          double normal_distribution_standard_deviation;

          // For initializing the grain size with a lognormal distribution
          double lognormal_distribution_mean;
          double lognormal_distribution_standard_deviation;

          // For initializing the grain size with a uniform  distribution
          double uniform_distribution_max;
          double uniform_distribution_min;


          /**
          * Other D-Rex^{++} specific parameters
          *
          * drexpp_mobility - Dimensional grain boundary mobilities for different minerals
          * avrami_slope    - Constant rate of recrystalization.
          * interfacial_energy
          **/

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
