/*
  Copyright (C) 2022 by the authors of the ASPECT code.

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

//#include <cstdlib>
#include <aspect/particle/property/cpo_bingham_average.h>
#include <aspect/particle/property/crystal_preferred_orientation.h>
#include <aspect/particle/world.h>

#include <aspect/utilities.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {


      template <int dim>
      CpoBinghamAverage<dim>::CpoBinghamAverage ()
      {
        permutation_operator_3d[0][1][2]  = 1;
        permutation_operator_3d[1][2][0]  = 1;
        permutation_operator_3d[2][0][1]  = 1;
        permutation_operator_3d[0][2][1]  = -1;
        permutation_operator_3d[1][0][2]  = -1;
        permutation_operator_3d[2][1][0]  = -1;
      }

      template <int dim>
      void
      CpoBinghamAverage<dim>::initialize ()
      {
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);
        const auto &manager = this->get_particle_world().get_property_manager();
        AssertThrow(manager.plugin_name_exists("crystal preferred orientation"),
                    ExcMessage("No crystal preferred orientation property plugin found."));
        Assert(manager.plugin_name_exists("cpo bingham average"),
               ExcMessage("No bingham average property plugin found."));

        AssertThrow(manager.check_plugin_order("crystal preferred orientation","cpo bingham average"),
                    ExcMessage("To use the cpo bingham average plugin, the cpo plugin need to be defined before this plugin."));

        cpo_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("crystal preferred orientation"));

      }



      template <int dim>
      void
      CpoBinghamAverage<dim>::initialize_one_particle_property(const Point<dim> &,
                                                               std::vector<double> &data) const
      {

        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);

        // Get a reference to the CPO particle property.
        const Particle::Property::CrystalPreferredOrientation<dim> &cpo_particle_property =
          this->get_particle_world().get_property_manager().template get_matching_property<Particle::Property::CrystalPreferredOrientation<dim>>();

        std::vector<double> volume_fractions_grains(n_grains);
        std::vector<Tensor<2,3>> rotation_matrices_grains(n_grains);
        for (size_t mineral_i = 0; mineral_i < n_minerals; mineral_i++)
          {
            // create volume fractions and rotation matrix vectors in the order that it is stored in the data array
            for (size_t grain_i = 0; grain_i < n_grains; grain_i++)
              {
                volume_fractions_grains[grain_i] = cpo_particle_property.get_volume_fractions_grains(cpo_data_position,data,mineral_i,grain_i);
                rotation_matrices_grains[grain_i] = cpo_particle_property.get_rotation_matrix_grains(cpo_data_position,data,mineral_i,grain_i);
              }

            const std::vector<Tensor<2,3>> weighted_rotation_matrices = random_draw_volume_weighting(volume_fractions_grains, rotation_matrices_grains, n_samples);
            std::array<std::array<double,3>,3> bingham_average = compute_bingham_average(weighted_rotation_matrices);

            for (unsigned int i = 0; i < 3; i++)
              for (unsigned int j = 0; j < 3; j++)
                data.emplace_back(bingham_average[i][j]);
          }
      }

      template <int dim>
      void
      CpoBinghamAverage<dim>::update_one_particle_property(const unsigned int data_position,
                                                           const Point<dim> &,
                                                           const Vector<double> &,
                                                           const std::vector<Tensor<1,dim>> &,
                                                           const ArrayView<double> &data) const
      {
        // Get a reference to the CPO particle property.
        const Particle::Property::CrystalPreferredOrientation<dim> &cpo_particle_property =
          this->get_particle_world().get_property_manager().template get_matching_property<Particle::Property::CrystalPreferredOrientation<dim>>();

        std::vector<double> volume_fractions_grains(n_grains);
        std::vector<Tensor<2,3>> rotation_matrices_grains(n_grains);
        for (size_t mineral_i = 0; mineral_i < n_minerals; mineral_i++)
          {
            // create volume fractions and rotation matrix vectors in the order that it is stored in the data array
            for (size_t grain_i = 0; grain_i < n_grains; grain_i++)
              {
                volume_fractions_grains[grain_i] = cpo_particle_property.get_volume_fractions_grains(cpo_data_position,data,mineral_i,grain_i);
                rotation_matrices_grains[grain_i] = cpo_particle_property.get_rotation_matrix_grains(cpo_data_position,data,mineral_i,grain_i);
              }

            const std::vector<Tensor<2,3>> weighted_rotation_matrices = random_draw_volume_weighting(volume_fractions_grains, rotation_matrices_grains, n_samples);
            std::array<std::array<double,3>,3> bingham_average = compute_bingham_average(weighted_rotation_matrices);

            unsigned int counter = 0;
            for (unsigned int i = 0; i < 3; i++)
              for (unsigned int j = 0; j < 3; j++)
                {
                  data[data_position + mineral_i*9 + counter] = bingham_average[i][j];
                  counter++;
                }
          }
      }


      template<int dim>
      std::array<std::array<double,3>,3>
      CpoBinghamAverage<dim>::compute_bingham_average(std::vector<Tensor<2,3>> matrices) const
      {
        SymmetricTensor< 2, 3, double > sum_matrix_a;
        SymmetricTensor< 2, 3, double > sum_matrix_b;
        SymmetricTensor< 2, 3, double > sum_matrix_c;

        // extracting the a, b and c orientations from the olivine a matrix
        // see https://courses.eas.ualberta.ca/eas421/lecturepages/orientation.html
        for (unsigned int i_grain = 0; i_grain < matrices.size(); i_grain++)
          {
            sum_matrix_a[0][0] += matrices[i_grain][0][0] * matrices[i_grain][0][0]; // SUM(l^2)
            sum_matrix_a[1][1] += matrices[i_grain][0][1] * matrices[i_grain][0][1]; // SUM(m^2)
            sum_matrix_a[2][2] += matrices[i_grain][0][2] * matrices[i_grain][0][2]; // SUM(n^2)
            sum_matrix_a[0][1] += matrices[i_grain][0][0] * matrices[i_grain][0][1]; // SUM(l*m)
            sum_matrix_a[0][2] += matrices[i_grain][0][0] * matrices[i_grain][0][2]; // SUM(l*n)
            sum_matrix_a[1][2] += matrices[i_grain][0][1] * matrices[i_grain][0][2]; // SUM(m*n)


            sum_matrix_b[0][0] += matrices[i_grain][1][0] * matrices[i_grain][1][0]; // SUM(l^2)
            sum_matrix_b[1][1] += matrices[i_grain][1][1] * matrices[i_grain][1][1]; // SUM(m^2)
            sum_matrix_b[2][2] += matrices[i_grain][1][2] * matrices[i_grain][1][2]; // SUM(n^2)
            sum_matrix_b[0][1] += matrices[i_grain][1][0] * matrices[i_grain][1][1]; // SUM(l*m)
            sum_matrix_b[0][2] += matrices[i_grain][1][0] * matrices[i_grain][1][2]; // SUM(l*n)
            sum_matrix_b[1][2] += matrices[i_grain][1][1] * matrices[i_grain][1][2]; // SUM(m*n)


            sum_matrix_c[0][0] += matrices[i_grain][2][0] * matrices[i_grain][2][0]; // SUM(l^2)
            sum_matrix_c[1][1] += matrices[i_grain][2][1] * matrices[i_grain][2][1]; // SUM(m^2)
            sum_matrix_c[2][2] += matrices[i_grain][2][2] * matrices[i_grain][2][2]; // SUM(n^2)
            sum_matrix_c[0][1] += matrices[i_grain][2][0] * matrices[i_grain][2][1]; // SUM(l*m)
            sum_matrix_c[0][2] += matrices[i_grain][2][0] * matrices[i_grain][2][2]; // SUM(l*n)
            sum_matrix_c[1][2] += matrices[i_grain][2][1] * matrices[i_grain][2][2]; // SUM(m*n)

          }
        const std::array<std::pair<double,Tensor<1,3,double>>, 3> eigenvectors_a = eigenvectors(sum_matrix_a, SymmetricTensorEigenvectorMethod::jacobi);
        const std::array<std::pair<double,Tensor<1,3,double>>, 3> eigenvectors_b = eigenvectors(sum_matrix_b, SymmetricTensorEigenvectorMethod::jacobi);
        const std::array<std::pair<double,Tensor<1,3,double>>, 3> eigenvectors_c = eigenvectors(sum_matrix_c, SymmetricTensorEigenvectorMethod::jacobi);


        const Tensor<1,3,double> averaged_a = eigenvectors_a[0].second * eigenvectors_a[0].first;
        const Tensor<1,3,double> averaged_b = eigenvectors_b[0].second * eigenvectors_b[0].first;
        const Tensor<1,3,double> averaged_c = eigenvectors_c[0].second * eigenvectors_a[0].first;

        return
        {
          {
            {{averaged_a[0],averaged_a[1],averaged_a[2]}},
            {{averaged_b[0],averaged_b[1],averaged_b[2]}},
            {{averaged_c[0],averaged_c[1],averaged_c[2]}}
          }
        };
      }

      template<int dim>
      std::vector<Tensor<2,3>>
      CpoBinghamAverage<dim>::random_draw_volume_weighting(std::vector<double> fv,
                                                           std::vector<Tensor<2,3>> matrices,
                                                           unsigned int n_output_grains) const
      {
        // Get volume weighted euler angles, using random draws to convert odf
        // to a discrete number of orientations, weighted by volume
        // 1a. Get index that would sort volume fractions AND
        //ix = np.argsort(fv[q,:]);
        // 1b. Get the sorted volume and angle arrays
        std::vector<double> fv_to_sort = fv;
        std::vector<double> fv_sorted = fv;
        std::vector<Tensor<2,3>> matrices_sorted = matrices;

        unsigned int n_grain = fv_to_sort.size();


        /**
         * ...
         */
        for (int i = n_grain-1; i >= 0; --i)
          {
            unsigned int index_max_fv = std::distance(fv_to_sort.begin(),max_element(fv_to_sort.begin(), fv_to_sort.end()));

            fv_sorted[i] = fv_to_sort[index_max_fv];
            matrices_sorted[i] = matrices[index_max_fv];
            fv_to_sort[index_max_fv] = -1;
          }

        // 2. Get cumulative weight for volume fraction
        std::vector<double> cum_weight(n_grains);
        std::partial_sum(fv_sorted.begin(),fv_sorted.end(),cum_weight.begin());
        // 3. Generate random indices
        std::uniform_real_distribution<> dist(0, 1);
        std::vector<double> idxgrain(n_output_grains);
        for (unsigned int grain_i = 0; grain_i < n_output_grains; ++grain_i)
          {
            idxgrain[grain_i] = dist(this->random_number_generator);
          }

        // 4. Find the maximum cum_weight that is less than the random value.
        // the euler angle index is +1. For example, if the idxGrain(g) < cumWeight(1),
        // the index should be 1 not zero)
        std::vector<Tensor<2,3>> matrices_out(n_output_grains);
        for (unsigned int grain_i = 0; grain_i < n_output_grains; ++grain_i)
          {
            unsigned int counter = 0;
            for (unsigned int grain_j = 0; grain_j < n_grains; ++grain_j)
              {
                // find the first cummulative weight which is larger than the random number
                // todo: there may be algorithms to do this faster
                if (cum_weight[grain_j] < idxgrain[grain_i])
                  {
                    counter++;
                  }
                else
                  {
                    break;
                  }
              }
            matrices_out[grain_i] = matrices_sorted[counter];
          }
        return matrices_out;
      }



      template <int dim>
      UpdateTimeFlags
      CpoBinghamAverage<dim>::need_update() const
      {
        return update_output_step;
      }

      template <int dim>
      UpdateFlags
      CpoBinghamAverage<dim>::get_needed_update_flags () const
      {
        return update_default;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      CpoBinghamAverage<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;
        for (size_t mineral_i = 0; mineral_i < n_minerals; mineral_i++)
          {
            property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " bingham average a axis",3));
            property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " bingham average b axis",3));
            property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " bingham average c axis",3));
          }

        return property_information;
      }

      template <int dim>
      void
      CpoBinghamAverage<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("CpoBinghamAverage");
            {
              prm.declare_entry ("Random number seed", "1",
                                 Patterns::Integer (0),
                                 "The seed used to generate random numbers. This will make sure that "
                                 "results are reproducable as long as the problem is run with the "
                                 "same amount of MPI processes. It is implemented as final seed = "
                                 "user seed + MPI Rank. ");

              prm.declare_entry ("Number of samples", "0",
                                 Patterns::Double(0),
                                 "This determines how many samples are taken when using the random "
                                 "draw volume averaging. Setting it to zero means that the number of "
                                 "samples is set to be equal to the number of grains.");
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
      }


      template <int dim>
      void
      CpoBinghamAverage<dim>::parse_parameters (ParameterHandler &prm)
      {

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("CpoBinghamAverage");
            {
              // Get a reference to the CPO particle property.
              const Particle::Property::CrystalPreferredOrientation<dim> &cpo_particle_property =
                this->get_particle_world().get_property_manager().template get_matching_property<Particle::Property::CrystalPreferredOrientation<dim>>();

              random_number_seed = prm.get_integer ("Random number seed");
              n_grains = cpo_particle_property.get_number_of_grains();
              n_minerals = cpo_particle_property.get_number_of_minerals();
              n_samples = prm.get_integer("Number of samples");
              if (n_samples == 0)
                n_samples = n_grains;
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();


      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(CpoBinghamAverage,
                                        "cpo bingham average",
                                        "This is a particle property plugin which computes the Bingham "
                                        "average for the Crystal Preferred Orienation particle property "
                                        "plugin so that it can be visualized.")
    }
  }
}
