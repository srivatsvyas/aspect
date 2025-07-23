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

#include <aspect/particle/property/crystal_preferred_orientation.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/utilities.h>

#ifdef ASPECT_WITH_WORLD_BUILDER
#include <world_builder/grains.h>
#include <world_builder/world.h>
#endif

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {

      template <int dim>
      void
      CrystalPreferredOrientation<dim>::initialize ()
      {
        CitationInfo::add("CPO");
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);

        // Don't assert when called by the unit tester.
        if (this->simulator_is_past_initialization())
          {
            AssertThrow(this->introspection().compositional_name_exists("water"),
                        ExcMessage("Particle property CPO only works if"
                                   "there is a compositional field called water."));
            water_index = this->introspection().compositional_index_for_name("water");
          }
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::compute_random_rotation_matrix(Tensor<2,3> &rotation_matrix) const
      {

        // This function is based on an article in Graphic Gems III, written by James Arvo, Cornell University (p 116-120).
        // The original code can be found on  http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        // and is licenced according to this website with the following licence:
        //
        // "The Graphics Gems code is copyright-protected. In other words, you cannot claim the text of the code as your own and
        // resell it. Using the code is permitted in any program, product, or library, non-commercial or commercial. Giving credit
        // is not required, though is a nice gesture. The code comes as-is, and if there are any flaws or problems with any Gems
        // code, nobody involved with Gems - authors, editors, publishers, or webmasters - are to be held responsible. Basically,
        // don't be a jerk, and remember that anything free comes with no guarantee.""
        //
        // The book states in the preface the following: "As in the first two volumes, all of the C and C++ code in this book is in
        // the public domain, and is yours to study, modify, and use."

        // first generate three random numbers between 0 and 1 and multiply them with 2 PI or 2 for z. Note that these are not the same as phi_1, theta and phi_2.

       // I want to add option to add initial fabric
        boost::random::uniform_real_distribution<double> uniform_distribution(0,1);
        boost::random::uniform_real_distribution<double> uniform_distribution1(-numbers::PI/12,numbers::PI/12);
        double one = uniform_distribution(this->random_number_generator);
        double two = uniform_distribution(this->random_number_generator);
        double three = uniform_distribution(this->random_number_generator);
      
        double theta = 2.0 * M_PI * one;
        double phi = 2.0 * M_PI * two; // For direction of pole deflection.
        double z = 2.0* three; //For magnitude of pole deflection.

        // Compute a vector V used for distributing points over the sphere
        // via the reflection I - V Transpose(V).  This formulation of V
        // will guarantee that if x[1] and x[2] are uniformly distributed,
        // the reflected points will be uniform on the sphere.  Note that V
        // has length sqrt(2) to eliminate the 2 in the Householder matrix.

        double r  = std::sqrt( z );
        double Vx = std::sin( phi ) * r;
        double Vy = std::cos( phi ) * r;
        double Vz = std::sqrt( 2.f - z );

        // Compute the row vector S = Transpose(V) * R, where R is a simple
        // rotation by theta about the z-axis.  No need to compute Sz since
        // it's just Vz.

        double st = std::sin( theta );
        double ct = std::cos( theta );
        double Sx = Vx * ct - Vy * st;
        double Sy = Vx * st + Vy * ct;

        // Construct the rotation matrix  ( V Transpose(V) - I ) R, which
        // is equivalent to V S - R.

        rotation_matrix[0][0] = Vx * Sx - ct;
        rotation_matrix[0][1] = Vx * Sy - st;
        rotation_matrix[0][2] = Vx * Vz;
        rotation_matrix[1][0] = Vy * Sx + st;
        rotation_matrix[1][1] = Vy * Sy - ct;
        rotation_matrix[1][2] = Vy * Vz;
        rotation_matrix[2][0] = Vz * Sx;
        rotation_matrix[2][1] = Vz * Sy;
        rotation_matrix[2][2] = 1.0 - z;   // This equals Vz * Vz - 1.0
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                                         std::vector<double> &data) const
      {
        // the layout of the data vector per particle is the following:
        // 1. M mineral times
        //    1.1  olivine deformation type   -> 1 double, at location
        //                                      => data_position + 0 + mineral_i * (n_grains * 10 + 2)
        //    2.1. Mineral volume fraction    -> 1 double, at location
        //                                      => data_position + 1 + mineral_i *(n_grains * 10 + 2)
        //    2.2. N grains times:
        //         2.1. volume fraction grain -> 1 double, at location:
        //                                      => data_position + 2 + i_grain * 10 + mineral_i *(n_grains * 10 + 2), or
        //                                      => data_position + 2 + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.2. rotation matrix grain -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
        //                                      => data_position + 3 + i_grain * 10 + mineral_i * (n_grains * 10 + 2), or
        //                                      => data_position + 3 + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //
        // Note that we store exactly the same number of grains of all minerals (e.g. olivine and enstatite
        // grains), although their volume fractions may not be the same. We need a minimum amount
        // of grains per particle to perform reliable statistics on it. This minimum is the same for all phases.
        // and enstatite.
        //
        // Furthermore, for this plugin the following dims are always 3. When using 2d an infinitely thin 3d domain is assumed.
        //
        // The rotation matrix is a direction cosine matrix, representing the orientation of the grain in the domain.
        // The fabric is determined later in the computations, so initialize it to -1.

        std::vector<double> deformation_type(n_minerals, -1.0);
        std::vector<double> bulk_recrystalized_grain_size(n_minerals, 0.0);
        std::vector<double> mean_grain_size(n_minerals);
        std::vector<std::vector<double >>volume_fractions_grains(n_minerals);
        std::vector<std::vector<Tensor<2,3>>> rotation_matrices_grains(n_minerals);
        std::vector<std::vector<int    >>grain_status(n_minerals);
        std::vector<std::vector<double >>viscosity_ratio(n_minerals);
        std::vector<std::vector<double >>strain_rate_ratio(n_minerals);
        std::vector<std::vector<double >>rx_fractions(n_minerals);
        std::vector<std::vector<int    >>active_slip_system(n_minerals);
        std::vector<std::vector<double >>strain_rate_grains(n_minerals);
        std::vector<std::vector<double >>differential_stress(n_minerals);
        std::vector<std::vector<double >>strain_energy(n_minerals);
        std::vector<std::vector<double >>surface_energy(n_minerals);
        std::vector<std::vector<double >>grain_boundary_velocity(n_minerals);
        std::vector<std::vector<double >>grain_size_change(n_minerals);
        std::vector<std::vector<double >>dislocation_density(n_minerals);
        std::vector<std::vector<double >>strain_accumulated(n_minerals);
        std::vector<std::vector<double >>energy_ratio(n_minerals);

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            volume_fractions_grains[mineral_i].resize(n_grains);
            rotation_matrices_grains[mineral_i].resize(n_grains);
            grain_status[mineral_i].resize(n_grains);
            strain_rate_ratio[mineral_i].resize(n_grains);
            viscosity_ratio[mineral_i].resize(n_grains);
            rx_fractions[mineral_i].resize(n_grains);
            active_slip_system[mineral_i].resize(n_grains);
            strain_rate_grains[mineral_i].resize(n_grains);
            differential_stress[mineral_i].resize(n_grains);
            strain_energy[mineral_i].resize(n_grains);
            surface_energy[mineral_i].resize(n_grains);
            grain_boundary_velocity[mineral_i].resize(n_grains);
            grain_size_change[mineral_i].resize(n_grains);
            dislocation_density[mineral_i].resize(n_grains);
            strain_accumulated[mineral_i].resize(n_grains);
            energy_ratio[mineral_i].resize(n_grains);


            // This will be set by the initial grain subsection.
            if (initial_grains_model == CPOInitialGrainsModel::world_builder)
              {
#ifdef ASPECT_WITH_WORLD_BUILDER
                WorldBuilder::grains wb_grains = this->get_world_builder().grains(Utilities::convert_point_to_array(position),
                                                                                  -this->get_geometry_model().height_above_reference_surface(position),
                                                                                  mineral_i,
                                                                                  n_grains);
                double sum_volume_fractions = 0;
                for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
                  {
                    sum_volume_fractions += wb_grains.sizes[grain_i];
                    volume_fractions_grains[mineral_i][grain_i] = wb_grains.sizes[grain_i];
                    // we are receiving a array<array<double,3>,3> from the world builder,
                    // which needs to be copied in the correct way into a tensor<2,3>.
                    for (unsigned int component_i = 0; component_i < 3 ; ++component_i)
                      {
                        for (unsigned int component_j = 0; component_j < 3 ; ++component_j)
                          {
                            Assert(!std::isnan(wb_grains.rotation_matrices[grain_i][component_i][component_j]), ExcMessage("Error: not a number."));
                            rotation_matrices_grains[mineral_i][grain_i][component_i][component_j] = wb_grains.rotation_matrices[grain_i][component_i][component_j];
                          }
                      }
                  }

                AssertThrow(sum_volume_fractions != 0, ExcMessage("Sum of volumes is equal to zero, which is not supposed to happen. "
                                                                  "Make sure that all parts of the domain which contain particles are covered by the world builder."));
#else
                AssertThrow(false,
                            ExcMessage("The world builder was requested but not provided. Make sure that aspect is "
                                       "compiled with the World Builder and that you provide a world builder file in the input."));
                // this is to avoid a warning about unused variables
                (void)position;
#endif
              }
            else
              { 
                if(cpo_derivative_algorithm == CPODerivativeAlgorithm::drexpp)
                {
                  std::vector<std::array<double,3>>orientation; // declaring this vector ahead of time in the case of initializing D-Rex++ with fabric from an input file
                  if(initial_grains_model == CPOInitialGrainsModel::pre_existing_fabric)
                    {
                       std::ifstream file(input_orientation_file);
                       if(!file)
                       {
                         std::cerr<< "Error: Unable to open file!\n";
                         assert(false && 'File parsing error');
                       }

                       double phi1, phi2, phi3;
                       std::string line;

                       while(std::getline(file,line))
                       {
                          // Read file line by line
                          std::istringstream iss(line);
                          if(!(iss >>phi1 >>phi2 >>phi3))
                          {
                            // Parse three space-separated values
                            std::cerr<<"Error: Failed to parse line: "<<line<<std::endl;
                            assert(false && "File parsing error");
                          }
                          orientation.push_back({{phi1, phi2, phi3}});
                          file.close();
                       }
                     }

                  for(unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
                  {
                    if(grain_i < n_grains_init)
                    {
                      if(initial_grains_model == CPOInitialGrainsModel::normal_distribution_grainsize_and_random_initial_orientations)
                      {
                        std::normal_distribution<double> distribution(normal_distribution_mean,normal_distribution_standard_deviation); 
                        volume_fractions_grains[mineral_i][grain_i] = distribution(this->random_number_generator);
                        while(volume_fractions_grains[mineral_i][grain_i] <= 0.0)
                        {
                          volume_fractions_grains[mineral_i][grain_i] = distribution(this->random_number_generator);
                        }
                      }
                      else
                      {
                        volume_fractions_grains[mineral_i][grain_i] = initial_grain_size;
                      }
                      grain_status[mineral_i][grain_i]= 0;
                    }
                    else
                    if(grain_i >= n_grains - n_grains_buffer)
                    {
                      volume_fractions_grains[mineral_i][grain_i] = 0.;
                      grain_status[mineral_i][grain_i] = -2;
                    }
                    else
                    { 
                      volume_fractions_grains[mineral_i][grain_i] = 0.;
                      grain_status[mineral_i][grain_i] = -1;
                    }
                    
                    if(initial_grains_model == CPOInitialGrainsModel::pre_existing_fabric)
                    {
                       Tensor<2,3> dcmg;
                       dcmg = aspect::Utilities::zxz_euler_angles_to_rotation_matrix(orientation[grain_i][0],orientation[grain_i][1],orientation[grain_i][2]);
                       rotation_matrices_grains[mineral_i][grain_i] = dcmg;
                    }
                    else
                    {
                      this->compute_random_rotation_matrix(rotation_matrices_grains[mineral_i][grain_i]);
                    }
                      
                      strain_rate_ratio[mineral_i][grain_i] = 0.;
                      viscosity_ratio[mineral_i][grain_i] = 0.;
                      rx_fractions[mineral_i][grain_i] = 0.;
                      active_slip_system[mineral_i][grain_i] = 0;
                      strain_rate_grains[mineral_i][grain_i] = 0.;
                      differential_stress[mineral_i][grain_i] = 0.;
                      strain_energy[mineral_i][grain_i] = 0.;
                      surface_energy[mineral_i][grain_i] =0.;
                      grain_boundary_velocity[mineral_i][grain_i] = 0.;
                      grain_size_change[mineral_i][grain_i]= 0.;
                      dislocation_density[mineral_i][grain_i] =0.;
                      strain_accumulated[mineral_i][grain_i] =0.;
                      energy_ratio[mineral_i][grain_i] =0.;
                  }
                  
                  int grains  = 0;
                  double sum_grain_size = 0.0;
                  // Calculating the mean grain size for the mineral phase in question
                  for(unsigned int grain_i = 0; grain_i < n_grains_init; ++grain_i)
                  {
                    sum_grain_size += 1.0/volume_fractions_grains[mineral_i][grain_i];
                    grains += 1;
                    
                  }
                  
                  mean_grain_size[mineral_i] = std::pow(sum_grain_size/grains,-1.0);
                  std::cout<<"initial mean grain size = "<<mean_grain_size[mineral_i]<<std::endl;
                }
                else
                {
                  // set volume fraction
                  const double initial_volume_fraction = 1.0/n_grains;

                  for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
                    {
                      // set volume fraction
                      volume_fractions_grains[mineral_i][grain_i] = initial_volume_fraction;

                      // set a uniform random rotation_matrix per grain
                      this->compute_random_rotation_matrix(rotation_matrices_grains[mineral_i][grain_i]);
                      grain_status[mineral_i][grain_i] = 0;
                      strain_rate_ratio[mineral_i][grain_i] = 0.;
                      viscosity_ratio[mineral_i][grain_i] = 0.;
                      rx_fractions[mineral_i][grain_i] = 0.;
                      active_slip_system[mineral_i][grain_i] = 0;
                      strain_rate_grains[mineral_i][grain_i] = 0.;
                      differential_stress[mineral_i][grain_i] = 0.;
                      strain_energy[mineral_i][grain_i] = 0.;
                      surface_energy[mineral_i][grain_i] =0.;
                      grain_boundary_velocity[mineral_i][grain_i] = 0.;
                      grain_size_change[mineral_i][grain_i]= 0.;
                      dislocation_density[mineral_i][grain_i] =0.;
                      strain_accumulated[mineral_i][grain_i] =0.;
                      energy_ratio[mineral_i][grain_i] =0.;
                    }
                }
                
              }
          }

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            data.emplace_back(deformation_type[mineral_i]);
            data.emplace_back(volume_fractions_minerals[mineral_i]);
            data.emplace_back(bulk_recrystalized_grain_size[mineral_i]);
            data.emplace_back(mean_grain_size[mineral_i]);
            for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
              {
                data.emplace_back(volume_fractions_grains[mineral_i][grain_i]);
                for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
                  {
                    const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                    data.emplace_back(rotation_matrices_grains[mineral_i][grain_i][index]);
                  }
                    data.emplace_back(grain_status[mineral_i][grain_i]);
                    data.emplace_back(strain_rate_ratio[mineral_i][grain_i]);
                    data.emplace_back(viscosity_ratio[mineral_i][grain_i]);
                    data.emplace_back(rx_fractions[mineral_i][grain_i]);
                    data.emplace_back(active_slip_system[mineral_i][grain_i]);
                    data.emplace_back(strain_rate_grains[mineral_i][grain_i]);
                    data.emplace_back(differential_stress[mineral_i][grain_i]);
                    data.emplace_back(strain_energy[mineral_i][grain_i]);
                    data.emplace_back(surface_energy[mineral_i][grain_i]);
                    data.emplace_back(grain_boundary_velocity[mineral_i][grain_i]);
                    data.emplace_back(grain_size_change[mineral_i][grain_i]);
                    data.emplace_back(dislocation_density[mineral_i][grain_i]);
                    data.emplace_back(strain_accumulated[mineral_i][grain_i]);
                    data.emplace_back(energy_ratio[mineral_i][grain_i]);
                }
            
          }
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::update_particle_properties(const ParticleUpdateInputs<dim> &inputs,
                                                                   typename ParticleHandler<dim>::particle_iterator_range &particles) const
      {
        const unsigned int data_position = this->data_position;
        std::vector<double> compositions(this->n_compositional_fields());

        unsigned int p = 0;
        for (auto &particle: particles)
          {
            // STEP 1: Load data and preprocess it.

            // need access to the pressure, viscosity,
            // get velocity
            Tensor<1,dim> velocity;
            for (unsigned int i = 0; i < dim; ++i)
              velocity[i] = inputs.solution[p][this->introspection().component_indices.velocities[i]];

            // get velocity gradient tensor.
            Tensor<2,dim> velocity_gradient;
            for (unsigned int i = 0; i < dim; ++i)
              velocity_gradient[i] = inputs.gradients[p][this->introspection().component_indices.velocities[i]];

            // Calculate strain rate from velocity gradients
            const SymmetricTensor<2,dim> strain_rate = symmetrize (velocity_gradient);
            const SymmetricTensor<2,dim> deviatoric_strain_rate
              = (this->get_material_model().is_compressible()
                 ?
                 strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                 :
                 strain_rate);

            const double pressure = inputs.solution[p][this->introspection().component_indices.pressure];
            const double temperature = inputs.solution[p][this->introspection().component_indices.temperature];
            const double water_content = inputs.solution[p][this->introspection().component_indices.compositional_fields[water_index]];

            // get the composition of the particle
            for (unsigned int i = 0; i < this->n_compositional_fields(); ++i)
              {
                const unsigned int solution_component = this->introspection().component_indices.compositional_fields[i];
                compositions[i] = inputs.solution[p][solution_component];
              }

            const double dt = this->get_timestep();

            // even in 2d we need 3d strain-rates and velocity gradient tensors. So we make them 3d by
            // adding an extra dimension which is zero.
            SymmetricTensor<2,3> strain_rate_3d;
            strain_rate_3d[0][0] = strain_rate[0][0];
            strain_rate_3d[0][1] = strain_rate[0][1];
            //sym: strain_rate_3d[1][0] = strain_rate[1][0];
            strain_rate_3d[1][1] = strain_rate[1][1];

            if (dim == 3)
              {
                strain_rate_3d[0][2] = strain_rate[0][2];
                strain_rate_3d[1][2] = strain_rate[1][2];
                //sym: strain_rate_3d[2][0] = strain_rate[0][2];
                //sym: strain_rate_3d[2][1] = strain_rate[1][2];
                strain_rate_3d[2][2] = strain_rate[2][2];
              }
            Tensor<2,3> velocity_gradient_3d;
            velocity_gradient_3d[0][0] = velocity_gradient[0][0];
            velocity_gradient_3d[0][1] = velocity_gradient[0][1];
            velocity_gradient_3d[1][0] = velocity_gradient[1][0];
            velocity_gradient_3d[1][1] = velocity_gradient[1][1];
            if (dim == 3)
              {
                velocity_gradient_3d[0][2] = velocity_gradient[0][2];
                velocity_gradient_3d[1][2] = velocity_gradient[1][2];
                velocity_gradient_3d[2][0] = velocity_gradient[2][0];
                velocity_gradient_3d[2][1] = velocity_gradient[2][1];
                velocity_gradient_3d[2][2] = velocity_gradient[2][2];
              }

            ArrayView<double> data = particle.get_properties();
            const typename DoFHandler<dim>::active_cell_iterator cell(*particle.get_surrounding_cell(),&(this->get_dof_handler()));

            for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
              {

                /**
                * Now we have loaded all the data and can do the actual computation.
                * The computation consists of two parts. The first part is computing
                * the derivatives for the directions and grain sizes. Then those
                * derivatives are used to advect the particle properties.
                */
                double sum_volume_mineral = 0;
                std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
                derivatives_grains = this->compute_derivatives(data_position,
                                                               data,
                                                               mineral_i,
                                                               strain_rate_3d,
                                                               velocity_gradient_3d,
                                                               particle.get_location(),
                                                               cell,
                                                               temperature,
                                                               pressure,
                                                               velocity,
                                                               compositions,
                                                               strain_rate,
                                                               deviatoric_strain_rate,
                                                               water_content);

                switch (advection_method)
                  {
                    case AdvectionMethod::forward_euler:

                      sum_volume_mineral = this->advect_forward_euler(data_position,
                                                                      data,
                                                                      mineral_i,
                                                                      dt,
                                                                      derivatives_grains);

                      break;

                    case AdvectionMethod::backward_euler:
                      sum_volume_mineral = this->advect_backward_euler(data_position,
                                                                       data,
                                                                       mineral_i,
                                                                       dt,
                                                                       derivatives_grains);

                      break;
                  }

                // normalize the volume fractions back to a total of 1 for each mineral
                double inv_sum_volume_mineral;
                
                if(cpo_derivative_algorithm == CPODerivativeAlgorithm::drexpp)
                   {
                    inv_sum_volume_mineral = 1.0;
                   }
                else
                   {
                    inv_sum_volume_mineral = 1.0/sum_volume_mineral;
                   }

                Assert(std::isfinite(inv_sum_volume_mineral),
                       ExcMessage("inv_sum_volume_mineral is not finite. sum_volume_enstatite = "
                                  + std::to_string(sum_volume_mineral)));

                for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
                  {
                    const double volume_fraction_grains = get_volume_fractions_grains(data_position,data,mineral_i,grain_i)*inv_sum_volume_mineral;
                    set_volume_fractions_grains(data_position,data,mineral_i,grain_i,volume_fraction_grains);
                   
                    Assert(isfinite(get_volume_fractions_grains(data_position,data,mineral_i,grain_i)),
                           ExcMessage("volume_fractions_grains[mineral_i]" + std::to_string(grain_i) + "] is not finite: "
                                      + std::to_string(get_volume_fractions_grains(data_position,data,mineral_i,grain_i)) + ", inv_sum_volume_mineral = "
                                      + std::to_string(inv_sum_volume_mineral) + "."));

                    /**
                     * Correct direction rotation matrices numerical error (orthnormality) after integration
                     * Follows same method as in matlab version from Thissen (see https://github.com/cthissen/Drex-MATLAB/)
                     * of finding the nearest orthonormal matrix using the SVD
                     */
                    Tensor<2,3> rotation_matrix = get_rotation_matrix_grains(data_position,data,mineral_i,grain_i);
                    

                    rotation_matrix = dealii::project_onto_orthogonal_tensors(rotation_matrix);
                    set_rotation_matrix_grains(data_position,data,mineral_i,grain_i,rotation_matrix);
                    for (size_t i = 0; i < 3; ++i)
                      for (size_t j = 0; j < 3; ++j)
                        {
                          // I don't think this should happen with the projection, but D-Rex
                          // does not do the orthogonal projection, but just clamps the values
                          // to 1 and -1.
                          
                          Assert(std::fabs(rotation_matrix[i][j]) <= 1.0,
                                 ExcMessage("The rotation_matrix has a entry larger than 1."));

                          Assert(!std::isnan(rotation_matrix[i][j]),
                                 ExcMessage("rotation_matrix is nan after orthoganalization: "
                                            + std::to_string(rotation_matrix[i][j])));

                          Assert(std::abs(rotation_matrix[i][j]) <= 1.0,
                                 ExcMessage("3. rotation_matrix[" + std::to_string(i) + "][" + std::to_string(j) +
                                            "] is larger than one: "
                                            + std::to_string(rotation_matrix[i][j]) + " (" + std::to_string(rotation_matrix[i][j]-1.0) + "). rotation_matrix = \n"
                                            + std::to_string(rotation_matrix[0][0]) + " " + std::to_string(rotation_matrix[0][1]) + " " + std::to_string(rotation_matrix[0][2]) + "\n"
                                            + std::to_string(rotation_matrix[1][0]) + " " + std::to_string(rotation_matrix[1][1]) + " " + std::to_string(rotation_matrix[1][2]) + "\n"
                                            + std::to_string(rotation_matrix[2][0]) + " " + std::to_string(rotation_matrix[2][1]) + " " + std::to_string(rotation_matrix[2][2])));
                        }
                    set_rotation_matrix_grains(data_position,data,mineral_i,grain_i,rotation_matrix);
                  }
              }
            ++p;
          }
      }



      template <int dim>
      UpdateTimeFlags
      CrystalPreferredOrientation<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      InitializationModeForLateParticles
      CrystalPreferredOrientation<dim>::late_initialization_mode () const
      {
        return InitializationModeForLateParticles::interpolate;
      }



      template <int dim>
      UpdateFlags
      CrystalPreferredOrientation<dim>::get_update_flags (const unsigned int component) const
      {
        if (this->introspection().component_masks.velocities[component] == true)
          return update_values | update_gradients;

        return update_values;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      CrystalPreferredOrientation<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;
        property_information.reserve(n_minerals * n_grains * (1+Tensor<2,3>::n_independent_components));

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " type",1);
            property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " volume fraction",1);
            property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " bulk recrystalized grain size",1);
            property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain size",1);

            for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
              {
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " volume fraction",1);

                for (unsigned int index = 0; index < Tensor<2,3>::n_independent_components; ++index)
                  {
                    property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " rotation_matrix " + std::to_string(index),1);
                  }
                
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " grain status",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " strain rate ratio",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " viscosity ratio",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " recrystalized fractions",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " active slip system",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " strain rate",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " differential stress",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " strain energy",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " surface energy",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " grain boundary velocity",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " grain size change",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " dislocation density",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " strain accumulated",1);
                property_information.emplace_back("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " energy_ratio",1);
              }
          }

        return property_information;
      }



      template <int dim>
      double
      CrystalPreferredOrientation<dim>::advect_forward_euler(const unsigned int cpo_index,
                                                             const ArrayView<double> &data,
                                                             const unsigned int mineral_i,
                                                             const double dt,
                                                             const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const
      {
        double sum_volume_fractions = 0;
        Tensor<2,3> rotation_matrix;
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Do the volume fraction of the grain
            Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite before it is set."));
            double volume_fraction_grains = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            volume_fraction_grains =  volume_fraction_grains + dt * volume_fraction_grains * derivatives.first[grain_i];
            set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i, volume_fraction_grains);
            Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                   + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                   + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

            sum_volume_fractions += get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);

            // Do the rotation matrix for this grain
            rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            rotation_matrix += dt * rotation_matrix * derivatives.second[grain_i];
            set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,rotation_matrix);
          }
 
        Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
        return sum_volume_fractions;
      }



      template <int dim>
      double
      CrystalPreferredOrientation<dim>::advect_backward_euler(const unsigned int cpo_index,
                                                              const ArrayView<double> &data,
                                                              const unsigned int mineral_i,
                                                              const double dt,
                                                              const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const
      {
        switch(cpo_derivative_algorithm)
        {
          case CPODerivativeAlgorithm::drex_2004:
          {
            double sum_volume_fractions = 0;
            Tensor<2,3> cosine_ref;
            for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
              {
                // Do the volume fraction of the grain
                double vf_old = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                double vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set."));
                for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
                  {
                    Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set. grain_i = "
                                                        + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                                                        + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

                    vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) + dt * vf_new * derivatives.first[grain_i];

                    Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                           + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                           + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));
                    if (std::fabs(vf_new-vf_old) < property_advection_tolerance)
                      {
                        break;
                      }
                    vf_old = vf_new;
              }

            set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i,vf_new);
            sum_volume_fractions += vf_new;

            // Do the rotation matrix for this grain
            cosine_ref = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            Tensor<2,3> cosine_old = cosine_ref;
            Tensor<2,3> cosine_new = cosine_ref;

            for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
              {
                cosine_new = cosine_ref + dt * cosine_new * derivatives.second[grain_i];
                if ((cosine_new-cosine_old).norm() < property_advection_tolerance)
                  {
                    break;
                  }
                cosine_old = cosine_new;
              }
              set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,cosine_new);
            } 
            Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
            return sum_volume_fractions;
            break;    
          }
          case CPODerivativeAlgorithm::drexpp:
            {
              double area_sum = 0;
              double sum_volume_fractions = 0;
              double  sum_of_volumes = 0.;
              Tensor<2,3> cosine_ref;

              for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
                {
                  //std::cout<<"grain size of grain "<<grain_i<<" = "<<get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)<<"\t position = "<<cpo_index + 3 + grain_i * 32 + mineral_i * (n_grains * 32 + 2)<<std::endl;
                  area_sum += numbers::PI * std::pow(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)* 0.5,2.);
                }

              for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
                {
                  double grain_status = get_grain_status(cpo_index,data,mineral_i,grain_i);
                  
                  // Do the volume fraction of the grain
                  double vf_old = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                  double vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                  Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set."));
                  for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
                    {
                      Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set. grain_i = "
                                                              + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                                                              + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

                      if ((this ->get_time() != 0 ) && (area_sum > 0.))
                      {
                        //if(ratio != 0.0)
                        {
                        if(get_energy_ratio(cpo_index,data,mineral_i,grain_i) >= 1.0)
                        {
                          if(std::abs(dt * (( numbers::PI * std::pow(vf_new* 0.5,2.))/area_sum) * derivatives.first[grain_i]) <=get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i))  
                           {
                            vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) + dt * (( numbers::PI * std::pow(vf_new* 0.5,2.))/area_sum) * derivatives.first[grain_i];
                            set_grain_size_change(cpo_index,data,mineral_i,grain_i, dt * (( numbers::PI * std::pow(vf_new* 0.5,2.))/area_sum) * derivatives.first[grain_i]);
                           }
                          else
                          if((dt * (( numbers::PI * std::pow(vf_new* 0.5,2.))/area_sum) * derivatives.first[grain_i]) < 0)
                          {
                            vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) - get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i);
                            set_grain_size_change(cpo_index,data,mineral_i,grain_i, -1.0 * get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i));
                            
                          }
                          else 
                          if((dt * (( numbers::PI * std::pow(vf_new* 0.5,2.))/area_sum) * derivatives.first[grain_i]) > 0)
                          {
                            vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) + get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i);
                            set_grain_size_change(cpo_index,data,mineral_i,grain_i, 1.0 * get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i));
                            
                          }
                        }
                        else
                        if(get_energy_ratio(cpo_index,data,mineral_i,grain_i) < 1.0)
                        {
                          const double area_fraction = (( numbers::PI * std::pow(vf_new* 0.5,2.))/area_sum);
                          const double increment_term = (dt * derivatives.first[grain_i]);
                          const double initial_term = std::pow(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i),3.2);
                          vf_new = std::pow(initial_term + increment_term,1.0/3.2);
                          set_grain_size_change(cpo_index,data,mineral_i,grain_i, vf_new - get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i));
                          
                          break;
                        }
                      }
                      }
                      else
                      {
                        vf_new = vf_new;
                        set_grain_size_change(cpo_index,data,mineral_i,grain_i, 0.0);
                      }
                        
                      
                      if(vf_new <= 0.5 * std::pow(10,-6))
                      {
                        vf_new = 0.0;
                        set_grain_status(cpo_index,data,mineral_i,grain_i,-1);
                        break;
                      }
                      Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                             + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                             + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

                      Assert(vf_new >= 0,ExcMessage("volume_fractions[grain_i] is less than zero. grain_i = "
                                                    + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                                                    + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));
                      if (std::fabs(vf_new-vf_old) < property_advection_tolerance)
                        {
                          break;
                        }
                      vf_old = vf_new;

                    }
                      
                  set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i,vf_new);
                  sum_volume_fractions += vf_new;

                  // Do the rotation matrix for this grain
                  cosine_ref = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
                  Tensor<2,3> cosine_old = cosine_ref;
                  Tensor<2,3> cosine_new = cosine_ref;

                  for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
                    {
                      cosine_new = cosine_ref + dt * cosine_new * derivatives.second[grain_i];
                      if ((cosine_new-cosine_old).norm() < property_advection_tolerance)
                        {
                          break;
                        }
                      cosine_old = cosine_new;
                    }

                  set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,cosine_new);

                }
              
              double sum_grain_size = 0.0;
              int n_grains_considered = 0;
              for(unsigned int i_grain = 0; i_grain < n_grains; ++i_grain)
              {
                 if(get_volume_fractions_grains(cpo_index,data,mineral_i,i_grain)>0.)
                  {
                    sum_grain_size += (get_volume_fractions_grains(cpo_index,data,mineral_i,i_grain));
                    n_grains_considered += 1;
                  }
              }

              
              if(n_grains_considered > 0)
              {
                const double mean_grain_size = std::pow((sum_grain_size/n_grains_considered), 1.0);
                 set_mean_grain_size_mineral_mineral(cpo_index,data,mineral_i,mean_grain_size);
              } 
              else
              {
                 set_mean_grain_size_mineral_mineral(cpo_index,data,mineral_i,0.);
              }
              Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
              return sum_volume_fractions;
              break;
            }
          default:
              AssertThrow(false, ExcMessage("Internal error."));
              break;
        }        
      }
    


      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives(const unsigned int cpo_index,
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
                                                            const double water_content) const
      {
        std::pair<std::vector<double>, std::vector<Tensor<2,3>>> derivatives;
        switch (cpo_derivative_algorithm)
          {
            case CPODerivativeAlgorithm::spin_tensor:
            {
              return compute_derivatives_spin_tensor(velocity_gradient_tensor);
              break;
            }
            case CPODerivativeAlgorithm::drex_2004:
            {

              const DeformationType deformation_type = determine_deformation_type(deformation_type_selector[mineral_i],
                                                                                  position,
                                                                                  cell,
                                                                                  temperature,
                                                                                  pressure,
                                                                                  velocity,
                                                                                  compositions,
                                                                                  strain_rate,
                                                                                  deviatoric_strain_rate,
                                                                                  water_content);

              set_deformation_type(cpo_index,data,mineral_i,deformation_type);

              const std::array<double,4> ref_resolved_shear_stress = reference_resolved_shear_stress_from_deformation_type(deformation_type);

              return compute_derivatives_drex_2004(deformation_type,
                                                   cpo_index,
                                                   data,
                                                   mineral_i,
                                                   strain_rate_3d,
                                                   velocity_gradient_tensor,
                                                   ref_resolved_shear_stress);
              break;
            }
            case CPODerivativeAlgorithm::drexpp:
            {
              const DeformationType deformation_type = determine_deformation_type(deformation_type_selector[mineral_i],
                                                                                  position,
                                                                                  temperature,
                                                                                  pressure,
                                                                                  velocity,
                                                                                  compositions,
                                                                                  strain_rate,
                                                                                  deviatoric_strain_rate,
                                                                                  water_content);

              set_deformation_type(cpo_index,data,mineral_i,deformation_type);
              const std::array<double,4> ref_resolved_shear_stress = reference_resolved_shear_stress_from_deformation_type(deformation_type);
              
              double differential_stress;

                // now compute the normal viscosity to be able to computes the stress
              // Create the material model inputs and outputs to
              // retrieve the current viscosity.
              MaterialModel::MaterialModelInputs<dim> in = MaterialModel::MaterialModelInputs<dim>(1,compositions.size());
              in.pressure[0] = pressure;
              in.temperature[0] = temperature;
              in.position[0] = position;
              in.strain_rate[0] = strain_rate;
              in.composition[0] = compositions;
              

              in.requested_properties = MaterialModel::MaterialProperties::viscosity;

              MaterialModel::MaterialModelOutputs<dim> out(1,
                                                           this->n_compositional_fields());
              
              this->get_material_model().evaluate(in, out);

              // Compressive stress is positive in geoscience applications.
              SymmetricTensor<2, dim> stress = pressure * unit_symmetric_tensor<dim>();

              // Add elastic stresses if existent.
              AssertThrow(this->get_parameters().enable_elasticity == false, ExcMessage("Elasticity not supported when computing the CPO stress"));

              const double eta = out.viscosities[0];

              stress += -2. * eta * deviatoric_strain_rate;


              // Compute the deviatoric stress tensor after elastic stresses were added.
              const SymmetricTensor<2, dim> deviatoric_stress = deviator(stress);
              const std::array< double, dim > eigenvalues = dealii::eigenvalues(deviatoric_stress);
              differential_stress = eigenvalues[0]-eigenvalues[dim-1];
              
             
              // Compute the second moment invariant of the deviatoric stress
              // in the same way as the second moment invariant of the deviatoric
              // strain rate is computed in the viscoplastic material model.
              // TODO check that this is valid for the compressible case.
              //const double stress_invariant = (std::sqrt(std::max(-second_invariant(deviatoric_stress), 0.)));

              

              // SV: For now Im declaring only the bulk_recrystalization grain_size using the piezometer;
              std::array<double, 2> A = {{std::pow(10,4.18),std::pow(10.0,3.8)}};
              std::array<double, 2> m = {{-1.33, -1.28}};
              double piezometer;
              
              for(unsigned int i =0; i < n_minerals; i++)
              {
                if(differential_stress != 0.)
                 {
                  piezometer= (A[mineral_i] * std::pow(differential_stress/1e6,m[mineral_i]))*std::pow(10,-6);
                 }
                else
                 piezometer = 1.0;
                
                set_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i,piezometer);
              }
              
              // 
              const double gravity_norm = this->get_gravity_model().gravity_vector(position).norm();
              const double reference_density = (this->get_adiabatic_conditions().is_initialized())
                                               ?
                                               this->get_adiabatic_conditions().density(position)
                                               :
                                               3300.;
              
              MaterialModel::MaterialUtilities::PhaseFunctionInputs<dim> phase_inputs(temperature,
                                                                                      pressure,
                                                                                      this->get_geometry_model().depth(position),
                                                                                      gravity_norm * reference_density,
                                                                                      numbers::invalid_unsigned_int);
              
              std::vector<double> phase_function_values(phase_function->n_phase_transitions(),0.0);

              //Compute the value of phase functions

              for(unsigned int j = 0 ; j < phase_function->n_phase_transitions(); ++j)
              {
                phase_inputs.phase_index = j;
                phase_function_values[j] = phase_function->compute_value(phase_inputs);
              }
              
              double diffusion_pre_strainrate;
              double diffusion_grain_size_exponent;
              double dislocation_factor;
              double dislocation_stress_exponent;

              for(unsigned int composition = 1; composition < compositions.size(); ++composition)
              {
                const MaterialModel::Rheology::DiffusionCreepParameters p_dif = rheology_diff->compute_creep_parameters(composition,
                                                                                                                        phase_function_values,
                                                                                                                        phase_function->n_phase_transitions_for_each_composition());
                
                diffusion_pre_strainrate = p_dif.prefactor * std::exp(-1. * (p_dif.activation_energy +
                                                                  pressure * p_dif.activation_volume)/
                                                                  (constants::gas_constant * temperature));
                diffusion_grain_size_exponent = p_dif.grain_size_exponent;
                
                const MaterialModel::Rheology::DislocationCreepParameters p_dis = rheology_disl->compute_creep_parameters(composition,
                                                                                                                        phase_function_values,
                                                                                                                        phase_function->n_phase_transitions_for_each_composition());
                
                dislocation_factor = p_dis.prefactor * std::exp(-1. * (p_dis.activation_energy +
                                                                  pressure * p_dis.activation_volume)/
                                                                  (constants::gas_constant * temperature));
                
                dislocation_stress_exponent = p_dis.stress_exponent;
              }
              
              const double k0 = 1.8 * std::pow(10,3);
              const double activation_energy_growth = 620 * std::pow(10,3);
              const double activation_volume_growth = 5 * std::pow(10,-6);

              const double activation_enthalpy_growth = (activation_energy_growth + (pressure * activation_volume_growth));
              const double exponent_term = -1.0 * activation_enthalpy_growth/(constants::gas_constant * temperature);
              const double growth_rate = k0 * exp(exponent_term);

              return compute_derivatives_drexpp(cpo_index,
                                                   data,
                                                   mineral_i,
                                                   strain_rate_3d,
                                                   velocity_gradient_tensor,
                                                   ref_resolved_shear_stress,
                                                   diffusion_pre_strainrate,
                                                   deviatoric_stress,
                                                   diffusion_grain_size_exponent,
                                                   dislocation_factor,
                                                   dislocation_stress_exponent,
                                                   deviatoric_strain_rate,
                                                   growth_rate,
                                                   temperature);
              break;
            }
            default:
              AssertThrow(false, ExcMessage("Internal error."));
              break;
          }
        AssertThrow(false, ExcMessage("Internal error."));
        return derivatives;
      }



      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives_spin_tensor(const Tensor<2,3> &velocity_gradient_tensor) const
      {
        // dA/dt = W * A, where W is the spin tensor and A is the rotation matrix
        // The spin tensor is defined as W = 0.5 * ( L - L^T ), where L is the velocity gradient tensor.
        const Tensor<2,3> spin_tensor = -0.5 *(velocity_gradient_tensor - dealii::transpose(velocity_gradient_tensor));

        return std::pair<std::vector<double>, std::vector<Tensor<2,3>>>(std::vector<double>(n_grains,0.0), std::vector<Tensor<2,3>>(n_grains, spin_tensor));
      }

      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives_drex_2004(const DeformationType deformation_type,
                                                                      const unsigned int cpo_index,
                                                                      const ArrayView<double> &data,
                                                                      const unsigned int mineral_i,
                                                                      const SymmetricTensor<2,3> &strain_rate_3d,
                                                                      const Tensor<2,3> &velocity_gradient_tensor,
                                                                      const std::array<double,4> ref_resolved_shear_stress,
                                                                      const bool prevent_nondimensionalization) const
      {
        // This if statement is only there for the unit test. In normal situations it should always be set to false,
        // because the nondimensionalization should always be done (in this exact way), unless you really know what
        // you are doing.
        double nondimensionalization_value = 1.0;
        if (!prevent_nondimensionalization)
          {
            const std::array< double, 3 > eigenvalues = dealii::eigenvalues(strain_rate_3d);
            nondimensionalization_value = std::max(std::abs(eigenvalues[0]),std::abs(eigenvalues[2]));

            Assert(!std::isnan(nondimensionalization_value), ExcMessage("The second invariant of the strain rate is not a number."));
          }


        // Make the strain-rate and velocity gradient tensor non-dimensional
        // by dividing it through the second invariant
        const Tensor<2,3> strain_rate_nondimensional = nondimensionalization_value != 0 ? strain_rate_3d/nondimensionalization_value : strain_rate_3d;
        const Tensor<2,3> velocity_gradient_tensor_nondimensional = nondimensionalization_value != 0 ? velocity_gradient_tensor/nondimensionalization_value : velocity_gradient_tensor;

        // Create output variables
        std::vector<double> deriv_volume_fractions(n_grains);
        std::vector<Tensor<2,3>> deriv_a_cosine_matrices(n_grains);

        // Create shortcuts
        const std::array<double, 4> &tau = ref_resolved_shear_stress;
        std::vector<double> strain_energy(n_grains);
        double mean_strain_energy = 0;

        // First initiate the slip_normal_reference (vector n) and the
        // slip_direction_reference (vector l) in Kaminski (2001)
        std::array<Tensor<1,3>,4> slip_normal_reference;
        std::array<Tensor<1,3>,4> slip_direction_reference;

        // for olivine A,B,C,D,E and types and enstatite
        if (deformation_type == DeformationType::olivine_a_fabric ||
            deformation_type == DeformationType::olivine_b_fabric ||
            deformation_type == DeformationType::olivine_c_fabric ||
            deformation_type == DeformationType::olivine_d_fabric ||
            deformation_type == DeformationType::olivine_e_fabric ||
            deformation_type == DeformationType::enstatite)
          {
            slip_normal_reference = {{Tensor<1,3>({0,1,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,1,0}),Tensor<1,3>({1,0,0})}};
            slip_direction_reference = {{Tensor<1,3>({1,0,0}),Tensor<1,3>({1,0,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,0,1})}};
          }
        // Alternative Olivine D type fabric with {0kl} slip plane
        else if (deformation_type == DeformationType::olivine_d_0kl)
          {
            // Olivine axes length in angstrom (Å) https://www.mindat.org/min-29264.html
            const Tensor<1,3> olivine_a_axis ({4.816,0.,0.});
            const Tensor<1,3> olivine_b_axis ({0.,10.469,0.});
            const Tensor<1,3> olivine_c_axis ({0.,0.,6.099});

            // Crystal axis b minus axis c
            const Tensor<1,3> vec_13b_minus_c = 0.333*olivine_b_axis - olivine_c_axis;
            // Cross product between a*(b-c) to get normal vector for nsp{031}
            // Karato 2008 AnnRevEPS: D-type fabric slip system of {0kl}[100]
            // Bystricky_etal 2001 science {031}[100]
            Tensor<1,3> plane031_normal = cross_product_3d(olivine_a_axis,vec_13b_minus_c);
            plane031_normal /= plane031_normal.norm();
            // slip_normal_reference (vector n) & slip_direction_reference (vector l) for olivine D types
            slip_normal_reference = {{Tensor<1,3>({0,1,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,1,0}),plane031_normal}};
            slip_direction_reference = {{Tensor<1,3>({1,0,0}),Tensor<1,3>({1,0,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({1,0,0})}};
          }
        // CPX has different crystal structure (monoclinic) and slip systems than olivine or enstatite fabrics
        else if (deformation_type == DeformationType::clinopyroxene)
          {
            // More accurate way to calculate slip plane <110> normal by doing cross product
            // First CPX crystal structure info, length unit Angstrom (Å)
            // See Clinopyroxene on mindat.org at https://www.mindat.org/min-7630.html
            const Tensor<1,3> vec_a_axis ({9.658, 0., 0.});
            const Tensor<1,3> vec_b_axis ({0.,8.795, 0.});
            const double length_c = 5.294; //[Angstrom]
            const double angle_beta = 107.42 * constants::degree_to_radians;
            const Tensor<1,3> vec_c_axis ({length_c * std::cos(angle_beta), 0, length_c * std::sin(angle_beta)});

            // Tensor 1 is vector b axis minus the vector a axis
            // Calculate plane110_normal --> crystal axis b minus axis a
            const Tensor<1,3> vec_b_minus_a = vec_b_axis - vec_a_axis;
            // Cross product between c*(b-a) to get normal vector for slip plane nsp{110}
            Tensor<1,3> plane110_normal = cross_product_3d(vec_b_minus_a,vec_c_axis);
            plane110_normal /= plane110_normal.norm();

            // Calculate plane11_0_normal --> crystal axis a minus axis -b
            const Tensor<1,3> vec_a_minus_neg_b = vec_a_axis - (-vec_b_axis);
            // Cross product between (a-(-b))*c to get normal vector for slip plane nsp{110}
            Tensor<1,3> plane11_0_normal = cross_product_3d(vec_a_minus_neg_b,vec_c_axis);
            plane11_0_normal /= plane11_0_normal.norm();

            // Calculate slip direction vector 110 a + b
            Tensor<1,3> sd110 = vec_a_axis + vec_b_axis;
            sd110 /= sd110.norm();
            // Both Bascou etal., 2002 JSG and Zhang et al., 2006 EPSL agree on the three dominant slip systems:
            // {11_0}1/2[110],  {110}[001],  and  {100}[001],
            slip_normal_reference =  {{Tensor<1,3>({0,1,0}),plane11_0_normal,plane110_normal,Tensor<1,3>({1,0,0})}};
            slip_direction_reference = {{Tensor<1,3>({0,0,1}),0.5*sd110,Tensor<1,3>({0,0,1}),Tensor<1,3>({0,0,1})}};
          }
        else
          AssertThrow(false, ExcMessage("Unknown deformation type"));

        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Compute the Schmidt tensor for this grain (nu), s is the slip system.
            // We first compute beta_s,nu (equation 5, Kaminski & Ribe, 2001)
            // Then we use the beta to calculate the Schmidt tensor G_{ij} (Eq. 5, Kaminski & Ribe, 2001)
            Tensor<2,3> G;
            Tensor<1,3> w;
            Tensor<1,4> beta({1.0, 1.0, 1.0, 1.0});

            // These are variables we only need for olivine, but we need them for both
            // Within this if block and the next ones
            // Ordered vector where the first entry is the max/weakest and the last entry is the inactive slip system.
            std::array<unsigned int,4> indices {};

            // compute G and beta
            Tensor<1,4> bigI;
            const Tensor<2,3> rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            const Tensor<2,3> rotation_matrix_transposed = transpose(rotation_matrix);
            for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
              {
                const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
                const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
                const Tensor<2,3> slip_outer_product = outer_product(slip_direction_global,slip_normal_global);
                bigI[slip_system_i] = scalar_product(slip_outer_product,strain_rate_nondimensional);
              }

            if (bigI.norm() < 1e-10)
              {
                // In this case there is no shear, only (possibly) a rotation. So \gamma_y and/or G should be zero.
                // Which is the default value, so do nothing.
              }
            else
              {
                // compute the element wise absolute value of the element wise
                // division of bigI by tau (tau = ref_resolved_shear_stress).
                std::array<double,4> q_abs;
                for (unsigned int i = 0; i < 4; ++i)
                  {
                    q_abs[i] = std::abs(bigI[i] / tau[i]);
                  }

                // here we find the indices starting at the largest value and ending at the smallest value
                // and assign them to special variables. Because all the variables are absolute values,
                // we can set them to a negative value to ignore them. This should be faster then deleting
                // the element, which would require allocation. (not tested)
                for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                  {
                    indices[slip_system_i] = std::distance(q_abs.begin(),std::max_element(q_abs.begin(), q_abs.end()));
                    q_abs[indices[slip_system_i]] = -1;
                  }

                // compute the ordered beta vector, which is the relative slip rates of the active slip systems.
                // Test whether the max element is not equal to zero.
                Assert(bigI[indices[0]] != 0.0, ExcMessage("Internal error: bigI is zero."));
                beta[indices[0]] = 1.0; // max q_abs, weak system (most deformation) "s=1"

                const double ratio = tau[indices[0]]/bigI[indices[0]];
                for (unsigned int slip_system_i = 1; slip_system_i < 4-1; ++slip_system_i)
                  {
                    beta[indices[slip_system_i]] = std::pow(std::abs(ratio * (bigI[indices[slip_system_i]]/tau[indices[slip_system_i]])), stress_exponent);
                  }
                beta[indices.back()] = 0.0;

                // Now compute the crystal rate of deformation tensor. equation 4 of Kaminski&Ribe 2001
                // rotation_matrix_transposed = inverse of rotation matrix
                // (see Engler et al., 2024 book: Intro to Texture analysis chp 2.3.2 The Rotation Matrix)
                // this transform the crystal reference frame to specimen reference frame

                for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                  {
                    const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
                    const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
                    const Tensor<2,3> slip_outer_product = outer_product(slip_direction_global,slip_normal_global);
                    G += 2.0 * beta[slip_system_i] * slip_outer_product;
                  }
              }

            // Now calculate the analytic solution to the deformation minimization problem
            // compute gamma (equation 7, Kaminiski & Ribe, 2001)

            // Top is the numerator and bottom is the denominator in equation 7.
            double top = 0;
            double bottom = 0;
            for (unsigned int i = 0; i < 3; ++i)
              {
                // Following the actual Drex implementation we use i+2, which differs
                // from the EPSL paper, which says gamma_nu depends on i+1
                const unsigned int i_offset = (i==0) ? (i+2) : (i-1);

                top = top - (velocity_gradient_tensor_nondimensional[i][i_offset]-velocity_gradient_tensor_nondimensional[i_offset][i])*(G[i][i_offset]-G[i_offset][i]);
                bottom = bottom - (G[i][i_offset]-G[i_offset][i])*(G[i][i_offset]-G[i_offset][i]);

                for (unsigned int j = 0; j < 3; ++j)
                  {
                    top = top + 2.0 * G[i][j]*velocity_gradient_tensor_nondimensional[i][j];
                    bottom = bottom + 2.0* G[i][j] * G[i][j];
                  }
              }
            // see comment on if all bigI are zero. In that case gamma should be zero.
            const double gamma = (bottom != 0.0) ? top/bottom : 0.0;

            // compute w (equation 8, Kaminiski & Ribe, 2001)
            // w is the Rotation rate vector of the crystallographic axes of grain
            w[0] = 0.5*(velocity_gradient_tensor_nondimensional[2][1]-velocity_gradient_tensor_nondimensional[1][2]) - 0.5*(G[2][1]-G[1][2])*gamma;
            w[1] = 0.5*(velocity_gradient_tensor_nondimensional[0][2]-velocity_gradient_tensor_nondimensional[2][0]) - 0.5*(G[0][2]-G[2][0])*gamma;
            w[2] = 0.5*(velocity_gradient_tensor_nondimensional[1][0]-velocity_gradient_tensor_nondimensional[0][1]) - 0.5*(G[1][0]-G[0][1])*gamma;

            // Compute strain energy for this grain (abbreviated Estr)
            // For olivine: DREX only sums over 1-3. But Christopher Thissen's matlab
            // code (https://github.com/cthissen/Drex-MATLAB) corrected
            // this and writes each term using the indices created when calculating bigI.
            // Note tau = RRSS = (tau_m^s/tau_o), this why we get tau^(p-n)
            for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
              {
                const double rhos = std::pow(tau[indices[slip_system_i]],exponent_p-stress_exponent) *
                                    std::pow(std::abs(gamma*beta[indices[slip_system_i]]),exponent_p/stress_exponent);
                strain_energy[grain_i] += rhos * std::exp(-nucleation_efficiency * rhos * rhos);

                Assert(isfinite(strain_energy[grain_i]), ExcMessage("strain_energy[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(strain_energy[grain_i])
                                                                    + ", rhos (" + std::to_string(slip_system_i) + ") = " + std::to_string(rhos)
                                                                    + ", nucleation_efficiency = " + std::to_string(nucleation_efficiency) + "."));
              }

            // compute the derivative of the rotation matrix: \frac{\partial a_{ij}}{\partial t}
            // (Eq. 9, Kaminski & Ribe 2001)
            deriv_a_cosine_matrices[grain_i] = 0;
            const double volume_fraction_grain = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            if (volume_fraction_grain >= threshold_GBS/n_grains)
              {
                deriv_a_cosine_matrices[grain_i] = Utilities::Tensors::levi_civita<3>() * w * nondimensionalization_value;

                // volume averaged strain energy
                mean_strain_energy += volume_fraction_grain * strain_energy[grain_i];

                Assert(isfinite(mean_strain_energy), ExcMessage("mean_strain_energy when adding grain " + std::to_string(grain_i) + " is not finite: " + std::to_string(mean_strain_energy)
                                                                + ", volume_fraction_grain = " + std::to_string(volume_fraction_grain) + "."));
              }
            else
              {
                strain_energy[grain_i] = 0;
              }
          }

        // Change of volume fraction of grains by grain boundary migration
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Different than D-Rex. Here we actually only compute the derivative and do not multiply it with the volume_fractions. We do that when we advect.
            deriv_volume_fractions[grain_i] = get_volume_fraction_mineral(cpo_index,data,mineral_i) * mobility * (mean_strain_energy - strain_energy[grain_i]) * nondimensionalization_value;

            Assert(isfinite(deriv_volume_fractions[grain_i]),
                   ExcMessage("deriv_volume_fractions[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(deriv_volume_fractions[grain_i])));
          }

        return std::pair<std::vector<double>, std::vector<Tensor<2,3>>>(deriv_volume_fractions, deriv_a_cosine_matrices);
      }
      
      template <int dim>
      void 
      CrystalPreferredOrientation<dim>::recrystalize_grains(const unsigned int cpo_index,
                                         const ArrayView<double> &data,
                                         const unsigned int mineral_i,
                                         const std::vector<double> &recrystalized_fraction,
                                         const std::vector<Tensor<1,3>> &sgr_rotation_axis,
                                         std::vector<bool> &rx_now) const
      {
        
        std::vector<std::size_t> permutation_vector;
        std::vector<std::size_t> empty_buffer_vector;
        std::vector<std::size_t> buffer_vector;
        int buffer_vector_counter = n_grains_buffer;

        // Creating a vector of indices to track which slots are empty
        for (unsigned int i = 0;  i < n_grains ; ++i)
          {
            if (get_grain_status(cpo_index,data,mineral_i,i) == -1)
              {
                permutation_vector.push_back(i);
              }

            if (get_grain_status(cpo_index,data,mineral_i,i) == -2)
              {
                empty_buffer_vector.push_back(i);
              }
            if(i >= n_grains - n_grains_buffer)
              { 
                buffer_vector.push_back(i);
              }
          }

        if(empty_buffer_vector.size() == 0)
          {
            std::sort(buffer_vector.begin(), buffer_vector.end(),
                  [&](std::size_t grain_i, std::size_t grain_j)
              {
                return get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) < get_volume_fractions_grains(cpo_index,data,mineral_i,grain_j);
              });

            buffer_vector_counter = 0;
          }

        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            Tensor<2,3> parent_orientation = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);

            const double grain_size = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
           
            const double volume =  numbers::PI * std::pow(grain_size * 0.5 , 2.0);
            const double rx_volume =  numbers::PI * std::pow(get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i) * 0.5 , 2.0);

            Tensor<2,3> rotation_matrix ;
            int n_recrystalized_grains; 
            double replaced_grain_volume;

            if((volume >= 2. * rx_volume)&&(get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i) > 0.))
              {
                n_recrystalized_grains = (std::floor(recrystalized_fraction[grain_i] * (volume/rx_volume)));
               
              }
            else
                n_recrystalized_grains = 0;
            
            
            double left_overs = volume - (n_recrystalized_grains * rx_volume);
            
            double left_over_grain_size = 2.0 * std::pow((left_overs *  (1.0/numbers::PI)),(1.0/2.0));
            
             if(n_recrystalized_grains >= 1.)
              {
                if(left_over_grain_size < get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i))
                  {
                    n_recrystalized_grains += -1;
                    left_overs = volume - (n_recrystalized_grains * rx_volume);
                    left_over_grain_size = 2.0 * std::pow((left_overs * (1.0/numbers::PI)),(1.0/2.0));
                  }
                                
                double unrx_portion;
                if(volume != 0.)
                   unrx_portion = (volume -(n_recrystalized_grains * rx_volume)/volume)*recrystalized_fraction[grain_i];
                
                set_rx_fractions(cpo_index,data,mineral_i,grain_i,unrx_portion);
                set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i,left_over_grain_size);

                const double deg2rad = 0.01745329;
                double random_angle;
                if(max_dispersion != 0.0)
                {
                  boost::random::uniform_real_distribution<double> uniform_distribution1(-1.0 * max_dispersion * deg2rad,max_dispersion * deg2rad);
                  random_angle = uniform_distribution1(this->random_number_generator);
                }
                else
                {
                  random_angle = 0.0;
                }
                rotation_matrix = Utilities::rotation_matrix_from_axis(sgr_rotation_axis[grain_i],random_angle);
                
                if(permutation_vector.size() >= n_recrystalized_grains)
                  {
                    for (unsigned int recrystalize_grain_i = 0; recrystalize_grain_i < n_recrystalized_grains  ; ++recrystalize_grain_i)
                      {
                        int random_var = std::rand() % permutation_vector.size();
                        set_volume_fractions_grains(cpo_index,data,mineral_i,permutation_vector[random_var],get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i));
                        set_rotation_matrix_grains(cpo_index,data,mineral_i,permutation_vector[random_var],rotation_matrix * parent_orientation * transpose(rotation_matrix));            
                        set_grain_status(cpo_index,data,mineral_i,permutation_vector[random_var],1);
                        set_strain_rate_ratio(cpo_index,data,mineral_i,permutation_vector[random_var],0.0);
                        rx_now[permutation_vector[random_var]] = true;
                        permutation_vector.erase(permutation_vector.begin() + random_var);            
                      }
                  }
                else
                  {
                    if( empty_buffer_vector.size() >= buffer_vector_counter+ n_recrystalized_grains)
                     {
                       for (unsigned int recrystalize_grain_i = 0; recrystalize_grain_i < n_recrystalized_grains  ; ++recrystalize_grain_i)
                         {
                          int random_var = std::rand() % empty_buffer_vector.size();
                          set_volume_fractions_grains(cpo_index,data,mineral_i,empty_buffer_vector[random_var],get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i));
                          set_rotation_matrix_grains(cpo_index,data,mineral_i,empty_buffer_vector[random_var],rotation_matrix * parent_orientation * transpose(rotation_matrix));            
                          set_grain_status(cpo_index,data,mineral_i,empty_buffer_vector[random_var],2);
                          set_strain_rate_ratio(cpo_index,data,mineral_i,empty_buffer_vector[random_var],0.0);
                          rx_now[empty_buffer_vector[random_var]] = true;
                          empty_buffer_vector.erase(empty_buffer_vector.begin() + random_var);            
                      }
                     } 
                    else
                     {
                      if(buffer_vector_counter+n_recrystalized_grains >= n_grains_buffer)
                       {
                          std::sort(buffer_vector.begin(), buffer_vector.end(),
                             [&](std::size_t grain_i, std::size_t grain_j)
                            {
                              return get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) < get_volume_fractions_grains(cpo_index,data,mineral_i,grain_j);
                            });

                          buffer_vector_counter = 0;
                       }
                       
                       double replaced_grain_size;
                       double replaced_volume;

                       for (unsigned int recrystalize_grain_i = 0; recrystalize_grain_i <= n_recrystalized_grains; ++recrystalize_grain_i)
                          {
                             replaced_grain_volume += numbers::PI * std::pow(0.5 * get_volume_fractions_grains(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter+recrystalize_grain_i]),2.0);
                             set_volume_fractions_grains(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter+recrystalize_grain_i],0.);
                          }
                       
                      if(replaced_grain_volume > 2 * rx_volume)
                         {
                            replaced_grain_volume = 2 * rx_volume;
                         }     
                    
                      if(replaced_grain_volume > 0.0)
                        { 
                          set_volume_fractions_grains(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],2.0 *std::pow((replaced_grain_volume/numbers::PI),1./2.));
                          set_rotation_matrix_grains(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],rotation_matrix * parent_orientation * transpose(rotation_matrix));            
                          set_grain_status(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],3);
                          set_strain_accumulated(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.0);
                          set_strain_energy(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.);
                          set_rx_fractions(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.);
                          set_strain_rate_ratio(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.0);
                          rx_now[buffer_vector[buffer_vector_counter]] = true;
                          buffer_vector_counter++;
                        }         
                      for (unsigned int recrystalize_grain_i = 0; recrystalize_grain_i < n_recrystalized_grains  ; ++recrystalize_grain_i)
                        {
                          set_volume_fractions_grains(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],get_bulk_recrystalization_grain_size_mineral(cpo_index,data,mineral_i));
                          set_rotation_matrix_grains(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],rotation_matrix * parent_orientation * transpose(rotation_matrix));            
                          set_grain_status(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],4);
                          set_strain_accumulated(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.0);
                          set_strain_energy(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.);
                          set_rx_fractions(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.);
                          set_strain_rate_ratio(cpo_index,data,mineral_i,buffer_vector[buffer_vector_counter],0.0);
                          rx_now[buffer_vector[buffer_vector_counter]] = true;
                          buffer_vector_counter++;
                        }
                     }
                  } 
               }
          
      }
      }

      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives_drexpp(const unsigned int cpo_index,
                                                                      const ArrayView<double> &data,
                                                                      const unsigned int mineral_i,
                                                                      const SymmetricTensor<2,3> &strain_rate,
                                                                      const Tensor<2,3> &velocity_gradient_tensor,
                                                                      const std::array<double,4> ref_resolved_shear_stress,
                                                                      const double diffusion_pre_strainrate,
                                                                      const SymmetricTensor<2,dim> &deviatoric_stress,
                                                                      const double diffusion_grain_size_exponent,
                                                                      const double dislocation_factor,
                                                                      const double dislocation_stress_exponent,
                                                                      const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                                                                      const double growth_rate,
                                                                      const double temperature) const
      {
        // time variables - timestep/time/time increment

        const double t =this-> get_time();
        const double timestep =this-> get_timestep();
                 
        // create output variables
        std::vector<double> deriv_volume_fractions(n_grains);
        std::vector<Tensor<2,3>> deriv_a_cosine_matrices(n_grains);

        // Other variables that need to be stored globally in order to run D-Rex^{++}
        std::vector<int> grain_status(n_grains);
        std::vector<double> recrystalized_fractions(n_grains);
        std::vector<bool> rx_now(n_grains);

        std::vector<Tensor<1,3>> sgr_rotation_axis(n_grains);
        
        // create shorcuts
        const std::array<double, 4> &tau = ref_resolved_shear_stress;
        std::vector<SymmetricTensor<2,3>> dislocation_strain_rate_3d(n_grains);
        std::vector<Tensor<2,3>> dislocation_velocity_gradient_3d(n_grains);

        // Calculating Dislocation creep strain rate for all grains
        const std::array< double, dim > eigenvalues = dealii::eigenvalues(deviatoric_stress);
        const double differential_stress = eigenvalues[0]-eigenvalues[dim-1];
        const double dislocation_creep = dislocation_factor * std::pow(differential_stress,dislocation_stress_exponent);

        // Calculating Dislocation creep strain rate for all grains
        const std::array< double, dim > eigenvalues = dealii::eigenvalues(deviatoric_stress);
        const double differential_stress = eigenvalues[0]-eigenvalues[dim-1];
        const double dislocation_creep = dislocation_factor * std::pow(differential_stress,dislocation_stress_exponent);
       
        for(unsigned int i_grain = 0; i_grain < n_grains; ++i_grain)
        {
          if(get_volume_fractions_grains(cpo_index,data,mineral_i,i_grain) > 0.0)
          {
          // Calculating the strain rate tensor corresponding to diffusion creep
          const double grain_size = get_volume_fractions_grains(cpo_index,data,mineral_i,i_grain);
          double diffusion_strain_rate;
          diffusion_strain_rate = diffusion_pre_strainrate * differential_stress * std::pow(grain_size, -1.0 * diffusion_grain_size_exponent);
          double strr_ratio;

          if(t ==0)
          {
            strr_ratio= 0.;
          }
          else
          {
            strr_ratio = diffusion_strain_rate/(diffusion_strain_rate + dislocation_creep);
          }
          
          const SymmetricTensor<2,dim> diffusion_creep_strain_rate = strr_ratio * deviatoric_strain_rate;
          const SymmetricTensor<2,dim> dislocation_strain_rate = deviatoric_strain_rate - diffusion_creep_strain_rate;
          
          // even in 2d we need 3d strain-rates and velocity gradient tensors. So we make them 3d by
          // adding an extra dimension which is zero.
          dislocation_strain_rate_3d[i_grain] = strain_rate;
          /*
            dislocation_strain_rate_3d[i_grain][0][0] = dislocation_strain_rate[0][0];
            dislocation_strain_rate_3d[i_grain][0][1] = dislocation_strain_rate[0][1];
              //sym: strain_rate_3d[1][0] = strain_rate[1][0];
            dislocation_strain_rate_3d[i_grain][1][1] = dislocation_strain_rate[1][1];

            if (dim == 3)
              {
                dislocation_strain_rate_3d[i_grain][0][2] = dislocation_strain_rate[0][2];
                dislocation_strain_rate_3d[i_grain][1][2] = dislocation_strain_rate[1][2];
                //sym: strain_rate_3d[2][0] = strain_rate[0][2];
                //sym: strain_rate_3d[2][1] = strain_rate[1][2];
                dislocation_strain_rate_3d[i_grain][2][2] = dislocation_strain_rate[2][2];
              }
          
          */
          
         const Tensor<2,dim> diffusion_velocity_gradient_tensor = diffusion_creep_strain_rate;
          
          Tensor<2,3> diffusion_velocity_gradient_3d;
          diffusion_velocity_gradient_3d[0][0] = diffusion_velocity_gradient_tensor[0][0];
          diffusion_velocity_gradient_3d[0][1] = diffusion_velocity_gradient_tensor[0][1];
          diffusion_velocity_gradient_3d[1][0] = diffusion_velocity_gradient_tensor[1][0];
          diffusion_velocity_gradient_3d[1][1] = diffusion_velocity_gradient_tensor[1][1];
            if (dim == 3)
              {
                diffusion_velocity_gradient_3d[0][2] = diffusion_velocity_gradient_tensor[0][2];
                diffusion_velocity_gradient_3d[1][2] = diffusion_velocity_gradient_tensor[1][2];
                diffusion_velocity_gradient_3d[2][0] = diffusion_velocity_gradient_tensor[2][0];
                diffusion_velocity_gradient_3d[2][1] = diffusion_velocity_gradient_tensor[2][1];
                diffusion_velocity_gradient_3d[2][2] = diffusion_velocity_gradient_tensor[2][2];
              }
            
          dislocation_velocity_gradient_3d[i_grain] = velocity_gradient_tensor ;//- diffusion_velocity_gradient_3d;
          if(t != 0)
          {
            set_strain_rate_ratio(cpo_index,data,mineral_i,i_grain,std::sqrt(std::max(-second_invariant(dislocation_strain_rate_3d[i_grain]), 0.)/ std::max(-second_invariant(strain_rate), 0.)));
            
          }
          else 
            set_strain_rate_ratio(cpo_index,data,mineral_i,i_grain,0);
        }
      }

        // Variables that I will store for the purpose of benchmarking model behavior and will have to make them local variables before trying to merge with ASPECT
        std::vector<double> volume_derivative(n_grains);
        std::vector<double> dislocation_density(n_grains);
        std::vector<double> strain_energy(n_grains);
        std::vector<double> surface_energy(n_grains);
        std::vector<double> strain_accumulated(n_grains);
        std::vector<double> strain_increment(n_grains);
        std::vector<double> rho_scale(n_grains);
        std::vector<double> piezometer(n_grains);
        
        
        // create local variables
        std::vector<Tensor<1,3>> spin_vectors(n_grains);
        
        // Constants -> the values below are for olivine alone (SV: Do I add the citations?)
        const double shear_modulus = 8.0 * std::pow(10.0,10.0);
        const double burgers_vector = 5.0 * std::pow(10.0,-10.0);
        
        // first compute the amount of slip, G, strain accumulated and dislocation density for n_grains, as long as grain is initialized, i.e grain size is not equal to 0
        for(unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
        {
          const Tensor<2,3> rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
          
          const Tensor<2,3> rotation_matrix_transposed = transpose(get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i));
           
           // Compute the Schmidt tensor for this grain (nu) and the resolved shear strain rate along slip system s, s is the slip system
           std::array<Tensor<1,3>,4> slip_normal_reference {{Tensor<1,3>({0,1,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,1,0}),Tensor<1,3>({1,0,0})}};
           std::array<Tensor<1,3>,4> slip_direction_reference {{Tensor<1,3>({1,0,0}),Tensor<1,3>({1,0,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,0,1})}};
            std::array<Tensor<1,3>,4> rx_reference {{Tensor<1,3>({0,0,1}),Tensor<1,3>({0,1,0}),Tensor<1,3>({1,0,0}),Tensor<1,3>({0,1,0})}}; 
           Tensor<1,4> bigI;
           for(unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
           {
              const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
              const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
              const Tensor<2,3> slip_cross_product =(outer_product(slip_direction_global,slip_normal_global));
              
              bigI[slip_system_i] = scalar_product(slip_cross_product,dislocation_strain_rate_3d[grain_i]);
           }
           
           if(bigI.norm() < 1e-30)
           {
            // The resolved shear strain rates are too small to induce significant slip. Therefore the magnitudes of \gamma and \omega will be negligible so we ignore the effect of rotaiton in this case
            // Use this condition to set the dislocation density to zero so the driving energy for GBM will only be surface energy so we can somewhat simulate static recrystalization
           }
           else
           {
             Tensor<1,4> beta({1.0, 1.0, 1.0, 1.0});
             // Ordered vector where the first entry is the max/weakest and the last entry is the inactive slip system.
             std::array<unsigned int,4> indices {};
             
             std::array<double,4> q_abs;

             for (unsigned int i = 0; i < 4; ++i)
                {
                  q_abs[i] = std::abs(bigI[i] / tau[i]);
                }
             
             // here we find the indices starting at the largest value and ending at the smallest value
             // and assign them to special variables. Because all the variables are absolute values,
             // we can set them to a negative value to ignore them. This should be faster then deleting
             // the element, which would require allocation. (not tested)
             for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                {
                  indices[slip_system_i] = std::distance(q_abs.begin(),std::max_element(q_abs.begin(), q_abs.end()));
                  q_abs[indices[slip_system_i]] = -1;
                }

             // compute the ordered beta vector, which is the relative slip rates of the active slip systems.
             // Test whether the max element is not equal to zero.
             Assert(bigI[indices[0]] != 0.0, ExcMessage("Internal error: bigI is zero."));
             beta[indices[0]] = 1.0; // max q_abs, weak system (most deformation) "s=1"
             const double ratio = tau[indices[0]]/bigI[indices[0]];
             for (unsigned int slip_system_i = 1; slip_system_i < 4-1; ++slip_system_i)
               {
                  beta[indices[slip_system_i]] = std::pow(std::abs(ratio * (bigI[indices[slip_system_i]]/tau[indices[slip_system_i]])), drexpp_stress_exponent[mineral_i]);
               }
             beta[indices.back()] = 0.0;

              sgr_rotation_axis[grain_i] = transpose(get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i)) *  rx_reference[indices[0]];
             
              Tensor<2,3> schmidt_tensor; // comment- This is not the schmidt tensor. The symmetric part of the slip cross product calculated is the actual schmidt tensor.
             
             for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
               {
                 const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
                 const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
                 const Tensor<2,3> slip_cross_product = outer_product(slip_direction_global,slip_normal_global);
                 schmidt_tensor += 2.0 * beta[slip_system_i] * slip_cross_product;
               }
             
             // Now calculate the analytic solution to the deformation minimization problem
             // compute gamma (equation 7, Kaminiski & Ribe, 2001)

             // Top is the numerator and bottom is the denominator in equation 7.
             double top = 0;
             double bottom = 0;
             for (unsigned int i = 0; i < 3; ++i)
               {
                 // Following the actual Drex implementation we use i+2, which differs
                 // from the EPSL paper, which says gamma_nu depends on i+1
                 const unsigned int i_offset = (i==0) ? (i+2) : (i-1);

                 top = top - (dislocation_velocity_gradient_3d[grain_i][i][i_offset]-dislocation_velocity_gradient_3d[grain_i][i_offset][i])*(schmidt_tensor[i][i_offset]-schmidt_tensor[i_offset][i]);
                 bottom = bottom - (schmidt_tensor[i][i_offset]-schmidt_tensor[i_offset][i])*(schmidt_tensor[i][i_offset]-schmidt_tensor[i_offset][i]);

                 for (unsigned int j = 0; j < 3; ++j)
                  {
                    top = top + 2.0 * schmidt_tensor[i][j]*dislocation_velocity_gradient_3d[grain_i][i][j];
                    bottom = bottom + 2.0* schmidt_tensor[i][j] * schmidt_tensor[i][j];
                  }
               }
             
             // see comment on if all BigI are zero. In that case gamma should be zero.
             const double gamma = (bottom != 0.0) ? top/bottom : 0.0;
             
             // compute w (equation 8, Kaminiski & Ribe, 2001)
             // w is the Rotation rate vector of the crystallographic axes of grain

             spin_vectors[grain_i] = Tensor<1,3>
                                        (
             {
               0.5*(dislocation_velocity_gradient_3d[grain_i][2][1]-dislocation_velocity_gradient_3d[grain_i][1][2]) - 0.5*(schmidt_tensor[2][1]-schmidt_tensor[1][2]) *gamma,
               0.5*(dislocation_velocity_gradient_3d[grain_i][0][2]-dislocation_velocity_gradient_3d[grain_i][2][0]) - 0.5*(schmidt_tensor[0][2]-schmidt_tensor[2][0]) *gamma,
               0.5*(dislocation_velocity_gradient_3d[grain_i][1][0]-dislocation_velocity_gradient_3d[grain_i][0][1]) - 0.5*(schmidt_tensor[1][0]-schmidt_tensor[0][1]) *gamma
             });
             
             // Calculate the amount of strain accumulated by dislocation glide.
             Tensor<2,3> local_strain_rate = (schmidt_tensor * gamma) - (Utilities::Tensors::levi_civita<3>()*spin_vectors[grain_i]);
             SymmetricTensor<2,3>d = symmetrize (local_strain_rate);

             strain_increment[grain_i] = this->get_timestep() * std::sqrt(std::max(-second_invariant(d), 0.));
            
             set_strain_rate(cpo_index,data,mineral_i,grain_i,std::sqrt(std::max(-second_invariant(d), 0.)));
                          
             if(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.)
             {
               strain_accumulated[grain_i] =  get_strain_accumulated(cpo_index,data,mineral_i,grain_i) + strain_increment[grain_i];             
             }

             set_strain_accumulated(cpo_index,data,mineral_i,grain_i,strain_accumulated[grain_i]);

             // Compute the dislocation density for this grain
             // For olivine: DREX only sums over 1-3. But Christopher Thissen's matlab 
             // code (https://github.com/cthissen/Drex-MATLAB) corrected
             // this and writes each term using the indices created when calculating bigI.
             // Note tau = RRSS = (tau_m^s/tau_o), this why we get tau^(p-n)
            if (get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.)
              {
                const double non_dimensionalization = std::sqrt(std::max(-second_invariant(d), 0.));                
                double rho_scale;
                const double ref_stress = std::pow(non_dimensionalization/dislocation_factor,1.0/dislocation_stress_exponent);
               
                rho_scale = std::pow(ref_stress /(0.5 * shear_modulus * burgers_vector),drexpp_exponent_p[mineral_i]);
                             
  
                
                set_differential_stress(cpo_index,data,mineral_i,grain_i,ref_stress);

                for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                  {
                    const double rhos = rho_scale * std::pow(tau[indices[slip_system_i]], stress_exponent - drexpp_exponent_p[mineral_i]) *
                                                    std::pow(std::abs(beta[indices[slip_system_i]] * gamma/non_dimensionalization),drexpp_exponent_p[mineral_i]/stress_exponent);

                    dislocation_density[grain_i] += rhos;
                  }
                
                if(get_strain_accumulated(cpo_index,data,mineral_i,grain_i) >= 3.0)
                  dislocation_density[grain_i] = dislocation_density[grain_i];
                else 
                  dislocation_density[grain_i] = (get_strain_accumulated(cpo_index,data,mineral_i,grain_i)/3.0) * dislocation_density[grain_i];
                
                
                strain_energy[grain_i] = shear_modulus * burgers_vector * burgers_vector * dislocation_density[grain_i];

                set_dislocation_density(cpo_index,data,mineral_i,grain_i,dislocation_density[grain_i]);
                set_strain_energy(cpo_index,data,mineral_i,grain_i,strain_energy[grain_i]);
              }
           }
        }
        
        // Calculating rx kinetics
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            if ((get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.) && (get_strain_accumulated(cpo_index,data,mineral_i,grain_i) >= 0.25))
              {
                recrystalized_fractions[grain_i] = get_rx_fractions(cpo_index,data,mineral_i,grain_i);
                if(strain_accumulated[grain_i] - 0.25 < strain_increment[grain_i])
                {
                  recrystalized_fractions[grain_i] += (avrami_slope_input * (get_strain_accumulated(cpo_index,data,mineral_i,grain_i) - 0.25));  
                }
                else
                  recrystalized_fractions[grain_i] += (avrami_slope_input * strain_increment[grain_i]);
                
                if (recrystalized_fractions[grain_i] > 1.0)
                  recrystalized_fractions[grain_i] = 1.0;
              }
            else
              {
                recrystalized_fractions[grain_i] = 0.0;
              }
            set_rx_fractions(cpo_index,data,mineral_i,grain_i,recrystalized_fractions[grain_i]);
          }

        // Calling the rx module to carry out dynamic recrystalization        
        this->recrystalize_grains(cpo_index,
                                  data,
                                  mineral_i,
                                  recrystalized_fractions,
                                  sgr_rotation_axis,
                                  rx_now);
        
        
        // Calculating mean strain energy
        double mean_strain_energy = 0.0;
        double sum_volume = 0. ;

        for (unsigned int grain_i = 0; grain_i<n_grains; ++grain_i)
          {
            if (this-> get_time() != 0)
              {
                const double grain_size = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                
                if ((get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.) && (rx_now[grain_i] == false))
                {
                  const double volume = (4./3.) * numbers::PI * std::pow(0.5 * grain_size,3.);
                  sum_volume += volume;
                  mean_strain_energy += (volume * strain_energy[grain_i]);
                  const double surface_energy = 2. * interfacial_energy * (1.0/grain_size);
                  set_surface_energy(cpo_index,data,mineral_i,grain_i,surface_energy);
                }
                else
                {
                  set_surface_energy(cpo_index,data,mineral_i,grain_i,0.0);
                }
              }
          }

        if (sum_volume !=0. )
          mean_strain_energy = mean_strain_energy/sum_volume;
        Assert(isfinite(mean_strain_energy), ExcMessage("mean_strain_energy is not finite: " + std::to_string(mean_strain_energy) + "."));
        
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // compute the derivative of the rotation matrix: \frac{\partial a_{ij}}{\partial t}
            // (Eq. 9, Kaminski & Ribe 2001)
            deriv_a_cosine_matrices[grain_i] =  Utilities::Tensors::levi_civita<3>() * spin_vectors[grain_i];

            const double strain_energy = get_strain_energy(cpo_index,data,mineral_i,grain_i);
            const double surface_energy = get_surface_energy(cpo_index,data,mineral_i,grain_i);
            if(surface_energy != 0)
            {
              const double ratio = strain_energy/surface_energy;
              set_energy_ratio(cpo_index,data,mineral_i,grain_i,ratio);
            }
            else
            {
              set_energy_ratio(cpo_index,data,mineral_i,grain_i,0.0);
            }
          }

         for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            double volume_fraction_grain = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            if ((volume_fraction_grain != 0.0) && (rx_now[grain_i]) == false)
              {
                if(get_energy_ratio(cpo_index,data,mineral_i,grain_i) >= 1.0)
                {
                  const double driving_force = mean_strain_energy - get_strain_energy(cpo_index,data,mineral_i,grain_i);
                
                  // Different than D-Rex. Here we actually only compute the derivative and do not multiply it with the volume_fractions. We do that when we advect.
                  deriv_volume_fractions[grain_i] = get_volume_fraction_mineral(cpo_index,data,mineral_i) *  drexpp_mobility[mineral_i] * driving_force;                
                }
                else
                {
                  deriv_volume_fractions[grain_i] = growth_rate;
                }
              }
            else
              {
                deriv_volume_fractions[grain_i] = 0.;
              }
            
              set_grain_boundary_velocity(cpo_index,data,mineral_i,grain_i,deriv_volume_fractions[mineral_i]);  
          }

        return std::pair<std::vector<double>, std::vector<Tensor<2,3>>>(deriv_volume_fractions, deriv_a_cosine_matrices);
      
      }

      template <int dim>
      DeformationType
      CrystalPreferredOrientation<dim>::determine_deformation_type(const DeformationTypeSelector deformation_type_selector,
                                                                   const Point<dim> &position,
                                                                   const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                   const double temperature,
                                                                   const double pressure,
                                                                   const Tensor<1,dim> &velocity,
                                                                   const std::vector<double> &compositions,
                                                                   const SymmetricTensor<2,dim> &strain_rate,
                                                                   const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                                                                   const double water_content) const
      {
        // Now compute what type of deformation takes place.
        switch (deformation_type_selector)
          {
            case DeformationTypeSelector::passive:
              return DeformationType::passive;
            case DeformationTypeSelector::olivine_a_fabric:
              return DeformationType::olivine_a_fabric;
            case DeformationTypeSelector::olivine_b_fabric:
              return DeformationType::olivine_b_fabric;
            case DeformationTypeSelector::olivine_c_fabric:
              return DeformationType::olivine_c_fabric;
            case DeformationTypeSelector::olivine_d_fabric:
              return DeformationType::olivine_d_fabric;
            case DeformationTypeSelector::olivine_e_fabric:
              return DeformationType::olivine_e_fabric;
            case DeformationTypeSelector::enstatite:
              return DeformationType::enstatite;
            case DeformationTypeSelector::clinopyroxene:
              return DeformationType::clinopyroxene;
            case DeformationTypeSelector::olivine_d_0kl:
              return DeformationType::olivine_d_0kl;
            case DeformationTypeSelector::olivine_karato_2008:
              // construct the material model inputs and outputs
              // Since this function is only evaluating one particle,
              // we use 1 for the amount of quadrature points.
              MaterialModel::MaterialModelInputs<dim> material_model_inputs(1,this->n_compositional_fields());
              material_model_inputs.position[0] = position;
              material_model_inputs.temperature[0] = temperature;
              material_model_inputs.pressure[0] = pressure;
              material_model_inputs.velocity[0] = velocity;
              material_model_inputs.composition[0] = compositions;
              material_model_inputs.strain_rate[0] = strain_rate;
              material_model_inputs.current_cell = cell;

              MaterialModel::MaterialModelOutputs<dim> material_model_outputs(1,this->n_compositional_fields());
              this->get_material_model().evaluate(material_model_inputs, material_model_outputs);
              double eta = material_model_outputs.viscosities[0];

              const SymmetricTensor<2,dim> stress = 2*eta*deviatoric_strain_rate +
                                                    pressure * unit_symmetric_tensor<dim>();
              const std::array< double, dim > eigenvalues = dealii::eigenvalues(stress);
              double differential_stress = eigenvalues[0]-eigenvalues[dim-1];
              return determine_deformation_type_karato_2008(differential_stress, water_content);

          }

        AssertThrow(false, ExcMessage("Internal error. Deformation type not implemented."));
        return DeformationType::passive;
      }


      template <int dim>
      DeformationType
      CrystalPreferredOrientation<dim>::determine_deformation_type_karato_2008(const double stress, const double water_content) const
      {
        constexpr double MPa = 1e6;
        constexpr double ec_line_slope = -500./1050.;
        if (stress > (380. - 0.05 * water_content)*MPa)
          {
            if (stress > (625. - 2.5 * water_content)*MPa)
              {
                return DeformationType::olivine_b_fabric;
              }
            else
              {
                return DeformationType::olivine_d_fabric;
              }
          }
        else
          {
            if (stress < (625.0 -2.5 * water_content)*MPa)
              {
                return DeformationType::olivine_a_fabric;
              }
            else
              {
                if (stress < (500.0 + ec_line_slope*-100. + ec_line_slope * water_content)*MPa)
                  {
                    return DeformationType::olivine_e_fabric;
                  }
                else
                  {
                    return DeformationType::olivine_c_fabric;
                  }
              }
          }
      }


      template <int dim>
      std::array<double,4>
      CrystalPreferredOrientation<dim>::reference_resolved_shear_stress_from_deformation_type(DeformationType deformation_type,
          double max_value) const
      {
        std::array<double,4> ref_resolved_shear_stress;
        switch (deformation_type)
          {
            // from Kaminski and Ribe, GJI 2004 and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_a_fabric :
              ref_resolved_shear_stress[0] = 1;
              ref_resolved_shear_stress[1] = 2;
              ref_resolved_shear_stress[2] = 3;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // from Kaminski and Ribe, GJI 2004 and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_b_fabric :
              ref_resolved_shear_stress[0] = 3;
              ref_resolved_shear_stress[1] = 2;
              ref_resolved_shear_stress[2] = 1;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // from Kaminski and Ribe, GJI 2004 and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_c_fabric :
              ref_resolved_shear_stress[0] = 3;
              ref_resolved_shear_stress[1] = max_value;
              ref_resolved_shear_stress[2] = 2;
              ref_resolved_shear_stress[3] = 1;
              break;

            // from Kaminski and Ribe, GRL 2002
            case DeformationType::olivine_d_fabric :
              ref_resolved_shear_stress[0] = 1;
              ref_resolved_shear_stress[1] = 1;
              ref_resolved_shear_stress[2] = 3;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // Kaminski, Ribe and Browaeys, GJI, 2004 (same as in the matlab code) and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_e_fabric :
              ref_resolved_shear_stress[0] = 2;
              ref_resolved_shear_stress[1] = 1;
              ref_resolved_shear_stress[2] = max_value;
              ref_resolved_shear_stress[3] = 3;
              break;

            // from Kaminski and Ribe, GJI 2004.
            // Todo: this one is not used in practice, since there is an optimization in
            // the code. So maybe remove it in the future.
            case DeformationType::enstatite :
              ref_resolved_shear_stress[0] = max_value;
              ref_resolved_shear_stress[1] = max_value;
              ref_resolved_shear_stress[2] = max_value;
              ref_resolved_shear_stress[3] = 1;
              break;

            case DeformationType::clinopyroxene:
              for (unsigned int i=0; i<4; ++i)
                ref_resolved_shear_stress[i] = CPX_RRSS[i];
              break;

            case DeformationType::olivine_d_0kl:
              for (unsigned int i=0; i<4; ++i)
                ref_resolved_shear_stress[i] = OlivineD_RRSS[i];
              break;

            default:
              AssertThrow(false,
                          ExcMessage("Deformation type enum with number " + std::to_string(static_cast<unsigned int>(deformation_type))
                                     + " was not found."));
              break;
          }
        return ref_resolved_shear_stress;
      }

      template <int dim>
      unsigned int
      CrystalPreferredOrientation<dim>::get_number_of_grains() const
      {
        return n_grains;
      }



      template <int dim>
      unsigned int
      CrystalPreferredOrientation<dim>::get_number_of_minerals() const
      {
        return n_minerals;
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Crystal Preferred Orientation");
        {
          prm.declare_entry ("Random number seed", "1",
                             Patterns::Integer (0),
                             "The seed used to generate random numbers. This will make sure that "
                             "results are reproducible as long as the problem is run with the "
                             "same number of MPI processes. It is implemented as final seed = "
                             "user seed + MPI Rank.");

          prm.declare_entry ("Number of grains per particle", "50",
                             Patterns::Integer (1),
                             "The number of grains of each different mineral "
                             "each particle contains.");

          prm.declare_entry ("Property advection method", "Backward Euler",
                             Patterns::Anything(),
                             "Options: Forward Euler, Backward Euler");

          prm.declare_entry ("Property advection tolerance", "1e-10",
                             Patterns::Double(0),
                             "The Backward Euler property advection method involve internal iterations. "
                             "This option allows for setting a tolerance. When the norm of tensor new - tensor old is "
                             "smaller than this tolerance, the iteration is stopped.");

          prm.declare_entry ("Property advection max iterations", "100",
                             Patterns::Integer(0),
                             "The Backward Euler property advection method involve internal iterations. "
                             "This option allows for setting the maximum number of iterations. Note that when the iteration "
                             "is ended by the max iteration amount an assert is thrown.");

          prm.declare_entry ("CPO derivatives algorithm", "Spin tensor",
                             Patterns::List(Patterns::Anything()),
                             "Options: Spin tensor");

          prm.enter_subsection("Initial grains");
          {
            prm.declare_entry("Model name", "Uniform grains and random uniform rotations",
                              Patterns::Anything(),
                              "The model used to initialize the CPO for all particles. "
                              "Currently 'Uniform grains and random uniform rotations' and 'World Builder' are the only valid option.");

            prm.declare_entry ("Minerals", "Olivine: Karato 2008, Enstatite",
                               Patterns::List(Patterns::Anything()),
                               "This determines what minerals and fabrics or fabric selectors are used used for the LPO/CPO calculation. "
                               "The options are Olivine: Passive, A-fabric, Olivine: B-fabric, Olivine: C-fabric, Olivine: D-fabric, "
                               "Olivine: E-fabric, Olivine: Karato 2008 or Enstatite or CPX. Passive sets all RRSS entries to the maximum. The "
                               "Karato 2008 selector selects a fabric based on stress and water content as defined in "
                               "figure 4 of the Karato 2008 review paper (doi: 10.1146/annurev.earth.36.031207.124120).");

            prm.declare_entry ("CPX RRSS", "1,5,5,1.5",
                               Patterns::List(Patterns::Anything()),
                               "The default RRSS values for CPX, used in fabric calculations."
                               "(preliminary results, pending further investigations)."
                               "This list expects 4 entries separated by commas."
                               "Main slip systems from Bascou et al., 2002 JSG and "
                               "Zhang et al., 2006 EPSL and from numerical experiments");
            prm.declare_entry("Input orientation file","input_orientation.dat",
                                Patterns::Anything(),
                                "The model used to initialize the CPO for all particles. "
                                "Currently 'Uniform grains and random uniform rotations' and 'World Builder' are the only valid option.");

            prm.declare_entry ("OlivineD RRSS", "3.,5.,1.e60,1.",
                               Patterns::List(Patterns::Anything()),
                               "Alternative RRSS values for Olivine D-type fabric when "
                               "incorporating slip plane {0kl}, used in fabric calculations."
                               "(preliminary results, pending further investigations)."
                               "This list expects 4 entries separated by commas."
                               "Main slip systems from Karato 2008 and "
                               "Bystricky et al., 2001 and from numerical experiments");
            prm.declare_entry("Input orientation file","input_orientation.dat",
                                Patterns::Anything(),
                                "The model used to initialize the CPO for all particles. "
                                "Currently 'Uniform grains and random uniform rotations' and 'World Builder' are the only valid option.");

            prm.declare_entry ("Volume fractions minerals", "0.7, 0.3",
                               Patterns::List(Patterns::Double(0)),
                               "The volume fractions for the different minerals. "
                               "There need to be the same number of values as there are minerals. "
                               "Note that the currently implemented scheme is incompressible and "
                               "does not allow chemical interaction or the formation of new phases.");
          }
          prm.leave_subsection();

          prm.enter_subsection("D-Rex 2004");
          {
            prm.declare_entry ("Mobility", "50",
                               Patterns::Double(0),
                               "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");

            prm.declare_entry ("Volume fractions minerals", "0.5, 0.5",
                               Patterns::List(Patterns::Double(0)),
                               "The volume fraction for the different minerals. "
                               "There need to be the same amount of values as there are minerals");

            prm.declare_entry ("Stress exponents", "3.5",
                               Patterns::Double(0),
                               "This is the power law exponent that characterizes the rheology of the "
                               "slip systems. It is used in equation 11 of Kaminski et al., 2004.");

            prm.declare_entry ("Exponents p", "1.5",
                               Patterns::Double(0),
                               "This is exponent p as defined in equation 11 of Kaminski et al., 2004.");

            prm.declare_entry ("Nucleation efficiency", "5",
                               Patterns::Double(0),
                               "This is the dimensionless nucleation rate as defined in equation 8 of "
                               "Kaminski et al., 2004.");

            prm.declare_entry ("Threshold GBS", "0.3",
                               Patterns::Double(0),
                               "The Dimensionless Grain Boundary Sliding (GBS) threshold. "
                               "This is a grain size threshold below which grains deform by GBS and "
                               "become strain-free grains.");
          }
          prm.leave_subsection();

          prm.enter_subsection("D-Rex++");
          {
            prm.declare_entry ("Number of initial grains","500",
                                   Patterns::List(Patterns::Double(0)),
                                   "Initial no. of grains we want to start the model with." );

                prm.declare_entry ("Number of buffer grains","500",
                                   Patterns::List(Patterns::Double(0)),
                                   "Initial no. of grains we want to start the model with." );

                prm.declare_entry ("Mobility", "125",
                                   Patterns::List(Patterns::Double(0)),
                                   "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");

                prm.declare_entry ("Interfacial Energy", "0.1",
                                   Patterns::List(Patterns::Double(0)),
                                   "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");

                prm.declare_entry ("Avrami Slope Input", "0.15",
                                   Patterns::Double(0),
                                   "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");

                prm.declare_entry ("Volume fractions minerals", "0.5, 0.5",
                                   Patterns::List(Patterns::Double(0)),
                                   "The volume fraction for the different minerals. "
                                   "There need to be the same amount of values as there are minerals");

                prm.declare_entry ("Stress exponents", "3.5",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is the power law exponent that characterizes the rheology of the "
                                   "slip systems. It is used in equation 11 of Kaminski et al., 2004.");

                prm.declare_entry ("Exponents p", "1.5",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is exponent p as defined in equation 11 of Kaminski et al., 2004. ");

                prm.declare_entry ("Initial grain size", "1e-6",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is intial grain size we choose to prescribe to Drex ++ ");
              
                prm.declare_entry ("max dispersion angle", "15",
                                    Patterns::List(Patterns::Double(0)),
                                    "This is intial grain size we choose to prescribe to Drex ++ ");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
          prm.enter_subsection("Material model");
        {
          prm.enter_subsection ("Visco Plastic");
          {
             MaterialModel::MaterialUtilities::PhaseFunction<dim>::declare_parameters(prm);
              prm.declare_entry ("Thermal diffusivities", "1e-6",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is intial grain size we choose to prescribe to Drex ++ ");
              
              prm.declare_entry ("Define thermal conductivities", "false",           
                                  Patterns::Bool(),
                                  "This is intial grain size we choose to prescribe to Drex ++ ");
              
              prm.declare_entry ("Thermal conductivities", "1e-6",           
                                  Patterns::List(Patterns::Double(0)),
                                  "This is intial grain size we choose to prescribe to Drex ++ ");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::parse_parameters (ParameterHandler &prm)
      {
        AssertThrow(dim == 3, ExcMessage("CPO computations are currently only supported for 3d models. "
                                         "2d computations will work when this assert is removed, but you will need to make sure that the "
                                         "correct 3d strain-rate and velocity gradient tensors are provided to the algorithm."));

        prm.enter_subsection("Crystal Preferred Orientation");
        {
          random_number_seed = prm.get_integer ("Random number seed");
          n_grains = prm.get_integer("Number of grains per particle");

          property_advection_tolerance = prm.get_double("Property advection tolerance");
          property_advection_max_iterations = prm.get_integer ("Property advection max iterations");

          const std::string temp_cpo_derivative_algorithm = prm.get("CPO derivatives algorithm");

          if (temp_cpo_derivative_algorithm == "Spin tensor")
            {
              cpo_derivative_algorithm = CPODerivativeAlgorithm::spin_tensor;
            }
          else if (temp_cpo_derivative_algorithm ==  "D-Rex 2004")
            {
              cpo_derivative_algorithm = CPODerivativeAlgorithm::drex_2004;
            }
          else if (temp_cpo_derivative_algorithm ==  "D-Rex++")
            {
              cpo_derivative_algorithm = CPODerivativeAlgorithm::drexpp;
            }
          else
            {
              AssertThrow(false,
                          ExcMessage("The CPO derivatives algorithm needs to be one of the following: "
                                     "Spin tensor, D-Rex 2004."));
            }

          const std::string temp_advection_method = prm.get("Property advection method");
          if (temp_advection_method == "Forward Euler")
            {
              advection_method = AdvectionMethod::forward_euler;
            }
          else if (temp_advection_method == "Backward Euler")
            {
              advection_method = AdvectionMethod::backward_euler;
            }
          else
            {
              AssertThrow(false, ExcMessage("particle property advection method not found: \"" + temp_advection_method + "\""));
            }

          prm.enter_subsection("Initial grains");
          {
            const std::string model_name = prm.get("Model name");
            if (model_name == "Uniform grains and random uniform rotations")
              {
                initial_grains_model = CPOInitialGrainsModel::uniform_grains_and_random_uniform_rotations;
              }
            else if (model_name == "World Builder")
              {
                initial_grains_model = CPOInitialGrainsModel::world_builder;
              }
            else if (model_name == "Pre-existing fabric")
              {
                initial_grains_model = CPOInitialGrainsModel::pre_existing_fabric;
                input_orientation_file = prm.get("Input orientation file");
              }
            else
              {
                AssertThrow(false,
                            ExcMessage("No model named " + model_name + "for CPO particle property initialization. "
                                       + "Only the model \"Uniform grains and random uniform rotations\"  and "
                                       "\"World Builder\" are available."));
              }

            const std::vector<std::string> temp_deformation_type_selector = dealii::Utilities::split_string_list(prm.get("Minerals"));
            n_minerals = temp_deformation_type_selector.size();
            deformation_type_selector.resize(n_minerals);

            for (size_t mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
              {
                if (temp_deformation_type_selector[mineral_i] == "Passive")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::passive;
                  }
                else if (temp_deformation_type_selector[mineral_i] == "Olivine: Karato 2008")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_karato_2008;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: A-fabric")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_a_fabric;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: B-fabric")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_b_fabric;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: C-fabric")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_c_fabric;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: D-fabric")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_d_fabric;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: E-fabric")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_e_fabric;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Enstatite")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::enstatite;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Clinopyroxene")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::clinopyroxene;
                  }
                else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: D-fabric_0kl")
                  {
                    deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_d_0kl;
                  }
                else
                  {
                    AssertThrow(false,
                                ExcMessage("The fabric needs to be assigned one of the following comma-delimited values: Olivine: Karato 2008, "
                                           "Olivine: A-fabric, Olivine: B-fabric, Olivine: C-fabric, Olivine: D-fabric,"
                                           "Olivine: E-fabric, Enstatite, Passive, Clinopyroxene, Olivine: D-fabric_0kl."));
                  }
              }

            CPX_RRSS = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("CPX RRSS")));
            AssertThrow(CPX_RRSS.size()==4,
                        ExcMessage("The number of Reference Resolved Shear Stress (RRSS) entries for CPX has to be equal to four."));

            OlivineD_RRSS = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("OlivineD RRSS")));
            AssertThrow(OlivineD_RRSS.size()==4,
                        ExcMessage("The number of Reference Resolved Shear Stress (RRSS) entries for alternative Olivine D has to be equal to four."));

            volume_fractions_minerals = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Volume fractions minerals")));
            double volume_fractions_minerals_sum = 0;
            for (auto fraction : volume_fractions_minerals)
              {
                volume_fractions_minerals_sum += fraction;
              }

            AssertThrow(std::abs(volume_fractions_minerals_sum-1.0) < 2.0 * std::numeric_limits<double>::epsilon(),
                        ExcMessage("The sum of the CPO volume fractions should be one."));
          }
          prm.leave_subsection();

          prm.enter_subsection("D-Rex 2004");
          {
            mobility = prm.get_double("Mobility");
            volume_fractions_minerals = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Volume fractions minerals")));
            stress_exponent = prm.get_double("Stress exponents");
            exponent_p = prm.get_double("Exponents p");
            nucleation_efficiency = prm.get_double("Nucleation efficiency");
            threshold_GBS = prm.get_double("Threshold GBS");
          }
          prm.leave_subsection();
          
          prm.enter_subsection("D-Rex++");
              {
                n_grains_init = prm.get_double("Number of initial grains");
                n_grains_buffer = prm.get_double("Number of buffer grains");
                drexpp_mobility = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Mobility")));
                volume_fractions_minerals = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Volume fractions minerals")));
                drexpp_stress_exponent = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Stress exponents")));
                drexpp_exponent_p =  Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Exponents p")));
                avrami_slope_input = prm.get_double("Avrami Slope Input");
                interfacial_energy = prm.get_double("Interfacial Energy");
                initial_grain_size = prm.get_double("Initial grain size");
                max_dispersion = prm.get_double("max dispersion angle");
              }
          prm.leave_subsection();
        
        }
        prm.leave_subsection ();

        prm.enter_subsection("Material model");
        {
          prm.enter_subsection ("Visco Plastic");
          {
             // Phase transition parameters
             phase_function = std::make_shared<MaterialModel::MaterialUtilities::PhaseFunction<dim>>();
             phase_function->initialize_simulator(this->get_simulator());
             phase_function->declare_parameters(prm);
             phase_function->parse_parameters(prm);
             

             // Retrieve the list of composition names
             const std::vector<std::string> list_of_composition_names = this->introspection().get_composition_names();
 
             // Establish that a background field is required here
             const bool has_background_field = true;
 
            thermal_diffusivities = Utilities::parse_map_to_double_array (prm.get("Thermal diffusivities"),
                                                                           list_of_composition_names,
                                                                           has_background_field,
                                                                           "Thermal diffusivities");
 
             define_conductivities = prm.get_bool ("Define thermal conductivities");
 
             thermal_conductivities = Utilities::parse_map_to_double_array (prm.get("Thermal conductivities"),
                                                                            list_of_composition_names,
                                                                            has_background_field,
                                                                            "Thermal conductivities");
 
                                                                            
           
            rheology_vipl = std::make_shared<MaterialModel::Rheology::ViscoPlastic<dim>>();
             rheology_vipl->initialize_simulator (this->get_simulator());
             rheology_vipl->declare_parameters(prm);
             rheology_vipl->parse_parameters(prm, std::make_unique<std::vector<unsigned int>>(phase_function->n_phases_for_each_chemical_composition()));
             min_strain_rate = rheology_vipl->min_strain_rate;
             
             rheology_diff = std::make_shared<MaterialModel::Rheology::DiffusionCreep<dim>>();
             rheology_diff->initialize_simulator (this->get_simulator());
             rheology_diff->parse_parameters(prm, std::make_unique<std::vector<unsigned int>>(phase_function->n_phases_for_each_chemical_composition()));
 
             rheology_disl = std::make_shared<MaterialModel::Rheology::DislocationCreep<dim>>();
             rheology_disl->initialize_simulator (this->get_simulator());
             rheology_disl->parse_parameters(prm, std::make_unique<std::vector<unsigned int>>(phase_function->n_phases_for_each_chemical_composition()));

              
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();

      
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
      ASPECT_REGISTER_PARTICLE_PROPERTY(CrystalPreferredOrientation,
                                        "crystal preferred orientation",
                                        "WARNING: all the CPO plugins are a work in progress and not ready for production use yet. "
                                        "See https://github.com/geodynamics/aspect/issues/3885 for current status and alternatives. "
                                        "The plugin manages and computes the evolution of Lattice/Crystal Preferred Orientations (LPO/CPO) "
                                        "on particles. Each ASPECT particle can be assigned many grains. Each grain is assigned a size and a orientation "
                                        "matrix. This allows for CPO evolution tracking with polycrystalline kinematic CrystalPreferredOrientation evolution models such "
                                        "as D-Rex (Kaminski and Ribe, 2001; Kaminski et al., 2004).")
    }
  }
}
