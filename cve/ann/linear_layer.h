/*====================================================================
BSD 2-Clause License

Copyright (c) 2018, Ruler
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
====================================================================*/
#pragma once

#ifndef __ANN_LINEAR_LAYER_H__
#define __ANN_LINEAR_LAYER_H__

#include "layer_base.h"

namespace ann
{
	// Class template linear_layer
	template <class T, class Allocator = ::core::allocator<T> >
	class linear_layer : public layer_base<T, Allocator>
	{
	public:
		// types:

		typedef layer_base<T, Allocator>                         layer_base_type;

		typedef typename layer_base_type::allocator_type         allocator_type;
		typedef typename layer_base_type::allocator_traits_type  allocator_traits_type;
		typedef typename layer_base_type::scalar_type            scalar_type;
		typedef typename layer_base_type::vector_type            vector_type;
		typedef typename layer_base_type::matrix_type            matrix_type;
		typedef typename layer_base_type::tensor_type            tensor_type;
		typedef typename layer_base_type::const_scalar_type      const_scalar_type;
		typedef typename layer_base_type::const_vector_type      const_vector_type;
		typedef typename layer_base_type::const_matrix_type      const_matrix_type;
		typedef typename layer_base_type::const_tensor_type      const_tensor_type;
		typedef typename layer_base_type::scalar_pointer         scalar_pointer;
		typedef	typename layer_base_type::vector_pointer         vector_pointer;
		typedef	typename layer_base_type::matrix_pointer         matrix_pointer;
		typedef	typename layer_base_type::tensor_pointer         tensor_pointer;
		typedef typename layer_base_type::const_scalar_pointer   const_scalar_pointer;
		typedef	typename layer_base_type::const_vector_pointer   const_vector_pointer;
		typedef	typename layer_base_type::const_matrix_pointer   const_matrix_pointer;
		typedef	typename layer_base_type::const_tensor_pointer   const_tensor_pointer;
		typedef	typename layer_base_type::scalar_reference       scalar_reference;
		typedef	typename layer_base_type::vector_reference       vector_reference;
		typedef	typename layer_base_type::matrix_reference       matrix_reference;
		typedef	typename layer_base_type::tensor_reference       tensor_reference;
		typedef	typename layer_base_type::const_scalar_reference const_scalar_reference;
		typedef	typename layer_base_type::const_vector_reference const_vector_reference;
		typedef	typename layer_base_type::const_matrix_reference const_matrix_reference;
		typedef	typename layer_base_type::const_tensor_reference const_tensor_reference;

		typedef typename layer_base_type::value_type             value_type;
		typedef typename layer_base_type::pointer                pointer;
		typedef typename layer_base_type::const_pointer          const_pointer;
		typedef typename layer_base_type::reference              reference;
		typedef typename layer_base_type::const_reference        const_reference;
		typedef typename layer_base_type::size_type              size_type;
		typedef typename layer_base_type::difference_type        difference_type;

		// construct/copy/destroy:

		linear_layer(const Allocator& alloc = Allocator())
		{}

		linear_layer(size_type input_row_size, size_type output_row_size, value_type learn, bool training = false)
		{
			assign(input_row_size, output_row_size, learn, training);
		}

		void assign(size_type input_row_size, size_type output_row_size, value_type learn, bool training = false)
		{
			const size_type dimension = 1;
			weight.assign(input_row_size, output_row_size, dimension);
			bias.assign(output_row_size, dimension);
			if (training)
			{
				this->rate = learn;
				// The gradient of weight
				weight_gradient.assign(input_row_size, output_row_size, dimension);
				// The gradient of bias
				bias_gradient.assign(output_row_size, dimension);
				// The mean vector of the data
				data_mean.assign(input_row_size, dimension);
				// The mean vector of the loss
				loss_mean.assign(output_row_size, dimension);
			}
		}

		// Initialize linear layer

		void initialize_uniform(T min, T max, unsigned int seed = 1U)
		{
			::std::default_random_engine engine(seed);
			::std::uniform_real_distribution<T> distribution(min, max);
			::std::function<T(void)> generator = ::std::bind(distribution, engine);
			weight.generate(generator);
			bias.fill(0);
		}

		void initialize_normal(T mean, T sigma, unsigned int seed = 1U)
		{
			::std::default_random_engine engine(seed);
			::std::normal_distribution<T> distribution(mean, sigma);
			::std::function<T(void)> generator = ::std::bind(distribution, engine);
			weight.generate(generator);
			bias.fill(0);
		}

		// Forward propagation
		tensor_reference forward(tensor_reference data)
		{
			if (data.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);
			if (data.matrix_size() != weight.rows())
				throw ::std::invalid_argument(::core::invalid_shape);

			constexpr size_type one = 1;
			this->input.reassign(one, data.batch(), data.area(), data.dimension(), data.data(), false);
			this->output.reassign(one, data.batch(), weight.columns(), weight.dimension());
			if (this->train)
				this->error.reassign(this->input, ::core::without_copy);
			::core::cpu_matmul(this->output[0], this->input[0], weight);
			this->output.reshape(this->output.rows(), one, this->output.columns(), this->output.dimension());
			return this->output;
		}

		// Update
		tensor_reference Update(tensor_reference loss)
		{
			if (loss.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);
			if (loss.size() != this->output.size())
				throw ::std::invalid_argument(::core::invalid_size);

			// Mean vector of the data
			::core::reduce(data_mean, this->input[0], ::core::reduce_col_avg);
			// Mean vector of loss data
			::core::reduce(loss_mean, loss[0], ::core::reduce_col_avg);
			// Calculate the gradient of weight
			::core::cpu_matmul(weight_gradient, data_mean, loss_mean);
			// Update the weights
			::core::cpu_madd(weight, -this->rate, weight_gradient);
			// Update the bias
			::core::cpu_madd(bias, -this->rate, loss_mean);
		}

		// Back propagation
		tensor_reference backward(void)
		{
			::core::cpu_matmul(this->error, loss_mean, weight, true);
			return this->error;
		}
	private:
		matrix_type weight;
		vector_type bias;
		matrix_type weight_gradient;
		vector_type bias_gradient;
		vector_type data_mean;
		vector_type loss_mean;
	};

} // namespace ann

#endif
