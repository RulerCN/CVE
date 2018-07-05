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

		linear_layer(size_type input_row_size, size_type output_row_size, bool training = false)
		{
			assign(input_row_size, output_row_size, training);
		}

		void assign(size_type input_row_size, size_type output_row_size, bool training = false)
		{
			const size_type dimension = 1;
			weight.assign(input_row_size, output_row_size, dimension);
			bias.assign(output_row_size, dimension);
			if (training)
			{
				// The gradient of weight
				weight_gradient.assign(input_row_size, output_row_size, dimension);
				// The gradient of bias
				bias_gradient.assign(output_row_size, dimension);
				// The mean vector of the input
				input_mean.assign(input_row_size, dimension);
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
		void forward(const_tensor_reference input, tensor_reference output)
		{
			if (input.batch() != 1 || output.batch() != 1)
				throw ::std::invalid_argument(::core::invalid_shape);
			::core::cpu_matmul(output[0], input[0], weight);
			this->bind(input, output);
		}

		// Back propagation
		void backward(const_tensor_reference input, tensor_reference output, T rate)
		{
			if (input.batch() != 1 || output.batch() != 1)
				throw ::std::invalid_argument(::core::invalid_shape);
			// Mean vector of input data
			::core::reduce(input_mean, *this->input[0], ::core::reduce_col_avg);
			// Mean vector of loss data
			::core::reduce(loss_mean, input[0], ::core::reduce_col_avg);
			// Calculate the gradient of weight
			::core::cpu_matmul(weight_gradient, input_mean, loss_mean);
			// Update the weights
			::core::cpu_muladd(weight, -rate, weight_gradient);
		}
	private:
		matrix_type weight;
		matrix_type weight_gradient;
		vector_type bias;
		vector_type bias_gradient;
		vector_type input_mean;
		vector_type loss_mean;
	};

} // namespace ann

#endif
