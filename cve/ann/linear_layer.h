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
			: layer_base_type()
		{}

		linear_layer(size_type in_dim, size_type out_dim, bool has_bias, bool train, value_type rate)
		{
			assign(in_dim, out_dim, has_bias, rate, train);
		}

		void assign(size_type in_dim, size_type out_dim, bool has_bias = true)
		{
			_in_dim = in_dim;
			_out_dim = out_dim;
			_weight.assign(_in_dim, _out_dim, size_t(1));
			if (has_bias)
				_bias.assign(_out_dim, size_t(1));
		}

		// Forward propagation
		virtual tensor_reference forward(tensor_reference input)
		{
			if (input.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);
			if (input.matrix_size() != _weight.rows())
				throw ::std::invalid_argument(::core::invalid_shape);

			size_t batch = input.batch();
			// Reassign an output tensor
			_output.reassign(size_t(1), batch, _out_dim, size_t(1));
			// Reshape an input tensor
			input.shape(size_t(1), batch, _in_dim, size_t(1));
			// Assign an input matrix
			if (_input.empty())
				_input.assign(size_t(1), _in_dim, size_t(1));
			// Mean value of input tensor
			::core::cpu_mean(_input, input, ::core::axis_y);
			// output = intput * weight + bias
			if (has_bias)
				::core::cpu_gemm(_output, input, _weight, _bias);
			else
				::core::cpu_gemm(_output, input, _weight);
			// Reshape an output tensor
			_output.shape(batch, size_t(1), _out_dim, size_t(1));
			return _output;
		}

		// Update parameters
		virtual void update(tensor_reference loss)
		{
			if (loss.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);
			if (loss.size() != _out_dim)
				throw ::std::invalid_argument(::core::invalid_size);

			if (weight_gradient.empty())
				weight_gradient.assign(weight, ::core::without_copy);
			// weight_gradient = input^T * loss;
			::core::cpu_gtvv(weight_gradient, _input, loss);
			// weight -= rate * weight_gradient;
			::core::cpu_madd(weight, -this->rate, weight_gradient);
			// bias -= rate * loss;
			if (has_bias)
				::core::cpu_madd(bias, -this->rate, loss);
		}

		// Back propagation
		virtual tensor_reference backward(tensor_reference loss)
		{
			// Assign a loss vector
			if (_loss.empty())
				_loss.assign(size_t(1), _in_dim, size_t(1));
			// _loss = loss * weight^T
			::core::cpu_gemm(_loss, loss, weight, true);
			return _loss;
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
	private:
		size_type   _in_dim;
		size_type   _out_dim;
		matrix_type _weight;
		vector_type _bias;
		matrix_type _input;
		tensor_type _output;
		vector_type _loss;
		matrix_type weight_gradient;
	};

} // namespace ann

#endif
