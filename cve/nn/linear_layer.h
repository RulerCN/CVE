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

#ifndef __NN_LINEAR_LAYER_H__
#define __NN_LINEAR_LAYER_H__

#include "layer_base.h"

namespace nn
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
			, _rate(0)
			, _in_dim(0)
			, _out_dim(0)
			, _has_bias(false)
		{}
		linear_layer(size_type in_dim, size_type out_dim, bool has_bias, value_type rate)
			: layer_base_type()
			, _rate(rate)
			, _in_dim(0)
			, _out_dim(0)
			, _has_bias(false)
		{
			assign(in_dim, out_dim, has_bias);
		}
		linear_layer(size_type in_dim, size_type out_dim, bool has_bias, value_type rate, value_type mean, value_type sigma, unsigned int seed = 1U)
			: layer_base_type()
			, _rate(rate)
			, _in_dim(0)
			, _out_dim(0)
			, _has_bias(false)
		{
			assign(in_dim, out_dim, has_bias);
			normal(mean, sigma, seed);
		}

		void assign(size_type in_dim, size_type out_dim, bool has_bias = true)
		{
			_in_dim = in_dim;
			_out_dim = out_dim;
			_has_bias = has_bias;
			_weight.assign(size_t(1), _in_dim, _out_dim, size_t(1));
			if (_has_bias)
				_bias.assign(size_t(1), size_t(1), _out_dim, size_t(1));
		}

		// Initialization parameter

		void uniform(value_type min, value_type max, unsigned int seed = 1U)
		{
			::std::default_random_engine engine(seed);
			::std::uniform_real_distribution<value_type> distribution(min, max);
			::std::function<value_type(void)> generator = ::std::bind(distribution, engine);
			_weight.generate(generator);
			if (_has_bias)
				_bias.fill(0);
		}

		void normal(value_type mean, value_type sigma, unsigned int seed = 1U)
		{
			::std::default_random_engine engine(seed);
			::std::normal_distribution<value_type> distribution(mean, sigma);
			::std::function<value_type(void)> generator = ::std::bind(distribution, engine);
			_weight.generate(generator);
			if (_has_bias)
				_bias.fill(0);
		}

		// Forward propagation
		virtual tensor_reference forward(tensor_reference input)
		{
			if (input.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);
			if (input.matrix_size() != _weight.rows())
				throw ::std::invalid_argument(::core::invalid_shape);

			size_t batch = input.batch();
			// Assign an input matrix
			if (_input.empty())
				_input.assign(size_t(1), size_t(1), _in_dim, size_t(1));
			// Reassign an output tensor
			_output.reassign(size_t(1), batch, _out_dim, size_t(1));
			// Reshape an input tensor
			input.shape(size_t(1), batch, _in_dim, size_t(1));
			// Mean value of input tensor
			::core::cpu_mean(_input, input, ::core::axis_y);
			// output = intput * weight + bias
			if (_has_bias)
			{
				::core::cpu_repeat(_output, _bias, size_t(1), size_t(batch), size_t(1));
				::core::cpu_addmmt(_output, input, _weight);
			}
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

			if (_w_grad.empty())
				_w_grad.assign(_weight, ::core::without_copy);
			// _w_grad = input^T * loss;
			::core::cpu_gtvv(_w_grad, _input, loss);
			// weight -= rate * _w_grad;
			::core::cpu_madd(_weight, -_rate, _w_grad);
			if (_has_bias)
			{
				if (_b_grad.empty())
					_b_grad.assign(_bias, ::core::without_copy);
				// _b_grad = loss;
				_b_grad = loss;
				// bias -= rate * loss;
				::core::cpu_madd(_bias, -_rate, _b_grad);
			}
		}

		// Back propagation
		virtual tensor_reference backward(tensor_reference loss)
		{
			// Assign a loss vector
			if (_loss.empty())
				_loss.assign(size_t(1), size_t(1), _in_dim, size_t(1));
			// _loss = loss * weight^T
			::core::cpu_gemm(_loss, loss, _weight, true);
			return _loss;
		}
	private:
		value_type  _rate;
		bool        _has_bias;
		size_type   _in_dim;
		size_type   _out_dim;
		tensor_type _weight;
		tensor_type _bias;
		tensor_type _w_grad;
		tensor_type _b_grad;
		tensor_type _input;
		tensor_type _output;
		tensor_type _loss;
	};

} // namespace nn

#endif
