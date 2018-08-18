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

#ifndef __ANN_SOFTMAX_LAYER_H__
#define __ANN_SOFTMAX_LAYER_H__

#include "layer_base.h"

namespace ann
{
	// Class template softmax_layer
	template <class T, class Allocator = ::core::allocator<T> >
	class softmax_layer : public layer_base<T, Allocator>
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

		softmax_layer(const Allocator& alloc = Allocator())
		{}

		softmax_layer(size_type dimension, const Allocator& alloc = Allocator())
		{
			assign(dimension);
		}

		void assign(size_type dimension)
		{
			this->input_dimension = dimension;
			this->output_dimension = dimension;
			temporary_vector.assign(dimension, size_type(1));
		}

		// Forward propagation
		tensor_reference forward(tensor_reference data)
		{
			if (data.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);
			if (data.matrix_size() != this->input_dimension)
				throw ::std::invalid_argument(::core::invalid_shape);

			this->input.reassign(size_type(1), data.batch(), data.area(), data.dimension(), data.data());
			this->output.reassign(this->input, ::core::without_copy);
			matrix_type output_matrix = this->output[0];
			const_matrix_type input_matrix = this->input[0];
			::core::cpu_reduce_mean(temporary_vector, input_matrix, ::core::axis_y);
			::core::cpu_sub(output_matrix, input_matrix, temporary_vector);
			::core::cpu_exp(output_matrix, output_matrix);
			::core::cpu_reduce_sum(temporary_vector, output_matrix, ::core::axis_x);
			::core::cpu_div(output_matrix, output_matrix, temporary_vector);
			this->input.reshape(data.batch(), data.rows(), data.columns(), data.dimension());
			this->output.reshape(data.batch(), data.rows(), data.columns(), data.dimension());
			return this->output;
		}

		// Back propagation
		template<class T, class A>
		tensor_reference backward(const ::core::tensor<T, A> &labels)
		{
			if (labels.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);

			this->error.reassign(this->output, ::core::deep_copy);
			::core::cpu_onehot_sub(this->output, labels);
			return this->error;
		}
	private:
		vector_type temporary_vector;
	};

} // namespace ann

#endif
