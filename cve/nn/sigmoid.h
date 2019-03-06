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

#ifndef __NN_SIGMOID_H__
#define __NN_SIGMOID_H__

#include "layer.h"

namespace nn
{
	// Class template sigmoid
	template <class T, class Allocator = ::core::allocator<T> >
	class sigmoid : public layer<T, Allocator>
	{
	public:
		// types:

		typedef layer<T, Allocator>                         layer_type;

		typedef typename layer_type::allocator_type         allocator_type;
		typedef typename layer_type::allocator_traits_type  allocator_traits_type;
		typedef typename layer_type::scalar_type            scalar_type;
		typedef typename layer_type::vector_type            vector_type;
		typedef typename layer_type::matrix_type            matrix_type;
		typedef typename layer_type::tensor_type            tensor_type;
		typedef typename layer_type::const_scalar_type      const_scalar_type;
		typedef typename layer_type::const_vector_type      const_vector_type;
		typedef typename layer_type::const_matrix_type      const_matrix_type;
		typedef typename layer_type::const_tensor_type      const_tensor_type;
		typedef typename layer_type::scalar_pointer         scalar_pointer;
		typedef	typename layer_type::vector_pointer         vector_pointer;
		typedef	typename layer_type::matrix_pointer         matrix_pointer;
		typedef	typename layer_type::tensor_pointer         tensor_pointer;
		typedef typename layer_type::const_scalar_pointer   const_scalar_pointer;
		typedef	typename layer_type::const_vector_pointer   const_vector_pointer;
		typedef	typename layer_type::const_matrix_pointer   const_matrix_pointer;
		typedef	typename layer_type::const_tensor_pointer   const_tensor_pointer;

		typedef typename layer_type::value_type             value_type;
		typedef typename layer_type::pointer                pointer;
		typedef typename layer_type::const_pointer          const_pointer;
		typedef typename layer_type::size_type              size_type;
		typedef typename layer_type::difference_type        difference_type;

		// construct/copy/destroy:

		sigmoid(const Allocator& alloc = Allocator())
			: layer_type()
		{}

		// Forward propagation
		virtual tensor_type& forward(tensor_type& input)
		{
			if (input.empty())
				throw ::std::domain_error(::core::tensor_not_initialized);

			// Reassign an input tensor
			_input.reassign(input, ::core::shallow_copy);
			// Reassign an output tensor
			_output.reassign(input, ::core::without_copy);
			// _output = sigmoid(input)
			//::core::cpu_sigmoid(_output, _input);
			return _output;
		}

		// Update parameters
		virtual void update(tensor_type& /*loss*/)
		{}

		// Back propagation
		virtual tensor_type& backward(tensor_type& loss)
		{
			// Reassign an loss tensor
			_loss.reassign(_input, ::core::without_copy);
			// _loss = desigmoid(loss, _output)
			::core::cpu_desigmoid(_loss, loss, _output);
			return _loss;
		}
	private:
		tensor_type _input;
		tensor_type _output;
		tensor_type _loss;
	};

} // namespace nn

#endif
