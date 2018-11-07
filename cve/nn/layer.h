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

#ifndef __NN_LAYER_H__
#define __NN_LAYER_H__

#include <random>
#include <functional>
#include "../core/core.h"

namespace nn
{
	// Class template layer
	template <class T, class Allocator = ::core::allocator<T> >
	class layer
	{
	public:
		// types:

		typedef Allocator                                       allocator_type;
		typedef ::std::allocator_traits<Allocator>              allocator_traits_type;

		typedef ::core::scalar<T, Allocator>                    scalar_type;
		typedef ::core::vector<T, Allocator>                    vector_type;
		typedef ::core::matrix<T, Allocator>                    matrix_type;
		typedef ::core::tensor<T, Allocator>                    tensor_type;
		typedef const scalar_type                               const_scalar_type;
		typedef	const vector_type                               const_vector_type;
		typedef	const matrix_type                               const_matrix_type;
		typedef	const tensor_type                               const_tensor_type;
		typedef scalar_type*                                    scalar_pointer;
		typedef	vector_type*                                    vector_pointer;
		typedef	matrix_type*                                    matrix_pointer;
		typedef	tensor_type*                                    tensor_pointer;
		typedef const scalar_type*                              const_scalar_pointer;
		typedef	const vector_type*                              const_vector_pointer;
		typedef	const matrix_type*                              const_matrix_pointer;
		typedef	const tensor_type*                              const_tensor_pointer;
		typedef scalar_type&                                    scalar_reference;
		typedef	vector_type&                                    vector_reference;
		typedef	matrix_type&                                    matrix_reference;
		typedef	tensor_type&                                    tensor_reference;
		typedef const scalar_type&                              const_scalar_reference;
		typedef	const vector_type&                              const_vector_reference;
		typedef	const matrix_type&                              const_matrix_reference;
		typedef	const tensor_type&                              const_tensor_reference;

		typedef typename allocator_traits_type::value_type      value_type;
		typedef typename allocator_traits_type::pointer         pointer;
		typedef typename allocator_traits_type::const_pointer   const_pointer;
		typedef typename allocator_type::reference              reference;
		typedef typename allocator_type::const_reference        const_reference;
		typedef typename allocator_traits_type::size_type       size_type;
		typedef typename allocator_traits_type::difference_type difference_type;

		// construct/copy/destroy:
		layer(void) {}
		// Forward propagation
		virtual tensor_reference forward(tensor_reference input) = 0;
		// Update parameters
		virtual void update(tensor_reference loss) = 0;
		// Back propagation
		virtual tensor_reference backward(tensor_reference loss) = 0;
	};

} // namespace nn

#endif
