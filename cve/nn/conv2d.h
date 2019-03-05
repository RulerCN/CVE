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

#ifndef __NN_CONV2D_H__
#define __NN_CONV2D_H__

#include "layer.h"

namespace nn
{
	// Class template conv2d
	template <class T, class Allocator = ::core::allocator<T> >
	class conv2d : public layer<T, Allocator>
	{
	public:
		// types:

		using layer_type            = layer<T, Allocator>;
		using allocator_type        = typename layer_type::allocator_type;
		using value_type            = typename layer_type::value_type;
		using pointer               = typename layer_type::pointer;
		using const_pointer         = typename layer_type::const_pointer;
		using reference             = typename layer_type::reference;
		using const_reference       = typename layer_type::const_reference;
		using size_type             = typename layer_type::size_type;
		using difference_type       = typename layer_type::difference_type;

		using scalar_type           = typename layer_type::scalar_type;
		using vector_type           = typename layer_type::vector_type;
		using matrix_type           = typename layer_type::matrix_type;
		using tensor_type           = typename layer_type::tensor_type;

		// construct/copy/destroy:

		conv2d(const Allocator& alloc = Allocator())
		{}

		conv2d(size_type in_dim, size_type out_dim)
		{
			assign(in_dim, out_dim);
		}

		void assign(size_type in_dim, size_type out_dim)
		{
			weight.assign(in_dim, out_dim, 1);
			bias.assign(out_dim, 1);
			delta_w.assign(in_dim, out_dim, 1);
			delta_b.assign(out_dim, 1);
		}
	private:
		matrix_type weight;
		vector_type bias;
		matrix_type delta_w;
		vector_type delta_b;
	};

} // namespace nn

#endif
