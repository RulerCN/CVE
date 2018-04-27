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

#ifndef __CORE_CPU_BORDER_H__
#define __CORE_CPU_BORDER_H__

#include "../matrix.h"
#include "kernel/kernel_border_replicte.h"
#include "kernel/kernel_border_reflect.h"
#include "kernel/kernel_border_reflect101.h"
#include "kernel/kernel_border_wrap.h"

namespace core
{
	// Create a border around the image
	// Parameters:
	// 1. index - output matrix.
	// 2. left - left border width in number of pixels.
	// 3. top - top border width in number of pixels.
	// 4. right - right border width in number of pixels.
	// 5. bottom - bottom border width in number of pixels.
	// 6. type - border type:
	//     border_replicte:   Last element is replicated throughout, like this: aaaaaa|abcdefgh|hhhhhhh
	//     border_reflect:    Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
	//     border_reflect101: Same as above, but with a slight change, like this : gfedcb|abcdefgh|gfedcba
	//     border_wrap:       Can't explain, it will look like this : cdefgh|abcdefgh|abcdefg
	template <class T, class A>
	matrix<T, A>& cpu_border(matrix<T, A> &index, T left, T top, T right, T bottom, border_type type)
	{
		switch (type)
		{
		case border_replicte:
			return cpu_border_replicte(index, left, top, right, bottom);
		case border_reflect:
			return cpu_border_reflect(index, left, top, right, bottom);
		case border_reflect101:
			return cpu_border_reflect101(index, left, top, right, bottom);
		case border_wrap:
			return cpu_border_wrap(index, left, top, right, bottom);
		default:
			throw ::std::invalid_argument(invalid_mode_parameters);
		}
	}

	// Function template cpu_border_replicte
	template <class T, class A>
	matrix<T, A>& cpu_border_replicte(matrix<T, A> &index, T left, T top, T right, T bottom)
	{
		if (index.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (index.rows() <= top + bottom || index.columns() <= left + right)
			throw ::std::invalid_argument(invalid_border_size);
		T *data = index.data(top);
		T height = index.rows() - top - bottom;
		T width = index.columns() - left - right;

		// left border
		if (left > 0)
			kernel_border_replicte_left(data, index.columns(), width, index.dimension(), left);
		// center data
		kernel_border_replicte_center(data, index.columns(), width, index.dimension(), left);
		// right border
		if (right > 0)
			kernel_border_replicte_right(data, index.columns(), width, index.dimension(), right);
		// top border
		if (top > 0)
			kernel_border_replicte_top(data, index.columns(), height, width, index.dimension(), top);
		// middle data
		kernel_border_replicte_middle(data, index.columns(), height, width, index.dimension(), top);
		// bottom border
		if (bottom > 0)
			kernel_border_replicte_bottom(data, index.columns(), height, width, index.dimension(), bottom);
		return index;
	}

	// Function template cpu_border_reflect
	template <class T, class A>
	matrix<T, A>& cpu_border_reflect(matrix<T, A> &index, T left, T top, T right, T bottom)
	{
		if (index.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (index.rows() <= top + bottom || index.columns() <= left + right)
			throw ::std::invalid_argument(invalid_border_size);
		T *data = index.data(top);
		T height = index.rows() - top - bottom;
		T width = index.columns() - left - right;

		// left border
		if (left > 0)
			kernel_border_reflect_left(data, index.columns(), width, index.dimension(), left);
		// center data
		kernel_border_reflect_center(data, index.columns(), width, index.dimension(), left);
		// right border
		if (right > 0)
			kernel_border_reflect_right(data, index.columns(), width, index.dimension(), right);
		// top border
		if (top > 0)
			kernel_border_reflect_top(data, index.columns(), height, width, index.dimension(), top);
		// middle data
		kernel_border_reflect_middle(data, index.columns(), height, width, index.dimension(), top);
		// bottom border
		if (bottom > 0)
			kernel_border_reflect_bottom(data, index.columns(), height, width, index.dimension(), bottom);
		return index;
	}

	// Function template cpu_border_reflect101
	template <class T, class A>
	matrix<T, A>& cpu_border_reflect101(matrix<T, A> &index, T left, T top, T right, T bottom)
	{
		if (index.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (index.rows() <= top + bottom || index.columns() <= left + right)
			throw ::std::invalid_argument(invalid_border_size);
		T *data = index.data(top);
		T height = index.rows() - top - bottom;
		T width = index.columns() - left - right;

		// left border
		if (left > 0)
			kernel_border_reflect101_left(data, index.columns(), width, index.dimension(), left);
		// center data
		kernel_border_reflect101_center(data, index.columns(), width, index.dimension(), left);
		// right border
		if (right > 0)
			kernel_border_reflect101_right(data, index.columns(), width, index.dimension(), right);
		// top border
		if (top > 0)
			kernel_border_reflect101_top(data, index.columns(), height, width, index.dimension(), top);
		// middle data
		kernel_border_reflect101_middle(data, index.columns(), height, width, index.dimension(), top);
		// bottom border
		if (bottom > 0)
			kernel_border_reflect101_bottom(data, index.columns(), height, width, index.dimension(), bottom);
		return index;
	}

	// Function template cpu_border_wrap
	template <class T, class A>
	matrix<T, A>& cpu_border_wrap(matrix<T, A> &index, T left, T top, T right, T bottom)
	{
		if (index.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (index.rows() <= top + bottom || index.columns() <= left + right)
			throw ::std::invalid_argument(invalid_border_size);
		T *data = index.data(top);
		T height = index.rows() - top - bottom;
		T width = index.columns() - left - right;

		// left border
		if (left > 0)
			kernel_border_wrap_left(data, index.columns(), width, index.dimension(), left);
		// center data
		kernel_border_wrap_center(data, index.columns(), width, index.dimension(), left);
		// right border
		if (right > 0)
			kernel_border_wrap_right(data, index.columns(), width, index.dimension(), right);
		// top border
		if (top > 0)
			kernel_border_wrap_top(data, index.columns(), height, width, index.dimension(), top);
		// middle data
		kernel_border_wrap_middle(data, index.columns(), height, width, index.dimension(), top);
		// bottom border
		if (bottom > 0)
			kernel_border_wrap_bottom(data, index.columns(), height, width, index.dimension(), bottom);
		return index;
	}

} // namespace core

#endif
