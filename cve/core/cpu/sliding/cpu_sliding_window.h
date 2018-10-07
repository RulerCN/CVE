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

#ifndef __CORE_CPU_SLIDING_WINDOW_H__
#define __CORE_CPU_SLIDING_WINDOW_H__

#include "../../matrix.h"
#include "../../tensor.h"
#include "../kernel/sliding/kernel_sliding_window.h"

namespace core
{
	// Generate the mapping of sliding windows

	template<class T, class A>
	matrix<T, A>& cpu_sliding_window(matrix<T, A> &idx, const T rows, const T columns, const T dimension, const T window_h, const T window_w, const T stride_h = 1, const T stride_w = 1)
	{
		if (idx.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (window_h > rows || window_w > columns)
			throw ::std::invalid_argument(invalid_window_size);
		if (stride_h < 1 || stride_w < 1)
			throw ::std::invalid_argument(invalid_sliding_stride);
		if (idx.size() != ((rows - window_h) / stride_h + 1) * ((columns - window_w) / stride_w + 1) * window_h * window_w * dimension)
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sliding_window<cpu_avx2>(idx.data(), rows, columns, dimension, window_h, window_w, stride_h, stride_w);
		else if (cpu_inst::is_support_sse2())
			kernel_sliding_window<cpu_sse2>(idx.data(), rows, columns, dimension, window_h, window_w, stride_h, stride_w);
		else
			kernel_sliding_window<cpu_none>(idx.data(), rows, columns, dimension, window_h, window_w, stride_h, stride_w);
		return c;
	}

	template<class T, class U, class A1, class A2>
	matrix<T, A1>& cpu_sliding_window(matrix<T, A1> &idx, const matrix<U, A2> &a, const T window_h, const T window_w, const T stride_h, const T stride_w)
	{
		if (idx.empty() || a.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (window_h > a.rows() || window_w > a.columns())
			throw ::std::invalid_argument(invalid_window_size);
		if (stride_h < 1 || stride_w < 1)
			throw ::std::invalid_argument(invalid_sliding_stride);
		if (idx.size() != ((a.rows() - window_h) / stride_h + 1) * ((a.columns() - window_w) / stride_w + 1) * window_h * window_w * a.dimension())
			throw ::std::invalid_argument(invalid_size);

		if (cpu_inst::is_support_avx2())
			kernel_sliding_window<cpu_avx2>(idx.data(), a.rows(), a.columns(), a.dimension(), window_h, window_w, stride_h, stride_w);
		else if (cpu_inst::is_support_sse2())
			kernel_sliding_window<cpu_sse2>(idx.data(), a.rows(), a.columns(), a.dimension(), window_h, window_w, stride_h, stride_w);
		else
			kernel_sliding_window<cpu_none>(idx.data(), a.rows(), a.columns(), a.dimension(), window_h, window_w, stride_h, stride_w);
		return c;
	}

} // namespace core

#endif
