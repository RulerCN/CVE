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

#include "../matrix.h"
#include "kernel/kernel_sliding_window.h"

namespace core
{
	//template <class T, class A>
	//matrix<T, A>& cpu_sliding_window(matrix<T, A> &index, T rows, T columns, T channels, T window_h, T window_w, T stride_h = 1, T stride_w = 1)
	//{
	//	if (index.empty())
	//		throw ::std::invalid_argument(matrix_not_initialized);
	//	if (rows <= 0 || columns <= 0 || channels <= 0)
	//		throw ::std::invalid_argument(invalid_matrix_size);
	//	if (rows <= window_h || columns <= window_w)
	//		throw ::std::invalid_argument(invalid_window_size);
	//	if (stride_h <= 0 || stride_w <= 0)
	//		throw ::std::invalid_argument(invalid_sliding_stride);

	//	kernel_sliding_window(index.data(), rows, columns, channels, window_h, window_w, stride_h, stride_w);
	//	return index;
	//}

	template <class A>
	matrix<unsigned int, A>& cpu_sliding_window(matrix<unsigned int, A> &index, const unsigned int rows, const unsigned int columns, const unsigned int channels,
		const unsigned int window_h, const unsigned int window_w, const unsigned int stride_h = 1, const unsigned int stride_w = 1)
	{
		if (index.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (rows <= 0 || columns <= 0 || channels <= 0)
			throw ::std::invalid_argument(invalid_matrix_size);
		if (rows <= window_h || columns <= window_w)
			throw ::std::invalid_argument(invalid_window_size);
		if (stride_h <= 0 || stride_w <= 0)
			throw ::std::invalid_argument(invalid_sliding_stride);

		if (cpu::is_support_avx2())
			kernel_sliding_window<unsigned int, cpu_avx2>(index.data(), rows, columns, channels, window_h, window_w, stride_h, stride_w);
		else if (cpu::is_support_sse2())
			kernel_sliding_window<unsigned int, cpu_sse2>(index.data(), rows, columns, channels, window_h, window_w, stride_h, stride_w);
		else
			kernel_sliding_window<unsigned int, cpu_none>(index.data(), rows, columns, channels, window_h, window_w, stride_h, stride_w);
		return index;
	}

	template <class A>
	matrix<unsigned __int64, A>& cpu_sliding_window(matrix<unsigned __int64, A> &index, const unsigned __int64 rows, const unsigned __int64 columns, const unsigned __int64 channels,
		const unsigned __int64 window_h, const unsigned __int64 window_w, const unsigned __int64 stride_h = 1, const unsigned __int64 stride_w = 1)
	{
		if (index.empty())
			throw ::std::invalid_argument(matrix_not_initialized);
		if (rows <= 0 || columns <= 0 || channels <= 0)
			throw ::std::invalid_argument(invalid_matrix_size);
		if (rows <= window_h || columns <= window_w)
			throw ::std::invalid_argument(invalid_window_size);
		if (stride_h <= 0 || stride_w <= 0)
			throw ::std::invalid_argument(invalid_sliding_stride);

		if (cpu::is_support_avx2())
			kernel_sliding_window<unsigned __int64, cpu_avx2>(index.data(), rows, columns, channels, window_h, window_w, stride_h, stride_w);
		else if (cpu::is_support_sse2())
			kernel_sliding_window<unsigned __int64, cpu_sse2>(index.data(), rows, columns, channels, window_h, window_w, stride_h, stride_w);
		else
			kernel_sliding_window<unsigned __int64, cpu_none>(index.data(), rows, columns, channels, window_h, window_w, stride_h, stride_w);
		return index;
	}

} // namespace core

#endif
