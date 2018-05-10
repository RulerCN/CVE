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

#ifndef __CORE_CPU_KERNEL_SLIDING_WINDOW_H__
#define __CORE_CPU_KERNEL_SLIDING_WINDOW_H__

#include "../cpu.h"

namespace core
{
	// Function template kernel_sliding_window
	template<class T>
	void kernel_sliding_window(T *data, T rows, T columns, T channels, T window_h, T window_w, T stride_h, T stride_w)
	{
		T value, *src, *dst;
		T step_h = (rows - window_h) / stride_h;
		T step_w = (columns - window_w) / stride_w;
		T stride = columns * channels;
		T window_area = window_h * window_w;
		T window_size = channels * window_area;
		T sliding_size = (step_w + 1) * window_size;

		// 1 * window_w
		value = 0;
		for (T i = 0; i < window_w; ++i)
		{
			data[i] = value;
			value += channels;
		}
		// window_h * window_w
		src = data;
		dst = src + window_w;
		for (T i = 1; i < window_h; ++i)
		{
			for (T j = 0; j < window_w; ++j)
				dst[j] = src[j] + stride;
			src += window_w;
			dst += window_w;
		}
		// channels * window_h * window_w
		src = data;
		dst = src + window_area;
		for (T i = 1; i < channels; ++i)
		{
			for (T j = 0; j < window_area; ++j)
				dst[j] = src[j] + 1;
			src += window_area;
			dst += window_area;
		}
		// horizontal sliding window
		src = data;
		dst = src + window_size;
		value = stride_w * channels;
		for (T i = 0; i < step_w; ++i)
		{
			for (T j = 0; j < window_size; ++j)
				dst[j] = src[j] + value;
			src += window_size;
			dst += window_size;
		}
		// vertical sliding window
		src = data;
		dst = src + sliding_size;
		value = stride_h * stride;
		for (T i = 0; i < step_h; ++i)
		{
			for (T j = 0; j < sliding_size; ++j)
				dst[j] = src[j] + value;
			src += sliding_size;
			dst += sliding_size;
		}
	}

} // namespace core

#endif
