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

#ifndef __CORE_CPU_KERNEL_SLIDING_NEIGHBOUR_H__
#define __CORE_CPU_KERNEL_SLIDING_NEIGHBOUR_H__

#include "block_sliding.h"

namespace core
{
	// Function template kernel_sliding_neighbour
	template<class T, cpu_inst_type inst>
	void kernel_sliding_neighbour(T *idx, const T rows, const T columns, const T dimension, const T window_h, const T window_w, const T stride_h, const T stride_w)
	{
		const T stride = columns * dimension;
		const T window_area = window_h * window_w;
		const T window_size = dimension * window_area;
		const T number_h = (rows - window_h) / stride_h + 1;
		const T number_w = (columns - window_w) / stride_w + 1;
		const struct block_sliding<T, inst> functor;

		// 1 * window_w
		T val = T(0);
		for (T i = 0; i < window_w; ++i)
		{
			idx[i] = val;
			val += dimension;
		}
		// window_h * window_w
		functor(window_h, window_w, idx, stride);
		// dimension * window_h * window_w
		functor(dimension, window_area, idx, 1);
		// horizontal sliding window 
		functor(number_w, window_size, idx, stride_w * dimension);
		// vertical sliding window
		functor(number_h, number_w * window_size, idx, stride_h * stride);
	};

} // namespace core

#endif
