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

#ifndef __CORE_CPU_KERNEL_BORDER_REPLICTE_H__
#define __CORE_CPU_KERNEL_BORDER_REPLICTE_H__

#include <cstring>
#include "../../definition.h"
#include "../../instruction.h"

namespace core
{
	// Function template kernel_border_replicte_left
	template<class T>
	void kernel_border_replicte_left(T *data, T /*columns*/, T /*width*/, T channels, T left)
	{
		T border = left * channels;

		for (T j = 0; j < channels; ++j)
			for (T i = j; i < border; i += channels)
				data[i] = j;
	}

	// Function template kernel_border_replicte_center
	template<class T>
	void kernel_border_replicte_center(T *data, T /*columns*/, T width, T channels, T left)
	{
		T delta = width * channels;
		T *dst = data + left * channels;

		for (T i = 0; i < delta; ++i)
			dst[i] = i;
	}

	// Function template kernel_border_replicte_right
	template<class T>
	void kernel_border_replicte_right(T *data, T columns, T width, T channels, T right)
	{
		T border = right * channels;
		T value = (width - 1) * channels;
		T *dst = data + (columns - right) * channels;

		for (T j = 0; j < channels; ++j)
		{
			for (T i = j; i < border; i += channels)
				dst[i] = value;
			value += 1;
		}
	}

	// Function template kernel_border_replicte_top
	template<class T>
	void kernel_border_replicte_top(T *data, T columns, T /*height*/, T /*width*/, T channels, T top)
	{
		T stride = columns * channels;
		T size = stride * sizeof(T);
		T *dst = data - top * stride;

		for (T i = 0; i < top; ++i)
		{
			::std::memcpy(dst, data, size);
			dst += stride;
		}
	}

	// Function template kernel_border_replicte_middle
	template<class T>
	void kernel_border_replicte_middle(T *data, T columns, T height, T width, T channels, T top)
	{
		T delta = width * channels;
		T stride = columns * channels;
		T value = delta;
		T *dst = data + stride;

		for (T j = 1; j < height; ++j)
		{
			for (T i = 0; i < stride; ++i)
				dst[i] = data[i] + value;
			dst += stride;
			value += delta;
		}
	}

	// Function template kernel_border_replicte_bottom
	template<class T>
	void kernel_border_replicte_bottom(T *data, T columns, T height, T /*width*/, T channels, T bottom)
	{
		T stride = columns * channels;
		T size = stride * sizeof(T);
		T *src = data + (height - 1) * stride;
		T *dst = src + stride;

		for (T i = 0; i < bottom; ++i)
		{
			::std::memcpy(dst, src, size);
			dst += stride;
		}
	}

} // namespace core

#endif
