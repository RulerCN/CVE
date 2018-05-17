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

#ifndef __CORE_CPU_KERNEL_BORDER_REFLECT101_H__
#define __CORE_CPU_KERNEL_BORDER_REFLECT101_H__

#include "../cpu_inst.h"

namespace core
{
	// Function template kernel_border_reflect101_left
	template<class T>
	void kernel_border_reflect101_left(T *data, T columns, T width, T channels, T left)
	{
		T value;
		T repeat = (width - 1) * 2;
		T loop = left / repeat;
		T remain = (left % repeat) * channels;
		T delta = (width - 1) * channels;

		if (remain > 0)
		{
			if (remain <= delta)
			{
				value = remain;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < remain; i += channels)
						data[i] = value - i;
					value += 2;
				}
				data += remain;
			}
			else
			{
				remain -= delta;
				value = delta - remain;
				for (T i = 0; i < remain; ++i)
					data[i] = value + i;
				data += remain;
				value = delta;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < delta; i += channels)
						data[i] = value - i;
					value += 2;
				}
				data += delta;
			}
		}
		while (loop > 0)
		{
			for (T i = 0; i < delta; ++i)
				data[i] = i;
			data += delta;
			value = delta;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < delta; i += channels)
					data[i] = value - i;
				value += 2;
			}
			data += delta;
			--loop;
		}
	}

	// Function template kernel_border_reflect101_center
	template<class T>
	void kernel_border_reflect101_center(T *data, T /*columns*/, T width, T channels, T left)
	{
		T delta = width * channels;
		T *dst = data + left * channels;

		for (T i = 0; i < delta; ++i)
			dst[i] = i;
	}
	
	// Function template kernel_border_reflect101_right
	template<class T>
	void kernel_border_reflect101_right(T *data, T columns, T width, T channels, T right)
	{
		T value;
		T repeat = (width - 1) * 2;
		T loop = right / repeat;
		T remain = (right % repeat) * channels;
		T delta = (width - 1) * channels;
		T *dst = data + (columns - right) * channels;

		while (loop > 0)
		{
			value = delta - channels;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < delta; i += channels)
					dst[i] = value - i;
				value += 2;
			}
			dst += delta - channels;
			for (T i = channels; i <= delta; ++i)
				dst[i] = i;
			dst += delta + channels;
			--loop;
		}
		if (remain > 0)
		{
			if (remain <= delta)
			{
				value = delta - channels;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < remain; i += channels)
						dst[i] = value - i;
					value += 2;
				}
			}
			else
			{
				value = delta - channels;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < delta; i += channels)
						dst[i] = value - i;
					value += 2;
				}
				remain -= delta - channels;
				dst += delta - channels;
				for (T i = channels; i < remain; ++i)
					dst[i] = i;
			}
		}
	}

	// Function template kernel_border_reflect101_top
	template<class T>
	void kernel_border_reflect101_top(T *data, T columns, T height, T width, T channels, T top)
	{
		T value;
		T repeat = (height - 1) * 2;
		T loop = top / repeat;
		T remain = top % repeat;
		T delta = width * channels;
		T stride = columns * channels;
		T *dst = data - top * stride;

		if (remain > 0)
		{
			if (remain < height)
			{
				value = remain * delta;
				for (T j = 0; j < remain; ++j)
				{
					for (T i = 0; i < stride; ++i)
						dst[i] = data[i] + value;
					dst += stride;
					value -= delta;
				}
			}
			else
			{
				value = (repeat - remain) * delta;
				for (T j = height - 1; j < remain; ++j)
				{
					for (T i = 0; i < stride; ++i)
						dst[i] = data[i] + value;
					dst += stride;
					value += delta;
				}
				value = (height - 1) * delta;
				for (T j = 1; j < height; ++j)
				{
					for (T i = 0; i < stride; ++i)
						dst[i] = data[i] + value;
					dst += stride;
					value -= delta;
				}
			}
		}
		while (loop > 0)
		{
			value = 0;
			for (T j = 1; j < height; ++j)
			{
				for (T i = 0; i < stride; ++i)
					dst[i] = data[i] + value;
				dst += stride;
				value += delta;
			}
			value = (height - 1) * delta;
			for (T j = 1; j < height; ++j)
			{
				for (T i = 0; i < stride; ++i)
					dst[i] = data[i] + value;
				dst += stride;
				value -= delta;
			}
			--loop;
		}
	}

	// Function template kernel_border_reflect101_middle
	template<class T>
	void kernel_border_reflect101_middle(T *data, T columns, T height, T width, T channels, T /*top*/)
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

	// Function template kernel_border_reflect101_bottom
	template<class T>
	void kernel_border_reflect101_bottom(T *data, T columns, T height, T width, T channels, T bottom)
	{
		T value;
		T repeat = (height - 1) * 2;
		T loop = bottom / repeat;
		T remain = bottom % repeat;
		T delta = width * channels;
		T stride = columns * channels;
		T *dst = data + height * stride;

		while (loop > 0)
		{
			value = (height - 2) * delta;
			for (T j = 1; j < height; ++j)
			{
				for (T i = 0; i < stride; ++i)
					dst[i] = data[i] + value;
				dst += stride;
				value -= delta;
			}
			value = delta;
			for (T j = 1; j < height; ++j)
			{
				for (T i = 0; i < stride; ++i)
					dst[i] = data[i] + value;
				dst += stride;
				value += delta;
			}
			--loop;
		}
		if (remain > 0)
		{
			if (remain < height)
			{
				value = (height - 2) * delta;
				for (T j = 0; j < remain; ++j)
				{
					for (T i = 0; i < stride; ++i)
						dst[i] = data[i] + value;
					dst += stride;
					value -= delta;
				}
			}
			else
			{
				value = (height - 2) * delta;
				for (T j = 1; j < height; ++j)
				{
					for (T i = 0; i < stride; ++i)
						dst[i] = data[i] + value;
					dst += stride;
					value -= delta;
				}
				value = delta;
				for (T j = height - 1; j < remain; ++j)
				{
					for (T i = 0; i < stride; ++i)
						dst[i] = data[i] + value;
					dst += stride;
					value += delta;
				}
			}
		}
	}

} // namespace core

#endif
