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

#include "../../definition.h"
#include "../../instruction.h"

namespace core
{
	// Function template kernel_border_reflect101_left
	template<class T>
	void kernel_border_reflect101_left(T *data, T columns, T width, T channels, T left)
	{
		T value;
		T repeat = (width - 1) * 2;
		T stride = (width - 1) * channels;
		T loop = left / repeat;
		T remain = (left % repeat) * channels;

		if (remain > 0)
		{
			if (remain <= stride)
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
				remain -= stride;
				value = stride - remain;
				for (T i = 0; i < remain; ++i)
					data[i] = value + i;
				data += remain;
				value = stride;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < stride; i += channels)
						data[i] = value - i;
					value += 2;
				}
				data += stride;
			}
		}
		while (loop > 1)
		{
			for (T i = 0; i < stride; ++i)
				data[i] = i;
			data += stride;
			value = stride;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < stride; i += channels)
					data[i] = value - i;
				value += 2;
			}
			data += stride;
			--loop;
		}
	}

	// Function template kernel_border_reflect101_center
	template<class T>
	void kernel_border_reflect101_center(T *data, T /*columns*/, T width, T channels, T left)
	{
		T stride = width * channels;
		T *dst = data + left * channels;

		for (T i = 0; i < stride; ++i)
			dst[i] = i;
	}

	// Function template kernel_border_reflect101_right
	template<class T>
	void kernel_border_reflect101_right(T *data, T columns, T width, T channels, T right)
	{
		T value;
		T repeat = (width - 1) * 2;
		T stride = (width - 1) * channels;
		T loop = right / repeat;
		T remain = (right % repeat) * channels;
		T *dst = data + (columns - right) * channels;

		while (loop > 1)
		{
			value = stride - channels;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < stride; i += channels)
					dst[i] = value - i;
				value += 2;
			}
			dst += stride;
			for (T i = 0; i < stride; ++i)
				dst[i] = i;
			dst += stride;
			--loop;
		}
		if (remain > 0)
		{
			if (remain <= stride)
			{
				value = stride - channels;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < remain; i += channels)
						dst[i] = value - i;
					value += 2;
				}
			}
			else
			{
				remain -= stride;
				value = stride - channels;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < stride; i += channels)
						dst[i] = value - i;
					value += 2;
				}
				dst += stride;
				for (T i = 0; i < remain; ++i)
					dst[i] = i;
			}
		}
	}

	// Function template kernel_border_reflect101_top
	template<class T>
	void kernel_border_reflect101_top(T *data, T rows, T columns, T height, T width, T channels, T top)
	{
		T value;
		T stride = width * channels;
		T length = columns * channels;
		T loop = top / height;
		T remain = top % height;
		T *dst = data - top * columns * channels;

		if (remain > 0)
		{
			if (remain < height)
			{
				value = (height - remain) * stride;
				for (T j = 0; j < remain; ++j)
				{
					for (T i = 0; i < length; ++i)
						dst[i] = data[i] + value;
					dst += length;
					value += stride;
				}
			}
			else
			{

			}
		}
		while (loop > 1)
		{
			--loop;
		}
	}

	// Function template kernel_border_reflect101_middle
	template<class T>
	void kernel_border_reflect101_middle(T *data, T /*rows*/, T columns, T height, T width, T channels, T top)
	{
		T stride = width * channels;
		T length = columns * channels;
		T value = stride;
		T *dst = data + length;

		for (T j = 1; j < height; ++j)
		{
			for (T i = 0; i < length; ++i)
				dst[i] = data[i] + value;
			dst += length;
			value += stride;
		}
	}

	// Function template kernel_border_reflect101_bottom
	template<class T>
	void kernel_border_reflect101_bottom(T *data, T rows, T columns, T height, T width, T channels, T bottom)
	{
	}

} // namespace core

#endif
