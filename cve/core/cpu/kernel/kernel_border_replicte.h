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

#include "../../definition.h"
#include "../../instruction.h"

namespace core
{
	// Function template kernel_border_replicte_left
	template<class T>
	void kernel_border_replicte_left(T *data, T columns, T channels, T border)
	{
		T value;
		T loop = border / columns;
		T remain = (border % columns) * channels;
		T stride = columns * channels;

		if (remain > 0)
		{
			if (loop & 1)
			{
				value = stride - remain;
				for (T i = 0; i < remain; ++i)
					data[i] = value + i;
			}
			else
			{
				value = remain - channels;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < remain; i += channels)
						data[i] = value - i;
					value += 2;
				}
			}
			data += remain;
		}
		if (loop & 1)
		{
			value = stride - channels;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < stride; i += channels)
					data[i] = value - i;
				value += 2;
			}
			data += stride;
		}
		while (loop > 1)
		{
			for (T i = 0; i < stride; ++i)
				data[i] = i;
			data += stride;
			value = stride - channels;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < stride; i += channels)
					data[i] = value - i;
				value += 2;
			}
			data += stride;
			loop -= 2;
		}
	}

	// Function template kernel_border_replicte_right
	template<class T>
	void kernel_border_replicte_right(T *data, T columns, T channels, T border)
	{
		T value;
		T loop = border / columns;
		T remain = (border % columns) * channels;
		T stride = columns * channels;

		while (loop > 1)
		{
			value = stride - channels;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < stride; i += channels)
					data[i] = value - i;
				value += 2;
			}
			data += stride;
			for (T i = 0; i < stride; ++i)
				data[i] = i;
			data += stride;
			loop -= 2;
		}
		if (loop & 1)
		{
			value = stride - channels;
			for (T j = 0; j < channels; ++j)
			{
				for (T i = j; i < stride; i += channels)
					data[i] = value - i;
				value += 2;
			}
			data += stride;
		}
		if (remain > 0)
		{
			if (loop & 1)
			{
				for (T i = 0; i < remain; ++i)
					data[i] = i;
			}
			else
			{
				value = stride - channels;
				for (T j = 0; j < channels; ++j)
				{
					for (T i = j; i < remain; i += channels)
						data[i] = value - i;
					value += 2;
				}
			}
			data += remain;
		}
	}

	// Function template kernel_border_replicte_top
	template<class T>
	void kernel_border_replicte_top(T *data, T stride, T rows, T columns, T border)
	{
		T count;
		T loop = border / rows;
		T remain = border % rows;

		if (remain > 0)
		{
			if (loop & 1)
			{
				//count = rows - remain;
				//for (T i = 0; i < remain; ++i)
				//	data[i] = value + i;
			}
			else
			{
				//value = remain - 1;
				//for (T j = 0; j < channels; ++j)
				//{
				//	for (T i = j; i < remain; i += channels)
				//		data[i] = value - i;
				//	value += 2;
				//}
			}
		}
	}

} // namespace core

#endif
