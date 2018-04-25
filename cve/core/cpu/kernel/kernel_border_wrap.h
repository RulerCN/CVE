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

#ifndef __CORE_CPU_KERNEL_BORDER_REPLICTE101_H__
#define __CORE_CPU_KERNEL_BORDER_REPLICTE101_H__

#include "../../definition.h"
#include "../../instruction.h"

namespace core
{
	// Function template kernel_border_wrap_left
	template<class T>
	void kernel_border_wrap_left(T *data, T columns, T width, T channels, T left)
	{
		T value;
		T stride = width * channels;
		T loop = left / width;
		T remain = (left % width) * channels;

		if (remain > 0)
		{
			value = stride - remain;
			for (T i = 0; i < remain; ++i)
				data[i] = value + i;
			data += remain;
		}
		while (loop > 1)
		{
			for (T i = 0; i < stride; ++i)
				data[i] = i;
			data += stride;
			--loop;
		}
	}

	// Function template kernel_border_wrap_center
	template<class T>
	void kernel_border_wrap_center(T *data, T columns, T width, T channels, T left)
	{
	}

	// Function template kernel_border_wrap_right
	template<class T>
	void kernel_border_wrap_right(T *data, T columns, T width, T channels, T right)
	{
	}

	// Function template kernel_border_wrap_top
	template<class T>
	void kernel_border_wrap_top(T *data, T rows, T columns, T height, T width, T channels, T top)
	{
	}

	// Function template kernel_border_wrap_middle
	template<class T>
	void kernel_border_wrap_middle(T *data, T rows, T columns, T height, T width, T channels, T top)
	{
	}

	// Function template kernel_border_wrap_bottom
	template<class T>
	void kernel_border_wrap_bottom(T *data, T rows, T columns, T height, T width, T channels, T bottom)
	{
	}

} // namespace core

#endif
