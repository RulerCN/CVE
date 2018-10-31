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

#ifndef __CORE_CPU_KERNEL_REPEAT_H__
#define __CORE_CPU_KERNEL_REPEAT_H__

#include <cstring>

namespace core
{
	// Function template kernel_repeat

	template<class T>
	void kernel_repeat(size_t n, const T *a, size_t len, T *b)
	{
		const size_t size = len * sizeof(T);

		for (size_t i = 0; i < n; ++i)
		{
			::std::memcpy(b, a, size);
			b += len;
		}
	}

	template<class T>
	void kernel_repeat(size_t m, size_t n, const T *a, size_t rows, size_t rsa, T *b)
	{
		size_t size;
		size_t stride;
		T *ptr_b = b;

		size = rsa * sizeof(T);
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < n; ++j)
			{
				::std::memcpy(ptr_b, a, size);
				ptr_b += rsa;
			}
			a += rsa;
		}
		stride = rows * n * rsa;
		size = stride * sizeof(T);
		ptr_b = b + stride;
		for (size_t i = 1; i < m; ++i)
		{
			::std::memcpy(ptr_b, b, size);
			ptr_b += stride;
		}
	}

	template<class T>
	void kernel_repeat(size_t l, size_t m, size_t n, const T *a, size_t batch, size_t rows, size_t rsa, T *b)
	{
		size_t size;
		size_t stride;
		T *ptr_b = b;

		for (size_t i = 0; i < m; ++i)
		{
			size = rsa * sizeof(T);
			for (size_t j = 0; j < rows; ++j)
			{
				for (size_t k = 0; k < n; ++k)
				{
					::std::memcpy(ptr_b, a, size);
					ptr_b += rsa;
				}
				a += rsa;
			}
			stride = rows * n * rsa;
			size = stride * sizeof(T);
			ptr_b = b + stride;
			for (size_t i = 1; i < m; ++i)
			{
				::std::memcpy(ptr_b, b, size);
				ptr_b += stride;
			}
		}
		stride = m * rows * n * rsa;
		size = stride * sizeof(T);
		ptr_b = b + stride;
		for (size_t i = 1; i < l; ++i)
		{
			::std::memcpy(ptr_b, b, size);
			ptr_b += stride;
		}
	}

} // namespace core

#endif
