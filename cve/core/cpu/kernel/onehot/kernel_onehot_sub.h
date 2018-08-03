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

#ifndef __CORE_CPU_KERNEL_ONEHOT_SUB_H__
#define __CORE_CPU_KERNEL_ONEHOT_SUB_H__

#include <cstring>
#include "../../cpu_inst.h"

namespace core
{
	// Function template kernel_onehot_sub

	template<class T1, class T2>
	void kernel_onehot_sub(size_t m, size_t n, T1 *a, const T2 *b)
	{
		for (size_t i = 0; i < m; ++i)
		{
			a[b[i]] -= T1(1);
			a += n;
		}
	}

	template<class T1, class T2>
	void kernel_onehot_sub(size_t l, size_t m, size_t n, T1 *a, const T2 *b)
	{
		for (size_t j = 0; j < l; ++j)
		{
			for (size_t i = 0; i < m; ++i)
			{
				a[b[i]] -= T1(1);
				a += n;
			}
			b += m;
		}
	}

} // namespace core

#endif
