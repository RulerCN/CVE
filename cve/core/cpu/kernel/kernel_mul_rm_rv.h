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

#ifndef __CORE_CPU_KERNEL_MUL_RM_RV_H__
#define __CORE_CPU_KERNEL_MUL_RM_RV_H__

#include "../cpu_inst.h"

namespace core
{
	// Class template common_mul_rm_rv
	template<class T>
	struct common_mul_rv_rm
	{
		// C(mx1) += A(mxp) * B(px1)
		void operator()(size_t m, size_t p, const T *a, size_t rsa, const T *b, T *c, size_t rsc) const
		{
			for (size_t i = 0; i < m; ++i)
			{
				for (size_t k = 0; k < p; ++k)
				{

				}
			}

			T val_a;
			for (size_t k = 0; k < p; ++k)
			{
				val_a = a[k];
				for (size_t j = 0; j < n; ++j)
					c[j] += val_a * b[j];
				b += rsb;
			}
		}
	};

} // namespace core

#endif
