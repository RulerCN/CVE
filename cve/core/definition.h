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

#ifndef __CORE_DEFINITION_H__
#define __CORE_DEFINITION_H__

#include "error.h"

namespace core
{
	// Extremum
	static constexpr signed char        int8_min           = -0x7f - 1;                  // -128
	static constexpr signed short       int16_min          = -0x7fff - 1;                // -32768
	static constexpr signed int         int32_min          = -0x7fffffff - 1;            // -2147483648
	static constexpr signed long long   int64_min          = -0x7fffffffffffffffLL - 1;  // -9223372036854775808LL
	static constexpr signed char        int8_zero          = 0x00;                       // 0
	static constexpr signed short       int16_zero         = 0x0000;                     // 0
	static constexpr signed int         int32_zero         = 0x00000000;                 // 0
	static constexpr signed long long   int64_zero         = 0x0000000000000000LL;       // 0LL
	static constexpr signed char        int8_max           = 0x7f;                       // 127
	static constexpr signed short       int16_max          = 0x7fff;                     // 32767
	static constexpr signed int         int32_max          = 0x7fffffff;                 // 2147483647
	static constexpr signed long long   int64_max          = 0x7fffffffffffffffLL;       // 9223372036854775807LL
	static constexpr unsigned char      uint8_min          = 0x00U;                      // 0U
	static constexpr unsigned short     uint16_min         = 0x0000U;                    // 0U
	static constexpr unsigned int       uint32_min         = 0x00000000U;                // 0U
	static constexpr unsigned long long uint64_min         = 0x0000000000000000ULL;      // 0ULL
	static constexpr unsigned char      uint8_max          = 0xffU;                      // 255U
	static constexpr unsigned short     uint16_max         = 0xffffU;                    // 65535U
	static constexpr unsigned int       uint32_max         = 0xffffffffU;                // 4294967295U
	static constexpr unsigned long long uint64_max         = 0xffffffffffffffffULL;      // 18446744073709551615ULL
	// Logical constants
	static constexpr signed char        int8_false         = 0x00;                       // 0
	static constexpr signed short       int16_false        = 0x0000;                     // 0
	static constexpr signed int         int32_false        = 0x00000000;                 // 0
	static constexpr signed long long   int64_false        = 0x0000000000000000LL;       // 0LL
	static constexpr unsigned char      uint8_false        = 0x00U;                      // 0U
	static constexpr unsigned short     uint16_false       = 0x0000U;                    // 0U
	static constexpr unsigned int       uint32_false       = 0x00000000U;                // 0U
	static constexpr unsigned long long uint64_false       = 0x0000000000000000ULL;      // 0ULL
	static constexpr signed char        int8_true          = -1;                         // 0xff
	static constexpr signed short       int16_true         = -1;                         // 0xffff
	static constexpr signed int         int32_true         = -1;                         // 0xffffffff
	static constexpr signed long long   int64_true         = -1LL;                       // 0xffffffffffffffffLL
	static constexpr unsigned char      uint8_true         = 0xffU;                      // 255U
	static constexpr unsigned short     uint16_true        = 0xffffU;                    // 65535U
	static constexpr unsigned int       uint32_true        = 0xffffffffU;                // 4294967295U
	static constexpr unsigned long long uint64_true        = 0xffffffffffffffffULL;      // 18446744073709551615ULL
	// Sign bit
	static constexpr signed char        int8_sign          = -0x7f - 1;                  // 0x80
	static constexpr signed short       int16_sign         = -0x7fff - 1;                // 0x8000
	static constexpr signed int         int32_sign         = -0x7fffffff - 1;            // 0x80000000
	static constexpr signed long long   int64_sign         = -0x7fffffffffffffffLL - 1;  // 0x8000000000000000LL
	// Absolute value
	static constexpr signed char        int8_abs           = 0x7f;                       // 127
	static constexpr signed short       int16_abs          = 0x7fff;                     // 32767
	static constexpr signed int         int32_abs          = 0x7fffffff;                 // 2147483647
	static constexpr signed long long   int64_abs          = 0x7fffffffffffffffLL;       // 9223372036854775807LL
	// Floating point
	static constexpr int                flt_dig            = 6;
	static constexpr int                flt_mant_dig       = 24;
	static constexpr float              flt_epsilon        = 1.192092896e-07F;
	static constexpr float              flt_min            = 1.175494351e-38F;
	static constexpr int                flt_min_10_exp     = -37;
	static constexpr int                flt_min_exp        = -125;
	static constexpr float              flt_max            = 3.402823466e+38F;
	static constexpr int                flt_max_10_exp     = 38;
	static constexpr int                flt_max_exp        = 128;
	static constexpr int                dbl_dig            = 15;
	static constexpr int                dbl_mant_dig       = 53;
	static constexpr double             dbl_epsilon        = 2.2204460492503131e-016;
	static constexpr double             dbl_min            = 2.2250738585072014e-308;
	static constexpr int                dbl_min_10_exp     = -307;
	static constexpr int                dbl_min_exp        = -1021;
	static constexpr double             dbl_max            = 1.7976931348623158e+308;
	static constexpr int                dbl_max_10_exp     = 308;
	static constexpr int                dbl_max_exp        = 1024;
	// Mathematical constants
	static constexpr float              flt_e              = 2.71828182845904523536F;    // e
	static constexpr float              flt_log2e          = 1.44269504088896340736F;    // log2(e)
	static constexpr float              flt_log10e         = 0.434294481903251827651F;   // log10(e)
	static constexpr float              flt_ln2            = 0.693147180559945309417F;   // ln(2)
	static constexpr float              flt_ln10           = 2.30258509299404568402F;    // ln(10)
	static constexpr float              flt_pi             = 3.14159265358979323846F;    // pi
	static constexpr float              flt_pi_2           = 1.57079632679489661923F;    // pi/2
	static constexpr float              flt_pi_4           = 0.785398163397448309616F;   // pi/4
	static constexpr float              flt_1_pi           = 0.318309886183790671538F;   // 1/pi
	static constexpr float              flt_2_pi           = 0.636619772367581343076F;   // 2/pi
	static constexpr float              flt_2_sqrtpi       = 1.12837916709551257390F;    // 2/sqrt(pi)
	static constexpr float              flt_sqrt2          = 1.41421356237309504880F;    // sqrt(2)
	static constexpr float              flt_sqrt1_2        = 0.707106781186547524401F;   // 1/sqrt(2)
	static constexpr double             dbl_e              = 2.71828182845904523536;     // e
	static constexpr double             dbl_log2e          = 1.44269504088896340736;     // log2(e)
	static constexpr double             dbl_log10e         = 0.434294481903251827651;    // log10(e)
	static constexpr double             dbl_ln2            = 0.693147180559945309417;    // ln(2)
	static constexpr double             dbl_ln10           = 2.30258509299404568402;     // ln(10)
	static constexpr double             dbl_pi             = 3.14159265358979323846;     // pi
	static constexpr double             dbl_pi_2           = 1.57079632679489661923;     // pi/2
	static constexpr double             dbl_pi_4           = 0.785398163397448309616;    // pi/4
	static constexpr double             dbl_1_pi           = 0.318309886183790671538;    // 1/pi
	static constexpr double             dbl_2_pi           = 0.636619772367581343076;    // 2/pi
	static constexpr double             dbl_2_sqrtpi       = 1.12837916709551257390;     // 2/sqrt(pi)
	static constexpr double             dbl_sqrt2          = 1.41421356237309504880;     // sqrt(2)
	static constexpr double             dbl_sqrt1_2        = 0.707106781186547524401;    // 1/sqrt(2)

	// Instruction set
	typedef unsigned char inst_type;
	static constexpr inst_type          inst_none          = 0x00;
	static constexpr inst_type          inst_mmx           = 0x01;
	static constexpr inst_type          inst_sse           = 0x02;
	static constexpr inst_type          inst_sse2          = 0x03;
	static constexpr inst_type          inst_sse3          = 0x04;
	static constexpr inst_type          inst_ssse3         = 0x05;
	static constexpr inst_type          inst_sse41         = 0x06;
	static constexpr inst_type          inst_sse42         = 0x07;
	static constexpr inst_type          inst_avx           = 0x08;
	static constexpr inst_type          inst_avx2          = 0x09;
	static constexpr inst_type          inst_fma           = 0x10;
	static constexpr inst_type          inst_fma4          = 0x20;

} // namespace core

#endif
