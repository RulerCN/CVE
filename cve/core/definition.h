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
	// CPU instruction set
	typedef unsigned int cpu_inst_type;
	static constexpr cpu_inst_type      cpu_none           =  0x00000000;
	static constexpr cpu_inst_type      cpu_mmx            =  0x00000001;                 /* MMX instruction set */
	static constexpr cpu_inst_type      cpu_sse            =  0x00000100;                 /* SSE instruction set */
	static constexpr cpu_inst_type      cpu_sse2           =  0x00000200;                 /* SSE2 instruction set */
	static constexpr cpu_inst_type      cpu_sse3           =  0x00000400;                 /* SSE3 instruction set */
	static constexpr cpu_inst_type      cpu_ssse3          =  0x00000800;                 /* SSE3S instruction set */
	static constexpr cpu_inst_type      cpu_sse41          =  0x00001000;                 /* SSE4.1 instruction set */
	static constexpr cpu_inst_type      cpu_sse42          =  0x00002000;                 /* SSE4.2 instruction set */
	static constexpr cpu_inst_type      cpu_avx            =  0x00010000;                 /* AVX instruction set */
	static constexpr cpu_inst_type      cpu_avx2           =  0x00020000;                 /* AVX2 instruction set */
	static constexpr cpu_inst_type      cpu_f16c           =  0x01000000;                 /* F16C instruction set */
	static constexpr cpu_inst_type      cpu_fma            =  0x02000000;                 /* FMA instruction set */
	static constexpr cpu_inst_type      cpu_fma4           =  0x04000000;                 /* FMA4 instruction set */
	static constexpr cpu_inst_type      cpu_xop            =  0x08000000;                 /* XOP instruction set */
	// Extremum
	static constexpr signed char        int8_min           = -0x7f - 1;                   /* -128 */
	static constexpr signed short       int16_min          = -0x7fff - 1;                 /* -32768 */
	static constexpr signed int         int32_min          = -0x7fffffff - 1;             /* -2147483648 */
	static constexpr signed long long   int64_min          = -0x7fffffffffffffffLL - 1;   /* -9223372036854775808LL */
	static constexpr signed char        int8_zero          =  0x00;                       /* 0 */
	static constexpr signed short       int16_zero         =  0x0000;                     /* 0 */
	static constexpr signed int         int32_zero         =  0x00000000;                 /* 0 */
	static constexpr signed long long   int64_zero         =  0x0000000000000000LL;       /* 0LL */
	static constexpr signed char        int8_max           =  0x7f;                       /* 127 */
	static constexpr signed short       int16_max          =  0x7fff;                     /* 32767 */
	static constexpr signed int         int32_max          =  0x7fffffff;                 /* 2147483647 */
	static constexpr signed long long   int64_max          =  0x7fffffffffffffffLL;       /* 9223372036854775807LL */
	static constexpr unsigned char      uint8_min          =  0x00U;                      /* 0U */
	static constexpr unsigned short     uint16_min         =  0x0000U;                    /* 0U */
	static constexpr unsigned int       uint32_min         =  0x00000000U;                /* 0U */
	static constexpr unsigned long long uint64_min         =  0x0000000000000000ULL;      /* 0ULL */
	static constexpr unsigned char      uint8_max          =  0xffU;                      /* 255U */
	static constexpr unsigned short     uint16_max         =  0xffffU;                    /* 65535U */
	static constexpr unsigned int       uint32_max         =  0xffffffffU;                /* 4294967295U */
	static constexpr unsigned long long uint64_max         =  0xffffffffffffffffULL;      /* 18446744073709551615ULL */
	// Logical constants
	static constexpr signed char        int8_false         =  0x00;                       /* 0 */
	static constexpr signed short       int16_false        =  0x0000;                     /* 0 */
	static constexpr signed int         int32_false        =  0x00000000;                 /* 0 */
	static constexpr signed long long   int64_false        =  0x0000000000000000LL;       /* 0LL */
	static constexpr unsigned char      uint8_false        =  0x00U;                      /* 0U */
	static constexpr unsigned short     uint16_false       =  0x0000U;                    /* 0U */
	static constexpr unsigned int       uint32_false       =  0x00000000U;                /* 0U */
	static constexpr unsigned long long uint64_false       =  0x0000000000000000ULL;      /* 0ULL */
	static constexpr signed char        int8_true          = -1;                          /* 0xff */
	static constexpr signed short       int16_true         = -1;                          /* 0xffff */
	static constexpr signed int         int32_true         = -1;                          /* 0xffffffff */
	static constexpr signed long long   int64_true         = -1LL;                        /* 0xffffffffffffffffLL */
	static constexpr unsigned char      uint8_true         =  0xffU;                      /* 255U */
	static constexpr unsigned short     uint16_true        =  0xffffU;                    /* 65535U */
	static constexpr unsigned int       uint32_true        =  0xffffffffU;                /* 4294967295U */
	static constexpr unsigned long long uint64_true        =  0xffffffffffffffffULL;      /* 18446744073709551615ULL */
	// Sign bit
	static constexpr signed char        int8_sign          = -0x7f - 1;                   /* 0x80 */
	static constexpr signed short       int16_sign         = -0x7fff - 1;                 /* 0x8000 */
	static constexpr signed int         int32_sign         = -0x7fffffff - 1;             /* 0x80000000 */
	static constexpr signed long long   int64_sign         = -0x7fffffffffffffffLL - 1;   /* 0x8000000000000000LL */
	// Absolute value
	static constexpr signed char        int8_abs           =  0x7f;                       /* 127 */
	static constexpr signed short       int16_abs          =  0x7fff;                     /* 32767 */
	static constexpr signed int         int32_abs          =  0x7fffffff;                 /* 2147483647 */
	static constexpr signed long long   int64_abs          =  0x7fffffffffffffffLL;       /* 9223372036854775807LL */
	// Floating point
	static constexpr signed int         flt_nan            =  0xffc00000;
	static constexpr signed int         flt_inf            =  0x7f800000;
	static constexpr signed int         flt_ninf           =  0xff800000;
	static constexpr signed int         flt_sign           =  0x80000000;
	static constexpr signed int         flt_exp_mask       =  0x7f800000;
	static constexpr signed int         flt_mant_mask      =  0x007fffff;
	static constexpr signed int         flt_base           =  0x0000007f;
	static constexpr signed int         flt_dig            =  0x00000006;
	static constexpr signed int         flt_mant_dig       =  0x00000018;
	static constexpr signed int         flt_min_10_exp     = -0x00000025;
	static constexpr signed int         flt_min_exp        = -0x0000007d;
	static constexpr signed int         flt_max_10_exp     =  0x00000026;
	static constexpr signed int         flt_max_exp        =  0x00000080;
	static constexpr float              flt_epsilon        =  1.1920928960000000e-07F;
	static constexpr float              flt_min            =  1.1754943510000000e-38F;
	static constexpr float              flt_max            =  3.4028234660000000e+38F;
	static constexpr signed long long   dbl_nan            =  0xfff8000000000000LL;
	static constexpr signed long long   dbl_inf            =  0x7ff0000000000000LL;
	static constexpr signed long long   dbl_ninf           =  0xfff0000000000000LL;
	static constexpr signed long long   dbl_sign           =  0x8000000000000000LL;
	static constexpr signed long long   dbl_exp_mask       =  0x7ff0000000000000LL;
	static constexpr signed long long   dbl_mant_mask      =  0x000fffffffffffffLL;
	static constexpr signed long long   dbl_base           =  0x00000000000003ffLL;
	static constexpr signed long long   dbl_dig            =  0x000000000000000fLL;
	static constexpr signed long long   dbl_mant_dig       =  0x0000000000000035LL;
	static constexpr signed long long   dbl_min_10_exp     = -0x0000000000000133LL;
	static constexpr signed long long   dbl_min_exp        = -0x00000000000003fdLL;
	static constexpr signed long long   dbl_max_10_exp     =  0x0000000000000134LL;
	static constexpr signed long long   dbl_max_exp        =  0x0000000000000400LL;
	static constexpr double             dbl_epsilon        =  2.2204460492503131e-016;
	static constexpr double             dbl_min            =  2.2250738585072014e-308;
	static constexpr double             dbl_max            =  1.7976931348623158e+308;
	// Mathematical constants
	static constexpr float              flt_zero           =  0.00000000000000000000e00F; /* 0 */
	static constexpr float              flt_half           =  5.00000000000000000000e-1F; /* 0.5 */
	static constexpr float              flt_one            =  1.00000000000000000000e00F; /* 1 */
	static constexpr float              flt_two            =  2.00000000000000000000e00F; /* 2 */
	static constexpr float              flt_e              =  2.71828182845904523536e00F; /* e */
	static constexpr float              flt_log2e          =  1.44269504088896340736e00F; /* log2(e) */
	static constexpr float              flt_log10e         =  4.34294481903251827651e-1F; /* log10(e) */
	static constexpr float              flt_ln2            =  6.93147180559945309417e-1F; /* ln(2) */
	static constexpr float              flt_ln2_hi         =  6.93359375000000000000e-1F; /* ln2 of 11 digit mantissa */
	static constexpr float              flt_ln2_lo         = -2.12194440000000000000e-4F; /* ln2 - ln2_hi */
	static constexpr float              flt_ln10           =  2.30258509299404568402e00F; /* ln(10) */
	static constexpr float              flt_pi             =  3.14159265358979323846e00F; /* pi */
	static constexpr float              flt_pi_2           =  1.57079632679489661923e00F; /* pi/2 */
	static constexpr float              flt_pi_4           =  7.85398163397448309616e-1F; /* pi/4 */
	static constexpr float              flt_1_pi           =  3.18309886183790671538e-1F; /* 1/pi */
	static constexpr float              flt_2_pi           =  6.36619772367581343076e-1F; /* 2/pi */
	static constexpr float              flt_2_sqrtpi       =  1.12837916709551257390e00F; /* 2/sqrt(pi) */
	static constexpr float              flt_sqrt2          =  1.41421356237309504880e00F; /* sqrt(2) */
	static constexpr float              flt_sqrt1_2        =  7.07106781186547524401e-1F; /* 1/sqrt(2) */
	static constexpr double             dbl_zero           =  0.00000000000000000000e00;  /* 0 */
	static constexpr double             dbl_half           =  5.00000000000000000000e-1;  /* 0.5 */
	static constexpr double             dbl_one            =  1.00000000000000000000e00;  /* 1 */
	static constexpr double             dbl_two            =  2.00000000000000000000e00;  /* 2 */
	static constexpr double             dbl_e              =  2.71828182845904523536e00;  /* e */
	static constexpr double             dbl_log2e          =  1.44269504088896340736e00;  /* log2(e) */
	static constexpr double             dbl_log10e         =  4.34294481903251827651e-1;  /* log10(e) */
	static constexpr double             dbl_ln2            =  6.93147180559945309417e-1;  /* ln(2) */
	static constexpr double             dbl_ln2_hi         =  6.93147182464599610000e-1;  /* ln2 of 20 digit mantissa */
	static constexpr double             dbl_ln2_lo         = -1.90465430058276790000e-9;  /* ln2 - ln2_hi */
	static constexpr double             dbl_ln10           =  2.30258509299404568402e00;  /* ln(10) */
	static constexpr double             dbl_pi             =  3.14159265358979323846e00;  /* pi */
	static constexpr double             dbl_pi_2           =  1.57079632679489661923e00;  /* pi/2 */
	static constexpr double             dbl_pi_4           =  7.85398163397448309616e-1;  /* pi/4 */
	static constexpr double             dbl_1_pi           =  3.18309886183790671538e-1;  /* 1/pi */
	static constexpr double             dbl_2_pi           =  6.36619772367581343076e-1;  /* 2/pi */
	static constexpr double             dbl_2_sqrtpi       =  1.12837916709551257390e00;  /* 2/sqrt(pi) */
	static constexpr double             dbl_sqrt2          =  1.41421356237309504880e00;  /* sqrt(2) */
	static constexpr double             dbl_sqrt1_2        =  7.07106781186547524401e-1;  /* 1/sqrt(2) */

//	static constexpr float              flt_rcp_1          = 1.00000000000000000000F;     /* 1/1 */
//	static constexpr float              flt_rcp_2          = 0.500000000000000000000F;    /* 1/2 */
//	static constexpr float              flt_rcp_3          = 0.333333333333333333333F;    /* 1/3 */
//	static constexpr float              flt_rcp_4          = 0.250000000000000000000F;    /* 1/4 */
//	static constexpr float              flt_rcp_5          = 0.200000000000000000000F;    /* 1/5 */
//	static constexpr float              flt_rcp_6          = 0.166666666666666666667F;    /* 1/6 */
//	static constexpr float              flt_rcp_7          = 0.142857142857142857143F;    /* 1/7 */
//	static constexpr float              flt_rcp_8          = 0.125000000000000000000F;    /* 1/8 */
//	static constexpr float              flt_rcp_9          = 0.111111111111111111111F;    /* 1/9 */
//	static constexpr float              flt_rcp_10         = 0.100000000000000000000F;    /* 1/10 */
//	static constexpr float              flt_rcp_11         = 9.09090909090909090909e-2F;  /* 1/11 */
//	static constexpr float              flt_rcp_12         = 8.33333333333333333333e-2F;  /* 1/12 */
//	static constexpr float              flt_rcp_13         = 7.69230769230769230769e-2F;  /* 1/13 */
//	static constexpr float              flt_rcp_fact1      = 1.00000000000000000000F;     /* 1/1! */
//	static constexpr float              flt_rcp_fact2      = 0.500000000000000000000F;    /* 1/2! */
//	static constexpr float              flt_rcp_fact3      = 0.166666666666666666667F;    /* 1/3! */
//	static constexpr float              flt_rcp_fact4      = 4.16666666666666666667e-2F;  /* 1/4! */
//	static constexpr float              flt_rcp_fact5      = 8.33333333333333333333e-3F;  /* 1/5! */
//	static constexpr float              flt_rcp_fact6      = 1.38888888888888888889e-3F;  /* 1/6! */
//	static constexpr float              flt_rcp_fact7      = 1.98412698412698412698e-4F;  /* 1/7! */
//	static constexpr float              flt_rcp_fact8      = 2.48015873015873015873e-5F;  /* 1/8! */
//	static constexpr float              flt_rcp_fact9      = 2.75573192239858906526e-6F;  /* 1/9! */
//	static constexpr float              flt_rcp_fact10     = 2.75573192239858906526e-7F;  /* 1/10! */
//	static constexpr float              flt_rcp_fact11     = 2.50521083854417187751e-8F;  /* 1/11! */
//	static constexpr float              flt_rcp_fact12     = 2.08767569878680989792e-9F;  /* 1/12! */
//	static constexpr float              flt_rcp_fact13     = 1.60590438368216145994e-10F; /* 1/13! */

//	static constexpr double             dbl_rcp_1          = 1.00000000000000000000;      /* 1/1 */
//	static constexpr double             dbl_rcp_2          = 0.500000000000000000000;     /* 1/2 */
//	static constexpr double             dbl_rcp_3          = 0.333333333333333333333;     /* 1/3 */
//	static constexpr double             dbl_rcp_4          = 0.250000000000000000000;     /* 1/4 */
//	static constexpr double             dbl_rcp_5          = 0.200000000000000000000;     /* 1/5 */
//	static constexpr double             dbl_rcp_6          = 0.166666666666666666667;     /* 1/6 */
//	static constexpr double             dbl_rcp_7          = 0.142857142857142857143;     /* 1/7 */
//	static constexpr double             dbl_rcp_8          = 0.125000000000000000000;     /* 1/8 */
//	static constexpr double             dbl_rcp_9          = 0.111111111111111111111;     /* 1/9 */
//	static constexpr double             dbl_rcp_10         = 0.100000000000000000000;     /* 1/10 */
//	static constexpr double             dbl_rcp_11         = 9.09090909090909090909e-2;   /* 1/11 */
//	static constexpr double             dbl_rcp_12         = 8.33333333333333333333e-2;   /* 1/12 */
//	static constexpr double             dbl_rcp_13         = 7.69230769230769230769e-2;   /* 1/13 */
//	static constexpr double             dbl_rcp_fact1      = 1.00000000000000000000;      /* 1/1! */
//	static constexpr double             dbl_rcp_fact2      = 0.500000000000000000000;     /* 1/2! */
//	static constexpr double             dbl_rcp_fact3      = 0.166666666666666666667;     /* 1/3! */
//	static constexpr double             dbl_rcp_fact4      = 4.16666666666666666667e-2;   /* 1/4! */
//	static constexpr double             dbl_rcp_fact5      = 8.33333333333333333333e-3;   /* 1/5! */
//	static constexpr double             dbl_rcp_fact6      = 1.38888888888888888889e-3;   /* 1/6! */
//	static constexpr double             dbl_rcp_fact7      = 1.98412698412698412698e-4;   /* 1/7! */
//	static constexpr double             dbl_rcp_fact8      = 2.48015873015873015873e-5;   /* 1/8! */
//	static constexpr double             dbl_rcp_fact9      = 2.75573192239858906526e-6;   /* 1/9! */
//	static constexpr double             dbl_rcp_fact10     = 2.75573192239858906526e-7;   /* 1/10! */
//	static constexpr double             dbl_rcp_fact11     = 2.50521083854417187751e-8;   /* 1/11! */
//	static constexpr double             dbl_rcp_fact12     = 2.08767569878680989792e-9;   /* 1/12! */
//	static constexpr double             dbl_rcp_fact13     = 1.60590438368216145994e-10;  /* 1/13! */
	// Copy mode
	typedef unsigned char copy_mode_type;
	static constexpr copy_mode_type     without_copy       = 0x00;                        /* no copy or reference to any data */
	static constexpr copy_mode_type     shallow_copy       = 0x01;                        /* only reference the original data */
	static constexpr copy_mode_type     deep_copy          = 0x02;                        /* copy all the original data */
	// Border type
	typedef unsigned char border_type;
	static constexpr border_type        border_constant    = 0x00;                        /* iiii|abcdefgh|iiii */
	static constexpr border_type        border_replicate   = 0x01;                        /* aaaa|abcdefgh|hhhh */
	static constexpr border_type        border_reflect     = 0x02;                        /* dcba|abcdefgh|hgfe */
	static constexpr border_type        border_reflect101  = 0x03;                        /* edcb|abcdefgh|gfed */
	static constexpr border_type        border_wrap        = 0x04;                        /* efgh|abcdefgh|abcd */
	// Axis type
	typedef unsigned char axis_type;
	static constexpr axis_type          axis_none          = 0x00;
	static constexpr axis_type          axis_x             = 0x01;                        /* x-axis */
	static constexpr axis_type          axis_y             = 0x02;                        /* y-axis */
	static constexpr axis_type          axis_z             = 0x04;                        /* z-axis */
	static constexpr axis_type          axis_xy            = axis_x | axis_y;             /* x-axis and y-axis */
	static constexpr axis_type          axis_yz            = axis_y | axis_z;             /* y-axis and z-axis */
	static constexpr axis_type          axis_xyz           = axis_x | axis_y | axis_z;    /* x-axis, y-axis and z-axis */

} // namespace core

#endif
