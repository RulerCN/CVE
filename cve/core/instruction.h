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

#ifndef __CORE_INSTRUCTION_H__
#define __CORE_INSTRUCTION_H__

#include <omp.h>
// Microsoft Visual Studio
#if defined(_MSC_VER)
#include <intrin.h>
// GNU GCC/G++.
#elif defined(__GNUC__) || defined(__GNUG__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#define CPUIDFIELD_MASK_POS            0x0000001FU
#define CPUIDFIELD_MASK_LEN            0x000003E0U
#define CPUIDFIELD_MASK_REG            0x00000C00U
#define CPUIDFIELD_MASK_FIDSUB         0x000FF000U
#define CPUIDFIELD_MASK_FID            0xFFF00000U

#define CPUIDFIELD_SHIFT_POS           0
#define CPUIDFIELD_SHIFT_LEN           5
#define CPUIDFIELD_SHIFT_REG           10
#define CPUIDFIELD_SHIFT_FIDSUB        12
#define CPUIDFIELD_SHIFT_FID           20

#define CPUIDFIELD_MAKE(fid, fidsub, reg, pos, len) \
	(((fid) & 0xF0000000) | (((fid) << CPUIDFIELD_SHIFT_FID) & 0x0FF00000) | \
	(((fidsub) << CPUIDFIELD_SHIFT_FIDSUB) & CPUIDFIELD_MASK_FIDSUB) | \
	(((reg) << CPUIDFIELD_SHIFT_REG) & CPUIDFIELD_MASK_REG) | \
	(((pos) << CPUIDFIELD_SHIFT_POS) & CPUIDFIELD_MASK_POS) | \
	((((len) - 1) << CPUIDFIELD_SHIFT_LEN) & CPUIDFIELD_MASK_LEN))

#define CPUIDFIELD_FID(cpuidfield)     (((cpuidfield) & 0xF0000000) | (((cpuidfield) & 0x0FF00000) >> CPUIDFIELD_SHIFT_FID))
#define CPUIDFIELD_FIDSUB(cpuidfield)  (((cpuidfield) & CPUIDFIELD_MASK_FIDSUB) >> CPUIDFIELD_SHIFT_FIDSUB)
#define CPUIDFIELD_REG(cpuidfield)     (((cpuidfield) & CPUIDFIELD_MASK_REG) >> CPUIDFIELD_SHIFT_REG)
#define CPUIDFIELD_POS(cpuidfield)     (((cpuidfield) & CPUIDFIELD_MASK_POS) >> CPUIDFIELD_SHIFT_POS)
#define CPUIDFIELD_LEN(cpuidfield)     ((((cpuidfield) & CPUIDFIELD_MASK_LEN) >> CPUIDFIELD_SHIFT_LEN) + 1)
#define CPUID_GETBITS32(src, pos, len) (((src) >> (pos)) & (((unsigned int) -1) >> (32 - len)))

#define CPUF_LFuncStd                  CPUIDFIELD_MAKE(0, 0, 0, 0, 32)
#define CPUF_Stepping                  CPUIDFIELD_MAKE(1, 0, 0, 0, 4)
#define CPUF_BaseModel                 CPUIDFIELD_MAKE(1, 0, 0, 4, 4)
#define CPUF_BaseFamily                CPUIDFIELD_MAKE(1, 0, 0, 8, 4)
#define CPUF_ProcessorType             CPUIDFIELD_MAKE(1, 0, 0, 12, 2)
#define CPUF_ExtModel                  CPUIDFIELD_MAKE(1, 0, 0, 16, 4)
#define CPUF_ExtFamily                 CPUIDFIELD_MAKE(1, 0, 0, 20, 8)
#define CPUF_BrandId8                  CPUIDFIELD_MAKE(1, 0, 1, 0, 8)
#define CPUF_CLFlush                   CPUIDFIELD_MAKE(1, 0, 1, 8, 8)
#define CPUF_MaxApicId                 CPUIDFIELD_MAKE(1, 0, 1, 16, 8)
#define CPUF_ApicId                    CPUIDFIELD_MAKE(1, 0, 1, 24, 8)
#define CPUF_SSE3                      CPUIDFIELD_MAKE(1, 0, 2, 0, 1)
#define CPUF_PCLMULQDQ                 CPUIDFIELD_MAKE(1, 0, 2, 1, 1)
#define CPUF_DTES64                    CPUIDFIELD_MAKE(1, 0, 2, 2, 1)
#define CPUF_MONITOR                   CPUIDFIELD_MAKE(1, 0, 2, 3, 1)
#define CPUF_DS_CPL                    CPUIDFIELD_MAKE(1, 0, 2, 4, 1)
#define CPUF_VMX                       CPUIDFIELD_MAKE(1, 0, 2, 5, 1)
#define CPUF_SMX                       CPUIDFIELD_MAKE(1, 0, 2, 6, 1)
#define CPUF_EIST                      CPUIDFIELD_MAKE(1, 0, 2, 7, 1)
#define CPUF_TM2                       CPUIDFIELD_MAKE(1, 0, 2, 8, 1)
#define CPUF_SSSE3                     CPUIDFIELD_MAKE(1, 0, 2, 9, 1)
#define CPUF_CNXT_ID                   CPUIDFIELD_MAKE(1, 0, 2, 10, 1)
#define CPUF_FMA                       CPUIDFIELD_MAKE(1, 0, 2, 12, 1)
#define CPUF_CX16                      CPUIDFIELD_MAKE(1, 0, 2, 13, 1)
#define CPUF_xTPR                      CPUIDFIELD_MAKE(1, 0, 2, 14, 1)
#define CPUF_PDCM                      CPUIDFIELD_MAKE(1, 0, 2, 15, 1)
#define CPUF_PCID                      CPUIDFIELD_MAKE(1, 0, 2, 17, 1)
#define CPUF_DCA                       CPUIDFIELD_MAKE(1, 0, 2, 18, 1)
#define CPUF_SSE41                     CPUIDFIELD_MAKE(1, 0, 2, 19, 1)
#define CPUF_SSE42                     CPUIDFIELD_MAKE(1, 0, 2, 20, 1)
#define CPUF_x2APIC                    CPUIDFIELD_MAKE(1, 0, 2, 21, 1)
#define CPUF_MOVBE                     CPUIDFIELD_MAKE(1, 0, 2, 22, 1)
#define CPUF_POPCNT                    CPUIDFIELD_MAKE(1, 0, 2, 23, 1)
#define CPUF_TSC_DEADLINE              CPUIDFIELD_MAKE(1, 0, 2, 24, 1)
#define CPUF_AES                       CPUIDFIELD_MAKE(1, 0, 2, 25, 1)
#define CPUF_XSAVE                     CPUIDFIELD_MAKE(1, 0, 2, 26, 1)
#define CPUF_OSXSAVE                   CPUIDFIELD_MAKE(1, 0, 2, 27, 1)
#define CPUF_AVX                       CPUIDFIELD_MAKE(1, 0, 2, 28, 1)
#define CPUF_F16C                      CPUIDFIELD_MAKE(1, 0, 2, 29, 1)
#define CPUF_RDRAND                    CPUIDFIELD_MAKE(1, 0, 2, 30, 1)
#define CPUF_FPU                       CPUIDFIELD_MAKE(1, 0, 3, 0, 1)
#define CPUF_VME                       CPUIDFIELD_MAKE(1, 0, 3, 1, 1)
#define CPUF_DE                        CPUIDFIELD_MAKE(1, 0, 3, 2, 1)
#define CPUF_PSE                       CPUIDFIELD_MAKE(1, 0, 3, 3, 1)
#define CPUF_TSC                       CPUIDFIELD_MAKE(1, 0, 3, 4, 1)
#define CPUF_MSR                       CPUIDFIELD_MAKE(1, 0, 3, 5, 1)
#define CPUF_PAE                       CPUIDFIELD_MAKE(1, 0, 3, 6, 1)
#define CPUF_MCE                       CPUIDFIELD_MAKE(1, 0, 3, 7, 1)
#define CPUF_CX8                       CPUIDFIELD_MAKE(1, 0, 3, 8, 1)
#define CPUF_APIC                      CPUIDFIELD_MAKE(1, 0, 3, 9, 1)
#define CPUF_SEP                       CPUIDFIELD_MAKE(1, 0, 3, 11, 1)
#define CPUF_MTRR                      CPUIDFIELD_MAKE(1, 0, 3, 12, 1)
#define CPUF_PGE                       CPUIDFIELD_MAKE(1, 0, 3, 13, 1)
#define CPUF_MCA                       CPUIDFIELD_MAKE(1, 0, 3, 14, 1)
#define CPUF_CMOV                      CPUIDFIELD_MAKE(1, 0, 3, 15, 1)
#define CPUF_PAT                       CPUIDFIELD_MAKE(1, 0, 3, 16, 1)
#define CPUF_PSE36                     CPUIDFIELD_MAKE(1, 0, 3, 17, 1)
#define CPUF_PSN                       CPUIDFIELD_MAKE(1, 0, 3, 18, 1)
#define CPUF_CLFSH                     CPUIDFIELD_MAKE(1, 0, 3, 19, 1)
#define CPUF_DS                        CPUIDFIELD_MAKE(1, 0, 3, 21, 1)
#define CPUF_ACPI                      CPUIDFIELD_MAKE(1, 0, 3, 22, 1)
#define CPUF_MMX                       CPUIDFIELD_MAKE(1, 0, 3, 23, 1)
#define CPUF_FXSR                      CPUIDFIELD_MAKE(1, 0, 3, 24, 1)
#define CPUF_SSE                       CPUIDFIELD_MAKE(1, 0, 3, 25, 1)
#define CPUF_SSE2                      CPUIDFIELD_MAKE(1, 0, 3, 26, 1)
#define CPUF_SS                        CPUIDFIELD_MAKE(1, 0, 3, 27, 1)
#define CPUF_HTT                       CPUIDFIELD_MAKE(1, 0, 3, 28, 1)
#define CPUF_TM                        CPUIDFIELD_MAKE(1, 0, 3, 29, 1)
#define CPUF_PBE                       CPUIDFIELD_MAKE(1, 0, 3, 31, 1)
#define CPUF_Cache_Type                CPUIDFIELD_MAKE(4, 0, 0, 0, 5)
#define CPUF_Cache_Level               CPUIDFIELD_MAKE(4, 0, 0, 5, 3)
#define CPUF_CACHE_SI                  CPUIDFIELD_MAKE(4, 0, 0, 8, 1)
#define CPUF_CACHE_FA                  CPUIDFIELD_MAKE(4, 0, 0, 9, 1)
#define CPUF_MaxApicIdShare            CPUIDFIELD_MAKE(4, 0, 0, 14, 12)
#define CPUF_MaxApicIdCore             CPUIDFIELD_MAKE(4, 0, 0, 26, 6)
#define CPUF_Cache_LineSize            CPUIDFIELD_MAKE(4, 0, 1, 0, 12)
#define CPUF_Cache_Partitions          CPUIDFIELD_MAKE(4, 0, 1, 12, 10)
#define CPUF_Cache_Ways                CPUIDFIELD_MAKE(4, 0, 1, 22, 10)
#define CPUF_Cache_Sets                CPUIDFIELD_MAKE(4, 0, 2, 0, 32)
#define CPUF_CACHE_INVD                CPUIDFIELD_MAKE(4, 0, 3, 0, 1)
#define CPUF_CACHE_INCLUSIVENESS       CPUIDFIELD_MAKE(4, 0, 3, 1, 1)
#define CPUF_CACHE_COMPLEXINDEX        CPUIDFIELD_MAKE(4, 0, 3, 2, 1)
#define CPUF_MonLineSizeMin            CPUIDFIELD_MAKE(5, 0, 0, 0, 16)
#define CPUF_MonLineSizeMax            CPUIDFIELD_MAKE(5, 0, 1, 0, 16)
#define CPUF_EMX                       CPUIDFIELD_MAKE(5, 0, 2, 0, 1)
#define CPUF_IBE                       CPUIDFIELD_MAKE(5, 0, 2, 1, 1)
#define CPUF_MWAIT_Number_C0           CPUIDFIELD_MAKE(5, 0, 3, 0, 4)
#define CPUF_MWAIT_Number_C1           CPUIDFIELD_MAKE(5, 0, 3, 4, 4)
#define CPUF_MWAIT_Number_C2           CPUIDFIELD_MAKE(5, 0, 3, 8, 4)
#define CPUF_MWAIT_Number_C3           CPUIDFIELD_MAKE(5, 0, 3, 12, 4)
#define CPUF_MWAIT_Number_C4           CPUIDFIELD_MAKE(5, 0, 3, 16, 4)
#define CPUF_DTS                       CPUIDFIELD_MAKE(6, 0, 0, 0, 1)
#define CPUF_TURBO_BOOST               CPUIDFIELD_MAKE(6, 0, 0, 1, 1)
#define CPUF_ARAT                      CPUIDFIELD_MAKE(6, 0, 0, 2, 1)
#define CPUF_PLN                       CPUIDFIELD_MAKE(6, 0, 0, 4, 1)
#define CPUF_ECMD                      CPUIDFIELD_MAKE(6, 0, 0, 5, 1)
#define CPUF_PTM                       CPUIDFIELD_MAKE(6, 0, 0, 6, 1)
#define CPUF_DTS_ITs                   CPUIDFIELD_MAKE(6, 0, 1, 0, 4)
#define CPUF_PERF                      CPUIDFIELD_MAKE(6, 0, 2, 0, 1)
#define CPUF_ACNT2                     CPUIDFIELD_MAKE(6, 0, 2, 1, 1)
#define CPUF_ENERGY_PERF_BIAS          CPUIDFIELD_MAKE(6, 0, 2, 3, 1)
#define CPUF_Max07Subleaf              CPUIDFIELD_MAKE(7, 0, 0, 0, 32)
#define CPUF_FSGSBASE                  CPUIDFIELD_MAKE(7, 0, 1, 0, 1)
#define CPUF_TSC_ADJUST                CPUIDFIELD_MAKE(7, 0, 1, 1, 1)
#define CPUF_BMI1                      CPUIDFIELD_MAKE(7, 0, 1, 3, 1)
#define CPUF_HLE                       CPUIDFIELD_MAKE(7, 0, 1, 4, 1)
#define CPUF_AVX2                      CPUIDFIELD_MAKE(7, 0, 1, 5, 1)
#define CPUF_SMEP                      CPUIDFIELD_MAKE(7, 0, 1, 7, 1)
#define CPUF_BMI2                      CPUIDFIELD_MAKE(7, 0, 1, 8, 1)
#define CPUF_ERMS                      CPUIDFIELD_MAKE(7, 0, 1, 9, 1)
#define CPUF_INVPCID                   CPUIDFIELD_MAKE(7, 0, 1, 10, 1)
#define CPUF_RTM                       CPUIDFIELD_MAKE(7, 0, 1, 11, 1)
#define CPUF_RDSEED                    CPUIDFIELD_MAKE(7, 0, 1, 18, 1)
#define CPUF_ADX                       CPUIDFIELD_MAKE(7, 0, 1, 19, 1)
#define CPUF_SMAP                      CPUIDFIELD_MAKE(7, 0, 1, 20, 1)
#define CPUF_PLATFORM_DCA_CAP          CPUIDFIELD_MAKE(9, 0, 0, 0, 32)
#define CPUF_APM_Version               CPUIDFIELD_MAKE(0xA, 0, 0, 0, 8)
#define CPUF_APM_Counters              CPUIDFIELD_MAKE(0xA, 0, 0, 8, 8)
#define CPUF_APM_Bits                  CPUIDFIELD_MAKE(0xA, 0, 0, 16, 8)
#define CPUF_APM_Length                CPUIDFIELD_MAKE(0xA, 0, 0, 24, 8)
#define CPUF_APM_CC                    CPUIDFIELD_MAKE(0xA, 0, 1, 0, 1)
#define CPUF_APM_IR                    CPUIDFIELD_MAKE(0xA, 0, 1, 1, 1)
#define CPUF_APM_RC                    CPUIDFIELD_MAKE(0xA, 0, 1, 2, 1)
#define CPUF_APM_LLCR                  CPUIDFIELD_MAKE(0xA, 0, 1, 3, 1)
#define CPUF_APM_LLCM                  CPUIDFIELD_MAKE(0xA, 0, 1, 4, 1)
#define CPUF_APM_BIR                   CPUIDFIELD_MAKE(0xA, 0, 1, 5, 1)
#define CPUF_APM_BMR                   CPUIDFIELD_MAKE(0xA, 0, 1, 6, 1)
#define CPUF_APM_FC_Number             CPUIDFIELD_MAKE(0xA, 0, 3, 0, 5)
#define CPUF_APM_FC_Bits               CPUIDFIELD_MAKE(0xA, 0, 3, 5, 8)
#define CPUF_Topology_Bits             CPUIDFIELD_MAKE(0xB, 0, 0, 0, 5)
#define CPUF_Topology_Number           CPUIDFIELD_MAKE(0xB, 0, 1, 0, 16)
#define CPUF_Topology_Level            CPUIDFIELD_MAKE(0xB, 0, 2, 0, 8)
#define CPUF_Topology_Type             CPUIDFIELD_MAKE(0xB, 0, 2, 8, 8)
#define CPUF_X2APICID                  CPUIDFIELD_MAKE(0xB, 0, 3, 0, 32)
#define CPUF_XFeatureSupportedMaskLo   CPUIDFIELD_MAKE(0xD, 0, 0, 0, 32)
#define CPUF_XFeatureEnabledSizeMax    CPUIDFIELD_MAKE(0xD, 0, 1, 0, 32)
#define CPUF_XFeatureSupportedSizeMax  CPUIDFIELD_MAKE(0xD, 0, 2, 0, 32)
#define CPUF_XFeatureSupportedMaskHi   CPUIDFIELD_MAKE(0xD, 0, 3, 0, 32)
#define CPUF_XSAVEOPT                  CPUIDFIELD_MAKE(0xD, 1, 0, 0, 1)
#define CPUF_YmmSaveStateSize          CPUIDFIELD_MAKE(0xD, 2, 0, 0, 32)
#define CPUF_YmmSaveStateOffset        CPUIDFIELD_MAKE(0xD, 2, 1, 0, 32)
#define CPUF_LwpSaveStateSize          CPUIDFIELD_MAKE(0xD, 62, 0, 0, 32)
#define CPUF_LwpSaveStateOffset        CPUIDFIELD_MAKE(0xD, 62, 1, 0, 32)
#define CPUF_LFuncExt                  CPUIDFIELD_MAKE(0x80000000U, 0, 0, 0, 32)
#define CPUF_BrandId16                 CPUIDFIELD_MAKE(0x80000001U, 0, 1, 0, 16)
#define CPUF_PkgType                   CPUIDFIELD_MAKE(0x80000001U, 0, 1, 28, 4)
#define CPUF_LahfSahf                  CPUIDFIELD_MAKE(0x80000001U, 0, 2, 0, 1)
#define CPUF_CmpLegacy                 CPUIDFIELD_MAKE(0x80000001U, 0, 2, 1, 1)
#define CPUF_SVM                       CPUIDFIELD_MAKE(0x80000001U, 0, 2, 2, 1)
#define CPUF_ExtApicSpace              CPUIDFIELD_MAKE(0x80000001U, 0, 2, 3, 1)
#define CPUF_AltMovCr8                 CPUIDFIELD_MAKE(0x80000001U, 0, 2, 4, 1)
#define CPUF_ABM                       CPUIDFIELD_MAKE(0x80000001U, 0, 2, 5, 1)
#define CPUF_SSE4A                     CPUIDFIELD_MAKE(0x80000001U, 0, 2, 6, 1)
#define CPUF_MisAlignSse               CPUIDFIELD_MAKE(0x80000001U, 0, 2, 7, 1)
#define CPUF_3DNowPrefetch             CPUIDFIELD_MAKE(0x80000001U, 0, 2, 8, 1)
#define CPUF_OSVW                      CPUIDFIELD_MAKE(0x80000001U, 0, 2, 9, 1)
#define CPUF_IBS                       CPUIDFIELD_MAKE(0x80000001U, 0, 2, 10, 1)
#define CPUF_XOP                       CPUIDFIELD_MAKE(0x80000001U, 0, 2, 11, 1)
#define CPUF_SKINIT                    CPUIDFIELD_MAKE(0x80000001U, 0, 2, 12, 1)
#define CPUF_WDT                       CPUIDFIELD_MAKE(0x80000001U, 0, 2, 13, 1)
#define CPUF_LWP                       CPUIDFIELD_MAKE(0x80000001U, 0, 2, 15, 1)
#define CPUF_FMA4                      CPUIDFIELD_MAKE(0x80000001U, 0, 2, 16, 1)
#define CPUF_BIT_NODEID                CPUIDFIELD_MAKE(0x80000001U, 0, 2, 19, 1)
#define CPUF_TBM                       CPUIDFIELD_MAKE(0x80000001U, 0, 2, 21, 1)
#define CPUF_TopologyExtensions        CPUIDFIELD_MAKE(0x80000001U, 0, 2, 22, 1)
#define CPUF_SYSCALL                   CPUIDFIELD_MAKE(0x80000001U, 0, 3, 11, 1)
#define CPUF_XD                        CPUIDFIELD_MAKE(0x80000001U, 0, 3, 20, 1)
#define CPUF_MmxExt                    CPUIDFIELD_MAKE(0x80000001U, 0, 3, 22, 1)
#define CPUF_FFXSR                     CPUIDFIELD_MAKE(0x80000001U, 0, 3, 25, 1)
#define CPUF_Page1GB                   CPUIDFIELD_MAKE(0x80000001U, 0, 3, 26, 1)
#define CPUF_RDTSCP                    CPUIDFIELD_MAKE(0x80000001U, 0, 3, 27, 1)
#define CPUF_LM                        CPUIDFIELD_MAKE(0x80000001U, 0, 3, 29, 1)
#define CPUF_3DNowExt                  CPUIDFIELD_MAKE(0x80000001U, 0, 3, 30, 1)
#define CPUF_3DNow                     CPUIDFIELD_MAKE(0x80000001U, 0, 3, 31, 1)
#define CPUF_L1ITlb2and4MSize          CPUIDFIELD_MAKE(0x80000005U, 0, 0, 0, 8)
#define CPUF_L1ITlb2and4MAssoc         CPUIDFIELD_MAKE(0x80000005U, 0, 0, 8, 8)
#define CPUF_L1DTlb2and4MSize          CPUIDFIELD_MAKE(0x80000005U, 0, 0, 16, 8)
#define CPUF_L1DTlb2and4MAssoc         CPUIDFIELD_MAKE(0x80000005U, 0, 0, 24, 8)
#define CPUF_L1ITlb4KSize              CPUIDFIELD_MAKE(0x80000005U, 0, 1, 0, 8)
#define CPUF_L1ITlb4KAssoc             CPUIDFIELD_MAKE(0x80000005U, 0, 1, 8, 8)
#define CPUF_L1DTlb4KSize              CPUIDFIELD_MAKE(0x80000005U, 0, 1, 16, 8)
#define CPUF_L1DTlb4KAssoc             CPUIDFIELD_MAKE(0x80000005U, 0, 1, 24, 8)
#define CPUF_L1DcLineSize              CPUIDFIELD_MAKE(0x80000005U, 0, 2, 0, 8)
#define CPUF_L1DcLinesPerTag           CPUIDFIELD_MAKE(0x80000005U, 0, 2, 8, 8)
#define CPUF_L1DcAssoc                 CPUIDFIELD_MAKE(0x80000005U, 0, 2, 16, 8)
#define CPUF_L1DcSize                  CPUIDFIELD_MAKE(0x80000005U, 0, 2, 24, 8)
#define CPUF_L1IcLineSize              CPUIDFIELD_MAKE(0x80000005U, 0, 3, 0, 8)
#define CPUF_L1IcLinesPerTag           CPUIDFIELD_MAKE(0x80000005U, 0, 3, 8, 8)
#define CPUF_L1IcAssoc                 CPUIDFIELD_MAKE(0x80000005U, 0, 3, 16, 8)
#define CPUF_L1IcSize                  CPUIDFIELD_MAKE(0x80000005U, 0, 3, 24, 8)
#define CPUF_L2ITlb2and4MSize          CPUIDFIELD_MAKE(0x80000006U, 0, 0, 0, 12)
#define CPUF_L2ITlb2and4MAssoc         CPUIDFIELD_MAKE(0x80000006U, 0, 0, 12, 4)
#define CPUF_L2DTlb2and4MSize          CPUIDFIELD_MAKE(0x80000006U, 0, 0, 16, 12)
#define CPUF_L2DTlb2and4MAssoc         CPUIDFIELD_MAKE(0x80000006U, 0, 0, 28, 4)
#define CPUF_L2ITlb4KSize              CPUIDFIELD_MAKE(0x80000006U, 0, 1, 0, 12)
#define CPUF_L2ITlb4KAssoc             CPUIDFIELD_MAKE(0x80000006U, 0, 1, 12, 4)
#define CPUF_L2DTlb4KSize              CPUIDFIELD_MAKE(0x80000006U, 0, 1, 16, 12)
#define CPUF_L2DTlb4KAssoc             CPUIDFIELD_MAKE(0x80000006U, 0, 1, 28, 4)
#define CPUF_L2LineSize                CPUIDFIELD_MAKE(0x80000006U, 0, 2, 0, 8)
#define CPUF_L2LinesPerTag             CPUIDFIELD_MAKE(0x80000006U, 0, 2, 8, 4)
#define CPUF_L2Assoc                   CPUIDFIELD_MAKE(0x80000006U, 0, 2, 12, 4)
#define CPUF_L2Size                    CPUIDFIELD_MAKE(0x80000006U, 0, 2, 16, 16)
#define CPUF_L3LineSize                CPUIDFIELD_MAKE(0x80000006U, 0, 3, 0, 8)
#define CPUF_L3LinesPerTag             CPUIDFIELD_MAKE(0x80000006U, 0, 3, 8, 4)
#define CPUF_L3Assoc                   CPUIDFIELD_MAKE(0x80000006U, 0, 3, 12, 4)
#define CPUF_L3Size                    CPUIDFIELD_MAKE(0x80000006U, 0, 3, 18, 14)
#define CPUF_TS                        CPUIDFIELD_MAKE(0x80000007U, 0, 3, 0, 1)
#define CPUF_FID                       CPUIDFIELD_MAKE(0x80000007U, 0, 3, 1, 1)
#define CPUF_VID                       CPUIDFIELD_MAKE(0x80000007U, 0, 3, 2, 1)
#define CPUF_TTP                       CPUIDFIELD_MAKE(0x80000007U, 0, 3, 3, 1)
#define CPUF_HTC                       CPUIDFIELD_MAKE(0x80000007U, 0, 3, 4, 1)
#define CPUF_100MHzSteps               CPUIDFIELD_MAKE(0x80000007U, 0, 3, 6, 1)
#define CPUF_HwPstate                  CPUIDFIELD_MAKE(0x80000007U, 0, 3, 7, 1)
#define CPUF_TscInvariant              CPUIDFIELD_MAKE(0x80000007U, 0, 3, 8, 1)
#define CPUF_CPB                       CPUIDFIELD_MAKE(0x80000007U, 0, 3, 9, 1)
#define CPUF_EffFreqRO                 CPUIDFIELD_MAKE(0x80000007U, 0, 3, 10, 1)
#define CPUF_PhysAddrSize              CPUIDFIELD_MAKE(0x80000008U, 0, 0, 0, 8)
#define CPUF_LinAddrSize               CPUIDFIELD_MAKE(0x80000008U, 0, 0, 8, 8)
#define CPUF_GuestPhysAddrSize         CPUIDFIELD_MAKE(0x80000008U, 0, 0, 16, 8)
#define CPUF_NC                        CPUIDFIELD_MAKE(0x80000008U, 0, 2, 0, 8)
#define CPUF_ApicIdCoreIdSize          CPUIDFIELD_MAKE(0x80000008U, 0, 2, 12, 4)
#define CPUF_SvmRev                    CPUIDFIELD_MAKE(0x8000000AU, 0, 0, 0, 8)
#define CPUF_NASID                     CPUIDFIELD_MAKE(0x8000000AU, 0, 1, 0, 32)
#define CPUF_NP                        CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 0, 1)
#define CPUF_LbrVirt                   CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 1, 1)
#define CPUF_SVML                      CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 2, 1)
#define CPUF_NRIPS                     CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 3, 1)
#define CPUF_TscRateMsr                CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 4, 1)
#define CPUF_VmcbClean                 CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 5, 1)
#define CPUF_FlushByAsid               CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 6, 1)
#define CPUF_DecodeAssists             CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 7, 1)
#define CPUF_PauseFilter               CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 10, 1)
#define CPUF_PauseFilterThreshold      CPUIDFIELD_MAKE(0x8000000AU, 0, 3, 12, 1)
#define CPUF_L1ITlb1GSize              CPUIDFIELD_MAKE(0x80000019U, 0, 0, 0, 12)
#define CPUF_L1ITlb1GAssoc             CPUIDFIELD_MAKE(0x80000019U, 0, 0, 12, 4)
#define CPUF_L1DTlb1GSize              CPUIDFIELD_MAKE(0x80000019U, 0, 0, 16, 12)
#define CPUF_L1DTlb1GAssoc             CPUIDFIELD_MAKE(0x80000019U, 0, 0, 28, 4)
#define CPUF_L2ITlb1GSize              CPUIDFIELD_MAKE(0x80000019U, 0, 1, 0, 12)
#define CPUF_L2ITlb1GAssoc             CPUIDFIELD_MAKE(0x80000019U, 0, 1, 12, 4)
#define CPUF_L2DTlb1GSize              CPUIDFIELD_MAKE(0x80000019U, 0, 1, 16, 12)
#define CPUF_L2DTlb1GAssoc             CPUIDFIELD_MAKE(0x80000019U, 0, 1, 28, 4)
#define CPUF_FP128                     CPUIDFIELD_MAKE(0x8000001AU, 0, 0, 0, 1)
#define CPUF_MOVU                      CPUIDFIELD_MAKE(0x8000001AU, 0, 0, 1, 1)
#define CPUF_IBSFFV                    CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 0, 1)
#define CPUF_FetchSam                  CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 1, 1)
#define CPUF_OpSam                     CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 2, 1)
#define CPUF_RdWrOpCnt                 CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 3, 1)
#define CPUF_OpCnt                     CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 4, 1)
#define CPUF_BrnTrgt                   CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 5, 1)
#define CPUF_OpCntExt                  CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 6, 1)
#define CPUF_RipInvalidChk             CPUIDFIELD_MAKE(0x8000001BU, 0, 0, 7, 1)
#define CPUF_LwpAvail                  CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 0, 1)
#define CPUF_LwpVAL                    CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 1, 1)
#define CPUF_LwpIRE                    CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 2, 1)
#define CPUF_LwpBRE                    CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 3, 1)
#define CPUF_LwpDME                    CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 4, 1)
#define CPUF_LwpCNH                    CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 5, 1)
#define CPUF_LwpRNH                    CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 6, 1)
#define CPUF_LwpInt                    CPUIDFIELD_MAKE(0x8000001CU, 0, 0, 31, 1)
#define CPUF_LwpCbSize                 CPUIDFIELD_MAKE(0x8000001CU, 0, 1, 0, 8)
#define CPUF_LwpEventSize              CPUIDFIELD_MAKE(0x8000001CU, 0, 1, 8, 8)
#define CPUF_LwpMaxEvents              CPUIDFIELD_MAKE(0x8000001CU, 0, 1, 16, 8)
#define CPUF_LwpEventOffset            CPUIDFIELD_MAKE(0x8000001CU, 0, 1, 24, 8)
#define CPUF_LwpLatencyMax             CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 0, 5)
#define CPUF_LwpDataAddress            CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 5, 1)
#define CPUF_LwpLatencyRnd             CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 6, 3)
#define CPUF_LwpVersion                CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 9, 7)
#define CPUF_LwpMinBufferSize          CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 16, 8)
#define CPUF_LwpBranchPrediction       CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 28, 1)
#define CPUF_LwpIpFiltering            CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 29, 1)
#define CPUF_LwpCacheLevels            CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 30, 1)
#define CPUF_LwpCacheLatency           CPUIDFIELD_MAKE(0x8000001CU, 0, 2, 31, 1)
#define CPUF_D_LwpAvail                CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 0, 1)
#define CPUF_D_LwpVAL                  CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 1, 1)
#define CPUF_D_LwpIRE                  CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 2, 1)
#define CPUF_D_LwpBRE                  CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 3, 1)
#define CPUF_D_LwpDME                  CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 4, 1)
#define CPUF_D_LwpCNH                  CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 5, 1)
#define CPUF_D_LwpRNH                  CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 6, 1)
#define CPUF_D_LwpInt                  CPUIDFIELD_MAKE(0x8000001CU, 0, 3, 31, 1)
#define CPUF_CacheType                 CPUIDFIELD_MAKE(0x8000001DU, 0, 0, 0, 5)
#define CPUF_CacheLevel                CPUIDFIELD_MAKE(0x8000001DU, 0, 0, 5, 3)
#define CPUF_SelfInitialization        CPUIDFIELD_MAKE(0x8000001DU, 0, 0, 8, 1)
#define CPUF_FullyAssociative          CPUIDFIELD_MAKE(0x8000001DU, 0, 0, 9, 1)
#define CPUF_NumSharingCache           CPUIDFIELD_MAKE(0x8000001DU, 0, 0, 14, 12)
#define CPUF_CacheLineSize             CPUIDFIELD_MAKE(0x8000001DU, 0, 1, 0, 12)
#define CPUF_CachePhysPartitions       CPUIDFIELD_MAKE(0x8000001DU, 0, 1, 12, 10)
#define CPUF_CacheNumWays              CPUIDFIELD_MAKE(0x8000001DU, 0, 1, 22, 10)
#define CPUF_CacheNumSets              CPUIDFIELD_MAKE(0x8000001DU, 0, 2, 0, 32)
#define CPUF_WBINVD                    CPUIDFIELD_MAKE(0x8000001DU, 0, 3, 0, 1)
#define CPUF_CacheInclusive            CPUIDFIELD_MAKE(0x8000001DU, 0, 3, 1, 1)
#define CPUF_ExtendedApicId            CPUIDFIELD_MAKE(0x8000001EU, 0, 0, 0, 32)
#define CPUF_ComputeUnitId             CPUIDFIELD_MAKE(0x8000001EU, 0, 1, 0, 8)
#define CPUF_CoresPerComputeUnit       CPUIDFIELD_MAKE(0x8000001EU, 0, 1, 8, 2)
#define CPUF_NodeId                    CPUIDFIELD_MAKE(0x8000001EU, 0, 2, 0, 8)
#define CPUF_NodesPerProcessor         CPUIDFIELD_MAKE(0x8000001EU, 0, 2, 8, 3)

namespace core
{
	// Class global
	class global
	{
	public:
		// Enabling or disabling SIMD
		static int enable_simd(bool enable = true)
		{
			simd_inst = simd_none;
			if (enable)
			{
				simd_inst |= test_simd_mmx();
				simd_inst |= test_simd_sse();
			//	simd_inst |= test_simd_avx();
				simd_inst |= test_simd_fma();
			}
			return simd_inst;
		}
		// Enabling or disabling Multi-Processing
		static int enable_mp(int number = 0)
		{
			max_thread_num = omp_get_num_procs();
			// specify absolute number of threads
			if (number > 0)
				thread_num = number;
			// specify relative number of threads
			else if (max_thread_num + number > 0)
				thread_num = max_thread_num + number;
			else
				thread_num = 1;
			omp_set_num_threads(thread_num);
			return thread_num;
		}
		// Test MMX instruction
		static bool is_support_mmx(void)
		{
			return ((simd_inst & simd_mmx) == simd_mmx);
		}
		// Test SSE instruction
		static bool is_support_sse(void)
		{
			return ((simd_inst & simd_sse) == simd_sse);
		}
		// Test SSE2 instruction
		static bool is_support_sse2(void)
		{
			return ((simd_inst & simd_sse2) == simd_sse2);
		}
		// Test SSE3 instruction
		static bool is_support_sse3(void)
		{
			return ((simd_inst & simd_sse3) == simd_sse3);
		}
		// Test SSE3S instruction
		static bool is_support_ssse3(void)
		{
			return ((simd_inst & simd_ssse3) == simd_ssse3);
		}
		// Test SSE4.1 instruction
		static bool is_support_sse41(void)
		{
			return ((simd_inst & simd_sse41) == simd_sse41);
		}
		// Test SSE4.2 instruction
		static bool is_support_sse42(void)
		{
			return ((simd_inst & simd_sse42) == simd_sse42);
		}
		// Test AVX instruction
		static bool is_support_avx(void)
		{
			return ((simd_inst & simd_avx) == simd_avx);
		}
		// Test AVX2 instruction
		static bool is_support_avx2(void)
		{
			return ((simd_inst & simd_avx2) == simd_avx2);
		}
		// Test FMA instruction
		static bool is_support_fma(void)
		{
			return ((simd_inst & simd_fma) == simd_fma);
		}
		// Test FMA4 instruction
		static bool is_support_fma4(void)
		{
			return ((simd_inst & simd_fma4) == simd_fma4);
		}
		// Get the maximum number of threads
		static int get_max_threads(void)
		{
			return max_thread_num;
		}
		// Get the number of threads
		static int get_threads(void)
		{
			return thread_num;
		}
	private:
		// Get supported features and CPU type
		static void get_cpuid(int info[4], int type)
		{
#		if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
			__cpuid(info, type);
#		elif (defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)))
			__cpuid(type, info[0], info[1], info[2], info[3]);
#		else
			info[0] = info[1] = info[2] = info[3] = 0;
#		endif
		}
		// Get supported features and CPU type
		static void get_cpuidex(int info[4], int type, int value)
		{
#		if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
			__cpuidex(info, type, value);
#		elif (defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)))
			__cpuid_count(type, value, info[0], info[1], info[2], info[3]);
#		else
			info[0] = info[1] = info[2] = info[3] = 0;
#		endif
		}
		// Get CPUID field
		static unsigned int get_cpuid_field(unsigned int cpu_field)
		{
			int info[4] = { 0 };
			get_cpuidex(info, CPUIDFIELD_FID(cpu_field), CPUIDFIELD_FIDSUB(cpu_field));
			return CPUID_GETBITS32(info[CPUIDFIELD_REG(cpu_field)], CPUIDFIELD_POS(cpu_field), CPUIDFIELD_LEN(cpu_field));
		}
		// Test MMX instruction set
		static int test_simd_mmx(void)
		{
			int rst = simd_none;
			int info[4] = { 0 };
			get_cpuid(info, 1);
			if (info[3] & edx_mmx)
				rst |= simd_mmx;
			if (rst > simd_none)
			{
#			if defined(_WIN64)
				rst = simd_none;
#			else
				try
				{
					_mm_empty();
				}
				catch (...)
				{
					rst = simd_none;
				}
#			endif
			}
			return rst;
		}
		// Test SSE instruction set
		static int test_simd_sse(void)
		{
			int rst = simd_none;
			int info[4] = { 0 };
			get_cpuid(info, 1);
			if (info[3] & edx_sse)
				rst |= simd_sse;
			if (info[3] & edx_sse2)
				rst |= simd_sse2;
			if (info[2] & ecx_sse3)
				rst |= simd_sse3;
			if (info[2] & ecx_ssse3)
				rst |= simd_ssse3;
			if (info[2] & ecx_sse41)
				rst |= simd_sse41;
			if (info[2] & ecx_sse42)
				rst |= simd_sse42;
			if (rst > simd_none)
			{
				try
				{
					__m128 m = _mm_setzero_ps();
					int* p = reinterpret_cast<int*>(&m);
					if (*p != 0)
						rst = simd_none;
				}
				catch (...)
				{
					rst = simd_none;
				}
			}
			return rst;
		}
		// Test AVX instruction set
		static int test_simd_avx(void)
		{
			int rst = simd_none;
			if (get_cpuid_field(CPUF_AVX))
				rst |= simd_avx;
			if (get_cpuid_field(CPUF_AVX2))
				rst |= simd_avx2;
			if (rst > simd_none)
			{
				if (get_cpuid_field(CPUF_OSXSAVE))
				{
					if ((get_cpuid_field(CPUF_XFeatureSupportedMaskLo) & 6) != 6)
						rst = simd_none;
				}
			}
			return rst;
		}
		// Test FMA instruction set
		static int test_simd_fma(void)
		{
			int rst = simd_none;
			if (get_cpuid_field(CPUF_FMA))
				rst |= simd_fma;
			if (get_cpuid_field(CPUF_FMA4))
				rst |= simd_fma4;
			return rst;
		}
	private:
		static constexpr int edx_mmx    = 0x00800000;
		static constexpr int edx_sse    = 0x02000000;
		static constexpr int edx_sse2   = 0x04000000;
		static constexpr int ecx_sse3   = 0x00000001;
		static constexpr int ecx_ssse3  = 0x00000200;
		static constexpr int ecx_sse41  = 0x00080000;
		static constexpr int ecx_sse42  = 0x00100000;

		static constexpr int simd_none  = 0x00000000;
		static constexpr int simd_mmx   = 0x00000001;
		static constexpr int simd_sse   = 0x00000100;
		static constexpr int simd_sse2  = 0x00000200;
		static constexpr int simd_sse3  = 0x00000400;
		static constexpr int simd_ssse3 = 0x00000800;
		static constexpr int simd_sse41 = 0x00001000;
		static constexpr int simd_sse42 = 0x00002000;
		static constexpr int simd_avx   = 0x00010000;
		static constexpr int simd_avx2  = 0x00020000;
		static constexpr int simd_f16c  = 0x01000000;
		static constexpr int simd_fma   = 0x02000000;
		static constexpr int simd_fma4  = 0x04000000;
		static constexpr int simd_xop   = 0x08000000;

		static int           simd_inst;
		static int           thread_num;
		static int           max_thread_num;
	};

	// Initialize static variables
	int global::simd_inst      = simd_none;
	int global::max_thread_num = 1;
	int global::thread_num     = 1;

} // namespace core

#endif
