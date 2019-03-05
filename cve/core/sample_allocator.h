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

#ifndef __CORE_SAMPLE_ALLOCATOR_H__
#define __CORE_SAMPLE_ALLOCATOR_H__

#include <memory>
#include <exception>

namespace core
{
	template <class T> class sample_allocator;

	// Specialize for void
	template <>
	class sample_allocator<void>
	{
	public:
		using value_type = void;
		using pointer = void*;
		using const_pointer = const void*;
		using size_type = size_t;
		using difference_type = ptrdiff_t;

		template <class U>
		struct rebind
		{
			using other = sample_allocator<U>;
		};
	};

	// Class template sample_allocator
	template <class T>
	class sample_allocator
	{
	public:
		using value_type      = T;
		using pointer         = value_type*;
		using const_pointer   = const value_type*;
		using reference       = value_type&;
		using const_reference = const value_type&;
		using size_type       = size_t;
		using difference_type = ptrdiff_t;

		template <class U>
		struct rebind
		{
			using other = sample_allocator<U>;
		};

		using propagate_on_container_move_assignment = ::std::true_type;
		using propagate_on_container_move_assignment = ::std::true_type;
		using propagate_on_container_swap            = ::std::true_type;
		using is_always_equal                        = ::std::true_type;
	public:
		// Construct default sample_allocator
		sample_allocator(void) noexcept
		{
		}
		// Construct by copying
		sample_allocator(const sample_allocator&) noexcept
		{
		}
		// Construct by moving
		sample_allocator(const sample_allocator&&) noexcept
		{
		}
		// Construct from a related sample_allocator
		template<class U>
		sample_allocator(const sample_allocator<U>&) noexcept
		{
		}
		// Return address of mutable x
		pointer address(reference x) const noexcept
		{
			return (::std::addressof(x));
		}
		// Return address of nonmutable x
		const_pointer address(const_reference x) const noexcept
		{
			return (::std::addressof(x));
		}
		// Allocate array of n elements
		pointer allocate(size_type n, typename sample_allocator<void>::const_pointer hint = nullptr)
		{
			return (static_cast<pointer>(malloc(n * sizeof(value_type))));
		}
		// Deallocate object at p
		void deallocate(pointer p, size_type n)
		{
			free(p);
		}
		// Estimate maximum array size
		size_type max_size(void) const noexcept
		{
			return (static_cast<size_t>(-1) / sizeof(value_type));
		}
		// Construct U(Args...) at p
		template<class U, class... Args>
		void construct(U* p, Args&&... args)
		{
			::new ((void*)p) U(::std::forward<Args>(args)...);
		}
		// Destroy object at p
		template <class U>
		void destroy(U* p)
		{
			p->~U();
		}
	};

	template <class T1, class T2>
	inline bool operator==(const sample_allocator<T1>&, const sample_allocator<T2>&)
	{
		return true;
	}
	template <class T1, class T2>
	inline bool operator!=(const sample_allocator<T1>&, const sample_allocator<T2>&)
	{
		return false;
	}

} // namespace core

#endif
