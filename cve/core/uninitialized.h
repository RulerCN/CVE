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

#ifndef __CORE_UNINITIALIZED_H__
#define __CORE_UNINITIALIZED_H__

#include <memory>

namespace core
{
	template<class InputIt, class ForwardIt>
	inline ForwardIt uninitialized_move(InputIt first, InputIt last, ForwardIt d_first)
	{
		typedef typename ::std::iterator_traits<ForwardIt>::value_type Value;
		ForwardIt current = d_first;
		try
		{
			for (; first != last; ++first, (void) ++current)
				::new (static_cast<void*>(::std::addressof(*current))) Value(::std::move(*first));
			return current;
		}
		catch (...)
		{
			for (; d_first != current; ++d_first)
				d_first->~Value();
			throw;
		}
	}

	template<class InputIt, class Size, class ForwardIt>
	inline ::std::pair<InputIt, ForwardIt> uninitialized_move_n(InputIt first, Size count, ForwardIt d_first)
	{
		typedef typename ::std::iterator_traits<ForwardIt>::value_type Value;
		ForwardIt current = d_first;
		try
		{
			for (; count > 0; ++first, (void) ++current, --count)
				::new (static_cast<void*>(::std::addressof(*current))) Value(*first);
		}
		catch (...)
		{
			for (; d_first != current; ++d_first)
				d_first->~Value();
			throw;
		}
		return{ first, current };
	}

	template<class ForwardIt>
	inline void uninitialized_default_construct(ForwardIt first, ForwardIt last)
	{
		typedef typename ::std::iterator_traits<ForwardIt>::value_type Value;
		ForwardIt current = first;
		try
		{
			for (; current != last; ++current)
				::new (static_cast<void*>(::std::addressof(*current))) Value;
		}
		catch (...)
		{
			for (; first != current; ++first)
				first->~Value();
			throw;
		}
	}
	template<class ForwardIt, class Size>
	ForwardIt uninitialized_default_construct_n(ForwardIt first, Size n)
	{
		typedef typename ::std::iterator_traits<ForwardIt>::value_type Value;
		ForwardIt current = first;
		try
		{
			for (; n > 0; (void) ++current, --n)
				::new (static_cast<void*>(::std::addressof(*current))) Value;
			return current;
		}
		catch (...)
		{
			for (; first != current; ++first)
				first->~Value();
			throw;
		}
	}

	template<class ForwardIt>
	inline void uninitialized_value_construct(ForwardIt first, ForwardIt last)
	{
		typedef typename ::std::iterator_traits<ForwardIt>::value_type Value;
		ForwardIt current = first;
		try
		{
			for (; current != last; ++current)
				::new (static_cast<void*>(::std::addressof(*current))) Value();
		}
		catch (...)
		{
			for (; first != current; ++first)
				first->~Value();
			throw;
		}
	}

	template<class ForwardIt, class Size>
	ForwardIt uninitialized_value_construct_n(ForwardIt first, Size n)
	{
		typedef typename ::std::iterator_traits<ForwardIt>::value_type Value;
		ForwardIt current = first;
		try
		{
			for (; n > 0; (void) ++current, --n)
				::new (static_cast<void*>(::std::addressof(*current))) Value();
			return current;
		}
		catch (...)
		{
			for (; first != current; ++first)
				first->~Value();
			throw;
		}
	}

	template<class T>
	void destroy_at(T* p)
	{
		p->~T();
	}

	template<class ForwardIt>
	inline void destroy(ForwardIt first, ForwardIt last)
	{
		for (; first != last; ++first)
			destroy_at(::std::addressof(*first));
	}

	template<class ForwardIt, class Size>
	inline ForwardIt destroy_n(ForwardIt first, Size n)
	{
		for (; n > 0; (void) ++first, --n)
			destroy_at(::std::addressof(*first));
		return first;
	}

} // namespace core

#endif
