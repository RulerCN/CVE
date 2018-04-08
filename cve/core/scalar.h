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

#ifndef __CORE_SCALAR_H__
#define __CORE_SCALAR_H__

#include "definition.h"
#include "allocator.h"
#include "uninitialized.h"

namespace core
{
	template <class T, class Allocator> class scalar;

	// Specialize for void
	template <class Allocator>
	class scalar<void, Allocator>
	{
	public:
		// types:

		typedef Allocator                                       allocator_type;
		typedef ::std::allocator_traits<allocator_type>         allocator_traits_type;
		typedef typename allocator_traits_type::value_type      value_type;
		typedef typename allocator_traits_type::pointer         pointer;
		typedef typename allocator_traits_type::const_pointer   const_pointer;
		typedef typename allocator_traits_type::size_type       size_type;
		typedef typename allocator_traits_type::difference_type difference_type;

		template <class U> struct rebind
		{
			typedef scalar<U, Allocator> other;
		};
	};

	// Template class scalar
	template <class T, class Allocator = allocator<T> >
	class scalar : public Allocator
	{
	public:
		// types:

		typedef Allocator                                       allocator_type;
		typedef ::std::allocator_traits<allocator_type>         allocator_traits_type;
		typedef typename allocator_traits_type::value_type      value_type;
		typedef typename allocator_traits_type::pointer         pointer;
		typedef typename allocator_traits_type::const_pointer   const_pointer;
		typedef typename allocator_type::reference              reference;
		typedef typename allocator_type::const_reference        const_reference;
		typedef typename allocator_traits_type::size_type       size_type;
		typedef typename allocator_traits_type::difference_type difference_type;

		template <class U> struct rebind
		{
			typedef scalar<U, Allocator> other;
		};

		// construct/copy/destroy:

		explicit scalar(const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{}
		explicit scalar(size_type n)
			: Allocator()
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(n);
		}
		scalar(size_type n, pointer p, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			create(n, p);
		}

		scalar(size_type n, const value_type& value, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(n, value);
		}
		template <class InputIterator>
		scalar(InputIterator first, InputIterator last, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(last, last);
		}

		template <class U, class A>
		scalar(const_pointer source, const scalar<U, A>& maping)
			: Allocator(A::rebind<value_type>::other())
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(maping.size());
			remap(source, maping);
		}

		scalar(const scalar<T, Allocator>& x)
			: Allocator(x.get_allocator())
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(x);
		}
		scalar(scalar<T, Allocator>&& x)
			: Allocator(x.get_allocator())
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<scalar<T, Allocator> >(x));
		}
		scalar(const scalar<T, Allocator>& x, const Allocator& alloc)
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(x);
		}
		scalar(scalar<T, Allocator>&& x, const Allocator& alloc)
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<scalar<T, Allocator> >(x));
		}
		scalar(::std::initializer_list<T> il, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(il);
		}
		~scalar(void)
		{
			clear();
		}
		scalar<T, Allocator>& operator=(const scalar<T, Allocator>& x)
		{
			if (this != &x)
			{
				clear();
				if (x.owner)
					assign(x);
				else
					create(x);
			}
			return (*this);
		}
		scalar<T, Allocator>& operator=(scalar<T, Allocator>&& x)
		{
			if (this != &x)
				assign(::std::forward<scalar<T, Allocator> >(x));
			return (*this);
		}
		scalar<T, Allocator>& operator=(::std::initializer_list<T> il)
		{
			clear();
			assign(il);
			return (*this);
		}

		void assign(size_type n)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (n == 0)
				throw ::std::invalid_argument(scalar_invalid_size);
			owner = true;
			count = n;
			buffer = this->allocate(count);
			core::uninitialized_default_construct_n(buffer, count);
			std::
		}

		void assign(size_type n, const value_type& value)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (n == 0)
				throw ::std::invalid_argument(scalar_invalid_size);
			owner = true;
			count = n;
			buffer = this->allocate(count);
			::std::uninitialized_fill_n(buffer, count, value);
		}

		template <class InputIterator>
		void assign(InputIterator first, InputIterator last)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (::std::distance(first, last) <= 0)
				throw ::std::invalid_argument(invalid_initializer_list);
			owner = true;
			count = static_cast<size_type>(::std::distance(first, last));
			buffer = this->allocate(count);
			::std::uninitialized_copy(first, last, buffer);
		}

		void assign(::std::initializer_list<T> il)
		{
			assign(il.begin(), il.end());
		}

		void assign(const scalar<T, Allocator>& x)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (x.empty())
				throw ::std::domain_error(scalar_not_initialized);
			owner = true;
			count = x.count;
			buffer = this->allocate(count);
			::std::uninitialized_copy(x.buffer, x.buffer + count, buffer);
		}

		void assign(scalar<T, Allocator>&& x)
		{
			assign_rv(std::forward<scalar<T, Allocator> >(x), typename allocator_type::propagate_on_container_move_assignment());
		}

		void create(size_type n, pointer p)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (n == 0)
				throw ::std::invalid_argument(scalar_invalid_size);
			owner = false;
			count = n;
			buffer = p;
		}

		void create(scalar<T, Allocator>& x)
		{
			create(x.count, x.buffer);
		}

		// capacity:

		bool empty(void) const noexcept
		{
			return (buffer == nullptr);
		}

		size_type size(void) const noexcept
		{
			return count;
		}

		size_type max_size(void) const noexcept
		{
			return this->max_size();
		}

		// element access:

		reference operator[](size_type idx) noexcept
		{
			return buffer[idx];
		}
		const_reference operator[](size_type idx) const noexcept
		{
			return buffer[idx];
		}

		reference at(size_type idx)
		{
			if (idx >= count)
				throw ::std::out_of_range(scalar_out_of_range);
			return buffer[idx];
		}
		const_reference at(size_type idx) const
		{
			if (idx >= count)
				throw ::std::out_of_range(scalar_out_of_range);
			return buffer[idx];
		}

		pointer data(void) noexcept
		{
			return static_cast<pointer>(buffer);
		}
		const_pointer data(void) const noexcept
		{
			return static_cast<const_pointer>(buffer);
		}

		// modifiers:

		void fill(const value_type& value)
		{
			::std::fill_n(buffer, count, value);
		}

		void fill_n(size_type n, const value_type& value)
		{
			if (n == 0 || n > count)
				throw ::std::invalid_argument(invalid_length);
			::std::fill_n(buffer, n, value);
		}

		template <class InputIterator>
		void fill(InputIterator first, InputIterator last)
		{
			if (static_cast<size_t>(::std::distance(first, last)) != count)
				throw ::std::invalid_argument(invalid_iterator_distance);
			::std::copy(first, last, buffer);
		}

		void fill(::std::initializer_list<T> il)
		{
			fill(il.begin(), il.end());
		}

		void linear_fill(const value_type& init, const value_type& delta)
		{
			value_type value(init);
			for (size_type i = 0; i < count; ++i)
			{
				buffer[i] = value;
				value += delta;
			}
		}

		template<class Generator>
		void generate(Generator g)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] = g();
		}

		template <class U, class A>
		void remap(const_pointer source, const scalar<U, A>& maping)
		{
			if (empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (maping.size() != count)
				throw ::std::invalid_argument(scalar_different_size);
			typename scalar<U, A>::const_pointer index = maping.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] = source[index[i]];
		}

		void swap(scalar<T, Allocator>& rhs) noexcept
		{
			if (this != &rhs)
			{
				::std::swap(owner, rhs.owner);
				::std::swap(count, rhs.count);
				::std::swap(buffer, rhs.buffer);
			}
		}

		void clear(void) noexcept
		{
			if (owner && buffer != nullptr)
			{
				core::destroy_n(buffer, count);
				this->deallocate(buffer, count);
				buffer = nullptr;
			}
			owner = true;
			count = 0;
		}

		// allocator

		allocator_type get_allocator(void) const noexcept
		{
			return *static_cast<const allocator_type*>(this);
		}

		// operator:

		scalar<T, Allocator>& operator+=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] += value;
			return *this;
		}

		scalar<T, Allocator>& operator-=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] -= value;
			return *this;
		}

		scalar<T, Allocator>& operator*=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] *= value;
			return *this;
		}

		scalar<T, Allocator>& operator/=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] /= value;
			return *this;
		}

		scalar<T, Allocator>& operator&=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] &= value;
			return *this;
		}

		scalar<T, Allocator>& operator^=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] ^= value;
			return *this;
		}

		scalar<T, Allocator>& operator|=(const value_type& value)
		{
			for (size_type i = 0; i < count; ++i)
				buffer[i] |= value;
			return *this;
		}

		template <class A>
		scalar<T, Allocator>& operator+=(const scalar<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] += ptr[i];
			return *this;
		}

		template <class A>
		scalar<T, Allocator>& operator-=(const scalar<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] -= ptr[i];
			return *this;
		}

		template <class A>
		scalar<T, Allocator>& operator*=(const scalar<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] *= ptr[i];
			return *this;
		}

		template <class A>
		scalar<T, Allocator>& operator/=(const scalar<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] /= ptr[i];
			return *this;
		}

		template <class A>
		scalar<T, Allocator>& operator&=(const scalar<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] &= ptr[i];
			return *this;
		}

		template <class A>
		scalar<T, Allocator>& operator^=(const scalar<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] ^= ptr[i];
			return *this;
		}

		template <class A>
		scalar<T, Allocator>& operator|=(const scalar<T, A>& rhs)
		{
			if (count != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			const_pointer ptr = rhs.data();
			for (size_type i = 0; i < count; ++i)
				buffer[i] |= ptr[i];
			return *this;
		}
	public:
		// operator:

		template <class A>
		friend scalar<T, Allocator> operator+(const scalar<T, Allocator>& src1, const scalar<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(scalar_different_size);
			scalar<T, Allocator> dst(src1.size());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] + ptr2[i];
			return dst;
		}

		template <class A>
		friend scalar<T, Allocator> operator-(const scalar<T, Allocator>& src1, const scalar<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(scalar_different_size);
			scalar<T, Allocator> dst(src1.size());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] - ptr2[i];
			return dst;
		}

		template <class A>
		friend scalar<T, Allocator> operator*(const scalar<T, Allocator>& src1, const scalar<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(scalar_different_size);
			scalar<T, Allocator> dst(src1.size());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] * ptr2[i];
			return dst;
		}

		template <class A>
		friend scalar<T, Allocator> operator/(const scalar<T, Allocator>& src1, const scalar<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(scalar_different_size);
			scalar<T, Allocator> dst(src1.size());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] / ptr2[i];
			return dst;
		}

		template <class A>
		friend scalar<T, Allocator> operator&(const scalar<T, Allocator>& src1, const scalar<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(scalar_different_size);
			scalar<T, Allocator> dst(src1.size());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] & ptr2[i];
			return dst;
		}

		template <class A>
		friend scalar<T, Allocator> operator^(const scalar<T, Allocator>& src1, const scalar<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(scalar_different_size);
			scalar<T, Allocator> dst(src1.size());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] ^ ptr2[i];
			return dst;
		}

		template <class A>
		friend scalar<T, Allocator> operator|(const scalar<T, Allocator>& src1, const scalar<T, A>& src2)
		{
			if (src1.size() != src2.size())
				throw ::std::invalid_argument(scalar_different_size);
			scalar<T, Allocator> dst(src1.size());
			const_pointer ptr1 = src1.data();
			const_pointer ptr2 = src2.data();
			for (size_t i = 0; i < dst.count; ++i)
				dst.buffer[i] = ptr1[i] | ptr2[i];
			return dst;
		}

		template <class A>
		friend bool operator<(const scalar<T, Allocator>& lhs, const scalar<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			bool rst = true;
			const_pointer ptr1 = lhs.data();
			const_pointer ptr2 = rhs.data();
			for (size_t i = 0; i < lhs.count; ++i)
			{
				if (!(ptr1[i] < ptr2[i]))
				{
					rst = false;
					break;
				}
			}
			return rst;
		}
		template <class A>
		friend bool operator>(const scalar<T, Allocator>& lhs, const scalar<T, A>& rhs)
		{
			return (rhs < lhs);
		}
		template <class A>
		friend bool operator<=(const scalar<T, Allocator>& lhs, const scalar<T, A>& rhs)
		{
			return !(lhs > rhs);
		}
		template <class A>
		friend bool operator>=(const scalar<T, Allocator>& lhs, const scalar<T, A>& rhs)
		{
			return !(lhs < rhs);
		}

		template <class A>
		friend bool operator==(const scalar<T, Allocator>& lhs, const scalar<T, A>& rhs)
		{
			if (lhs.size() != rhs.size())
				throw ::std::invalid_argument(scalar_different_size);
			bool rst = true;
			const_pointer ptr1 = lhs.data();
			const_pointer ptr2 = rhs.data();
			for (size_t i = 0; i < lhs.count; ++i)
			{
				if (!(ptr1[i] == ptr2[i]))
				{
					rst = false;
					break;
				}
			}
			return rst;
		}
		template <class A>
		friend bool operator!=(const scalar<T, Allocator>& lhs, const scalar<T, A>& rhs)
		{
			return !(lhs == rhs);
		}
	private:
		void assign_rv(scalar<T, Allocator>&& right, ::std::true_type)
		{
			swap(right);
		}
		void assign_rv(scalar<T, Allocator>&& right, ::std::false_type)
		{
			if (get_allocator() == right.get_allocator())
				assign_rv(::std::forward<scalar<T, Allocator> >(right), ::std::true_type());
			else
				assign(right);
		}
	private:
		bool       owner;
		size_type  count;
		pointer    buffer;
	};

} // namespace core

#endif
