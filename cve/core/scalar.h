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

	// Class template scalar
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
		scalar(size_type n, const value_type& value, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(n, value);
		}
		scalar(size_type n, const_pointer p, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(n, p);
		}
		scalar(size_type n, pointer p, bool copy_data, const Allocator& alloc = Allocator())
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(n, p, copy_data);
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
		scalar(const scalar<T, Allocator>& other, copy_mode_type copy_mode = deep_copy)
			: Allocator(other.get_allocator())
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, copy_mode);
		}
		scalar(scalar<T, Allocator>&& other)
			: Allocator(other.get_allocator())
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<scalar<T, Allocator> >(other));
		}
		scalar(const scalar<T, Allocator>& other, const Allocator& alloc, copy_mode_type copy_mode = deep_copy)
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(other, copy_mode);
		}
		scalar(scalar<T, Allocator>&& other, const Allocator& alloc)
			: Allocator(alloc)
			, owner(true)
			, count(0)
			, buffer(nullptr)
		{
			assign(::std::forward<scalar<T, Allocator> >(other));
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
		scalar<T, Allocator>& operator=(const scalar<T, Allocator>& other)
		{
			if (this != &other)
			{
				clear();
				assign(other, other.owner ? deep_copy : shallow_copy);
			}
			return (*this);
		}
		scalar<T, Allocator>& operator=(scalar<T, Allocator>&& other)
		{
			if (this != &other)
				assign(::std::forward<scalar<T, Allocator> >(other));
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
				throw ::std::invalid_argument(invalid_scalar_size);
			owner = true;
			count = n;
			buffer = this->allocate(count);
			core::uninitialized_default_construct_n(buffer, count);
		}

		void assign(size_type n, const value_type& value)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (n == 0)
				throw ::std::invalid_argument(invalid_scalar_size);
			owner = true;
			count = n;
			buffer = this->allocate(count);
			::std::uninitialized_fill_n(buffer, count, value);
		}

		void assign(size_type n, const_pointer p)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (n == 0)
				throw ::std::invalid_argument(invalid_scalar_size);
			owner = true;
			count = n;
			buffer = this->allocate(count);
			::std::uninitialized_copy(p, p + count, buffer);
		}

		void assign(size_type n, pointer p, bool copy_data)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (n == 0)
				throw ::std::invalid_argument(invalid_scalar_size);
			owner = copy_data;
			count = n;
			if (copy_data)
			{
				buffer = this->allocate(count);
				::std::uninitialized_copy(p, p + count, buffer);
			}
			else
				buffer = p;
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

		void assign(const scalar<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (other.empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (copy_mode == shallow_copy)
				throw ::std::invalid_argument(invalid_copy_mode);
			owner = true;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
				break;
			case deep_copy:
				buffer = this->allocate(count);
				::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				break;
			}
		}

		void assign(scalar<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (!empty())
				throw ::std::domain_error(scalar_is_initialized);
			if (other.empty())
				throw ::std::domain_error(scalar_not_initialized);
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				owner = true;
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
				break;
			case shallow_copy:
				owner = false;
				buffer = other.data();
				break;
			case deep_copy:
				owner = true;
				buffer = this->allocate(count);
				::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				break;
			}
		}

		void assign(scalar<T, Allocator>&& other)
		{
			assign_rv(std::forward<scalar<T, Allocator> >(other), typename allocator_type::propagate_on_container_move_assignment());
		}

		void reassign(size_type n)
		{
			if (n == 0)
				throw ::std::invalid_argument(invalid_scalar_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			count = n;
			if (!original_owner || buffer == nullptr)
			{
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				core::uninitialized_default_construct_n(buffer, count);
			}
		}

		void reassign(size_type n, const value_type& value)
		{
			if (n == 0)
				throw ::std::invalid_argument(invalid_scalar_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			count = n;
			if (!original_owner || buffer == nullptr)
			{
				buffer = this->allocate(count);
				::std::uninitialized_fill_n(buffer, count, value);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				::std::uninitialized_fill_n(buffer, count, value);
			}
			else
				::std::fill_n(buffer, count, value);
		}

		void reassign(size_type n, const_pointer p)
		{
			if (n == 0)
				throw ::std::invalid_argument(invalid_scalar_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			count = n;
			if (!original_owner || buffer == nullptr)
			{
				buffer = this->allocate(count);
				::std::uninitialized_copy(p, p + count, buffer);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				::std::uninitialized_copy(p, p + count, buffer);
			}
			else
				::std::copy(p, p + count, buffer);
		}

		void reassign(size_type n, pointer p, bool copy_data)
		{
			if (n == 0)
				throw ::std::invalid_argument(invalid_scalar_size);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = copy_data;
			count = n;
			if (copy_data)
			{
				if (!original_owner || buffer == nullptr)
				{
					buffer = this->allocate(count);
					::std::uninitialized_copy(p, p + count, buffer);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					::std::uninitialized_copy(p, p + count, buffer);
				}
				else
					::std::copy(p, p + count, buffer);
			}
			else
			{
				if (original_owner && buffer != nullptr)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
				}
				buffer = p;
			}
		}

		template <class InputIterator>
		void reassign(InputIterator first, InputIterator last)
		{
			if (::std::distance(first, last) == 0)
				throw ::std::invalid_argument(invalid_initializer_list);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			count = ::std::distance(first, last);
			if (!original_owner || buffer == nullptr)
			{
				buffer = this->allocate(count);
				::std::uninitialized_copy(first, last, buffer);
			}
			else if (count != original_count)
			{
				core::destroy_n(buffer, original_count);
				this->deallocate(buffer, original_count);
				buffer = this->allocate(count);
				::std::uninitialized_copy(first, last, buffer);
			}
			else
				::std::copy(first, last, buffer);
		}

		void reassign(::std::initializer_list<T> il)
		{
			reassign(il.begin(), il.end());
		}

		void reassign(const scalar<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (other.empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (copy_mode == shallow_copy)
				throw ::std::invalid_argument(invalid_copy_mode);
			size_type original_owner = owner;
			size_type original_count = count;
			owner = true;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				if (!original_owner || buffer == nullptr)
				{
					buffer = this->allocate(count);
					core::uninitialized_default_construct_n(buffer, count);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					core::uninitialized_default_construct_n(buffer, count);
				}
				break;
			case deep_copy:
				if (!original_owner || buffer == nullptr)
				{
					buffer = this->allocate(count);
					::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				}
				else
					::std::copy(other.buffer, other.buffer + count, buffer);
				break;
			}
		}

		void reassign(scalar<T, Allocator>& other, copy_mode_type copy_mode)
		{
			if (other.empty())
				throw ::std::domain_error(scalar_not_initialized);
			size_type original_owner = owner;
			size_type original_count = count;
			count = other.count;
			switch (copy_mode)
			{
			case without_copy:
				owner = true;
				if (!original_owner || buffer == nullptr)
				{
					buffer = this->allocate(count);
					core::uninitialized_default_construct_n(buffer, count);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					core::uninitialized_default_construct_n(buffer, count);
				}
				break;
			case shallow_copy:
				owner = false;
				if (original_owner && buffer != nullptr)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
				}
				buffer = other.data();
				break;
			case deep_copy:
				owner = true;
				if (!original_owner || buffer == nullptr)
				{
					buffer = this->allocate(count);
					::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				}
				else if (count != original_count)
				{
					core::destroy_n(buffer, original_count);
					this->deallocate(buffer, original_count);
					buffer = this->allocate(count);
					::std::uninitialized_copy(other.buffer, other.buffer + count, buffer);
				}
				else
					::std::copy(other.buffer, other.buffer + count, buffer);
				break;
			}
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
			if (empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (idx >= count)
				throw ::std::out_of_range(scalar_out_of_range);
			return buffer[idx];
		}
		const_reference at(size_type idx) const
		{
			if (empty())
				throw ::std::domain_error(scalar_not_initialized);
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
			if (empty())
				throw ::std::domain_error(scalar_not_initialized);
			::std::fill_n(buffer, count, value);
		}

		void fill_n(size_type n, const value_type& value)
		{
			if (empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (n == 0 || n > count)
				throw ::std::invalid_argument(invalid_length);
			::std::fill_n(buffer, n, value);
		}

		template <class InputIterator>
		void fill(InputIterator first, InputIterator last)
		{
			if (empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (static_cast<size_t>(::std::distance(first, last)) != count)
				throw ::std::invalid_argument(invalid_iterator_distance);
			::std::copy(first, last, buffer);
		}

		void fill(::std::initializer_list<T> il)
		{
			fill(il.begin(), il.end());
		}

		void fill(const scalar<T, Allocator>& other)
		{
			if (empty() || other.empty())
				throw ::std::domain_error(scalar_not_initialized);
			if (count != other.size())
				throw ::std::invalid_argument(invalid_size);
			::std::copy(other.buffer, other.buffer + count, buffer);
		}

		template<class Generator>
		void generate(Generator g)
		{
			if (empty())
				throw ::std::domain_error(scalar_not_initialized);
			for (size_type i = 0; i < count; ++i)
				buffer[i] = g();
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

	public:
		// operator:

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
				assign(right, deep_copy);
		}
	private:
		bool       owner;
		size_type  count;
		pointer    buffer;
	};

} // namespace core

#endif
