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

#ifndef __CORE_ALIGNED_ALLOCATOR_H__
#define __CORE_ALIGNED_ALLOCATOR_H__

#include <memory>
#include <exception>

namespace core
{
	template <class T, size_t align, int inst> class aligned_allocator;

	// Specialize for void
	template <size_t align, int inst>
	class aligned_allocator<void, align, inst>
	{
	public:
		typedef void              value_type;
		typedef void*             pointer;
		typedef const void*       const_pointer;
		typedef size_t            size_type;
		typedef ptrdiff_t         difference_type;

		template <class U> struct rebind
		{
			typedef aligned_allocator<U, align, inst> other;
		};
	};

	// Template class aligned_allocator_base
	template <size_t align, int inst>
	class aligned_allocator_base
	{
	public:
		size_t capacity(void)
		{
			return heap_size;
		}
		size_t size(void)
		{
			return used_size;
		}
	protected:
		void* arw_allocate(size_t size)
		{
			if (size > max_bytes)
			{
				void* result = _aligned_malloc(size, align);
				if (result == nullptr)
					throw std::bad_alloc();
				return result;
			}
			else
			{
				char* result = nullptr;
				size_t count = chunk_count;
				free_chunk_node** current_list = reinterpret_cast<free_chunk_node**>(get_free_list(size));

				if (*current_list != nullptr)
				{
					result = (*current_list)->buffer;
					*current_list = (*current_list)->next;
				}
				else
				{
					size_t align_bytes = round_up(size);
					size_t total_bytes = align_bytes * count;
					size_t free_bytes = free_end - free_start;

					if (free_bytes >= total_bytes)
					{
						result = free_start;
						free_start += total_bytes;
					}
					else if (free_bytes >= align_bytes)
					{
						count = free_bytes / align_bytes;
						total_bytes = align_bytes * count;
						result = free_start;
						free_start += total_bytes;
					}
					else
					{
						if (free_bytes > 0)
						{
							free_chunk_node** free_list_left = reinterpret_cast<free_chunk_node**>(get_free_list(free_bytes));
							reinterpret_cast<free_chunk_node*>(free_start)->next = *free_list_left;
							*free_list_left = reinterpret_cast<free_chunk_node*>(free_start);
							free_start = free_end;
						}
						result = chunk_alloc(align_bytes, count);
					}
					if (result != nullptr && count > 1)
					{
						char *cur, *next = result + align_bytes;
						*current_list = reinterpret_cast<free_chunk_node*>(next);
						for (size_t i = 2; i < count; ++i)
						{
							cur = next;
							next += align_bytes;
							reinterpret_cast<free_chunk_node*>(cur)->next = reinterpret_cast<free_chunk_node*>(next);
						}
						reinterpret_cast<free_chunk_node*>(next)->next = nullptr;
					}
				}
				if (result != nullptr)
					used_size += size;
				else
					throw std::bad_alloc();
				return reinterpret_cast<void*>(result);
			}
		}
		void* arw_reallocate(void* p, size_t old_size, size_t new_size)
		{
			void* result = nullptr;
			size_t copy_size;

			if (old_size > max_bytes && new_size > max_bytes)
			{
				result = _aligned_realloc(p, new_size, align);
			}
			if (round_up(old_size) == round_up(new_size))
			{
				result = p;
				used_size -= old_size;
				used_size += new_size;
			}
			else
			{
				result = arw_allocate(new_size);
				if (result == nullptr)
				{
					copy_size = new_size > old_size ? old_size : new_size;
					memcpy(result, p, copy_size);
					arw_deallocate(p, old_size);
				}
			}
			if (result == nullptr)
				throw std::bad_alloc();
			return result;
		}
		void arw_deallocate(void* p, size_t size)
		{
			if (size > max_bytes)
			{
				_aligned_free(p);
			}
			else
			{
				free_chunk_node** current_list = reinterpret_cast<free_chunk_node**>(get_free_list(size));
				reinterpret_cast<free_chunk_node*>(p)->next = *current_list;
				*current_list = reinterpret_cast<free_chunk_node*>(p);
				used_size -= size;
			}
		}
		void release(void)
		{
			if (used_size != 0)
				return;
			while (memory_list != nullptr)
			{
				void* next = *reinterpret_cast<void**>(memory_list);
				// std::cout << "free: ";
				// std::cout << std::hex << std::setiosflags(std::ios::showbase);
				// std::cout << reinterpret_cast<size_t>(memory_list) << std::endl;
				free(memory_list);
				memory_list = next;
			}
			heap_size = 0;
			used_size = 0;
		}
	private:
		inline size_t round_up(size_t bytes)
		{
			return ((bytes + align - 1) & ~(align - 1));
		}
		inline void** get_free_list(size_t bytes)
		{
			return reinterpret_cast<void**>(free_list + (bytes + align - 1) / align - 1);
		}
		char* chunk_alloc(size_t align_bytes, size_t& count)
		{
			char* result = nullptr;
			size_t memory_size;
			size_t total_bytes = align_bytes * count;

			if (heap_size < heap_threshold)
				memory_size = total_bytes << 1;
			else
				memory_size = (heap_size >> 7) << 3;
			while (memory_size >= total_bytes)
			{
				void* p = malloc(memory_size + memory_surplus);
				// std::cout << "malloc: ";
				// std::cout << std::hex << std::setiosflags(std::ios::showbase);
				// std::cout << reinterpret_cast<size_t>(p) << std::endl;
				if (p != nullptr)
				{
					result = reinterpret_cast<char*>((reinterpret_cast<size_t>(p) + memory_surplus) & ~(align - 1));
					free_start = result + total_bytes;
					free_end = result + memory_size;
					heap_size += memory_size;
					*reinterpret_cast<void**>(p) = memory_list;
					memory_list = reinterpret_cast<void**>(p);
					break;
				}
				memory_size >>= 1;
			}
			if (result == nullptr)
			{
				for (size_t size = align_bytes; size <= max_bytes; size += align)
				{
					free_chunk_node** current_list = reinterpret_cast<free_chunk_node**>(get_free_list(size));
					if (*current_list != nullptr)
					{
						count = 1;
						result = (*current_list)->buffer;
						free_start = result + align_bytes;
						free_end = result + size;
						break;
					}
				}
			}
			return result;
		}
	private:
		union free_chunk_node
		{
			free_chunk_node*    next;
			char                buffer[1];
		};
	private:
		static constexpr size_t max_bytes = align << 4;
		static constexpr size_t free_list_size = 16;
		static constexpr size_t chunk_count = 16;
		static constexpr size_t heap_threshold = (max_bytes * chunk_count) << 5;
		static constexpr size_t memory_surplus = sizeof(void*) + align - 1;

		static size_t           heap_size;
		static size_t           used_size;
		static char*            free_start;
		static char*            free_end;
		static void*            memory_list;
		static free_chunk_node* free_list[free_list_size];
	};

	// Initialize static variables
	template <size_t align, int inst>
	size_t aligned_allocator_base<align, inst>::heap_size = 0;
	template <size_t align, int inst>
	size_t aligned_allocator_base<align, inst>::used_size = 0;
	template <size_t align, int inst>
	char*  aligned_allocator_base<align, inst>::free_start = nullptr;
	template <size_t align, int inst>
	char*  aligned_allocator_base<align, inst>::free_end = nullptr;
	template <size_t align, int inst>
	void* aligned_allocator_base<align, inst>::memory_list = nullptr;
	template <size_t align, int inst>
	typename aligned_allocator_base<align, inst>::free_chunk_node*
		aligned_allocator_base<align, inst>::free_list[free_list_size] =
	{
		nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
	};

	// Template class aligned_allocator
	template <class T, size_t align, int inst = 0>
	class aligned_allocator : public aligned_allocator_base<align, inst>
	{
	public:
		typedef T                 value_type;
		typedef value_type*       pointer;
		typedef const value_type* const_pointer;
		typedef value_type&       reference;
		typedef const value_type& const_reference;
		typedef size_t            size_type;
		typedef ptrdiff_t         difference_type;

		template <class U> struct rebind
		{
			typedef aligned_allocator<U, align, inst> other;
		};

		typedef ::std::false_type propagate_on_container_copy_assignment;
		typedef ::std::false_type propagate_on_container_move_assignment;
		typedef ::std::false_type propagate_on_container_swap;
		typedef ::std::true_type  is_always_equal;
	public:
		// Construct default aligned_allocator
		aligned_allocator(void) noexcept
		{
		}
		// Construct by copying
		aligned_allocator(const aligned_allocator<T, align, inst>&) noexcept
		{
		}
		// Construct from a related aligned_allocator
		template<class U>
		aligned_allocator(const aligned_allocator<U, align, inst>&) noexcept
		{
		}
		// Destructor
		~aligned_allocator(void)
		{
			this->release();
		}
		// Assign from a related aligned_allocator
		template<class U>
		aligned_allocator<T, align, inst>& operator=(const aligned_allocator<U, align, inst>&)
		{
			return (*this);
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
		pointer allocate(size_type n, typename aligned_allocator<void, align, inst>::const_pointer hint = nullptr)
		{
			return (static_cast<pointer>(this->arw_allocate(n * sizeof(value_type))));
		}
		// Deallocate object at p
		void deallocate(pointer p, size_type n)
		{
			this->arw_deallocate(p, n * sizeof(value_type));
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

	template <class T1, class T2, size_t align, int inst>
	inline bool operator==(const aligned_allocator<T1, align, inst>&, const aligned_allocator<T2, align, inst>&)
	{
		return true;
	}
	template <class T1, class T2, size_t align, int inst>
	inline bool operator!=(const aligned_allocator<T1, align, inst>&, const aligned_allocator<T2, align, inst>&)
	{
		return false;
	}

} // namespace core

#endif
