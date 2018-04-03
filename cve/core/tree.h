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

#ifndef __CORE_TREE_H__
#define __CORE_TREE_H__

#include <memory>
#include <iterator>
#include <functional>
#include <utility>

namespace core
{
	typedef signed char tree_node_state;
	static constexpr tree_node_state tree_state_parent = -0x10;
	static constexpr tree_node_state tree_state_sibling = 0x01;
	static constexpr tree_node_state tree_state_child = 0x12;

	// Template class tree_node
	template <class T>
	struct tree_node
	{
		typedef tree_node<T>     node_type;
		typedef node_type*       node_pointer;
		typedef const node_type* const_node_pointer;
		typedef node_type&       node_reference;
		typedef const node_type& const_node_reference;

		node_pointer             parent;
		node_pointer             prev_sibling;
		node_pointer             next_sibling;
		node_pointer             first_child;
		node_pointer             last_child;
		T                        data;
	};

	// Template class tree_type_traits

	template <class Tree, bool IsConst>
	struct tree_type_traits
	{
		typedef typename Tree::node_type       node_type;
		typedef typename Tree::node_pointer    node_pointer;
		typedef typename Tree::value_type      value_type;
		typedef typename Tree::pointer         pointer;
		typedef typename Tree::reference       reference;
		typedef typename Tree::size_type       size_type;
		typedef typename Tree::difference_type difference_type;
	};

	template <class Tree>
	struct tree_type_traits<Tree, true>
	{
		typedef typename Tree::node_type       node_type;
		typedef typename Tree::node_pointer    node_pointer;
		typedef typename Tree::value_type      value_type;
		typedef typename Tree::const_pointer   pointer;
		typedef typename Tree::const_reference reference;
		typedef typename Tree::size_type       size_type;
		typedef typename Tree::difference_type difference_type;
	};

	// Template class tree_iterator
	template <class Tree, bool IsConst>
	class tree_iterator
	{
	public:
		// types:

		typedef tree_iterator<Tree, IsConst>                              iterator_type;
		typedef ::std::bidirectional_iterator_tag                         iterator_category;

		typedef typename tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		tree_iterator(void) noexcept
			: node(nullptr)
		{}
		explicit tree_iterator(const node_pointer ptr) noexcept
			: node(ptr)
		{}
		tree_iterator(const tree_iterator<Tree, IsConst>& x) noexcept
			: node(x.get_pointer())
		{}

		tree_iterator<Tree, IsConst>& operator=(const tree_iterator<Tree, IsConst>& x) noexcept
		{
			if (this != &x)
				node = x.get_pointer();
			return (*this);
		}

		operator tree_iterator<Tree, true>(void) const noexcept
		{
			return tree_iterator<Tree, true>(node);
		}

		// tree_iterator operations:

		node_pointer get_parent(void) noexcept
		{
			return node->parent;
		}
		const node_pointer get_parent(void) const noexcept
		{
			return node->parent;
		}

		node_pointer get_pointer(void) noexcept
		{
			return node;
		}
		const node_pointer get_pointer(void) const noexcept
		{
			return node;
		}

		reference operator*(void) const noexcept
		{
			return node->data;
		}
		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		tree_iterator<Tree, IsConst>& operator++(void) noexcept
		{
			if (node->first_child != nullptr)
				node = node->first_child;
			else
			{
				while (node->next_sibling == nullptr)
					node = node->parent;
				node = node->next_sibling;
			}
			return *this;
		}
		tree_iterator<Tree, IsConst>& operator--(void) noexcept
		{
			if (node->prev_sibling != nullptr)
			{
				node = node->prev_sibling;
				while (node->last_child != nullptr)
					node = node->last_child;
			}
			else
				node = node->parent;
			return *this;
		}
		tree_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		tree_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const tree_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const tree_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer node;
	};

	// Template class tree_primitive_iterator
	template <class Tree, bool IsConst>
	class tree_primitive_iterator
	{
	public:
		// types:

		typedef tree_primitive_iterator<Tree, IsConst>                    iterator_type;
		typedef ::std::bidirectional_iterator_tag                         iterator_category;

		typedef typename tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		tree_primitive_iterator(void) noexcept
			: node(nullptr)
			, state(tree_state_child)
		{}
		explicit tree_primitive_iterator(const node_pointer ptr) noexcept
			: node(ptr)
			, state(tree_state_child)
		{}
		tree_primitive_iterator(const tree_primitive_iterator<Tree, IsConst>& x) noexcept
			: node(x.get_pointer())
			, state(x.get_state())
		{}

		tree_primitive_iterator<Tree, IsConst>& operator=(const tree_primitive_iterator<Tree, IsConst>& x) noexcept
		{
			if (this != &x)
			{
				node = x.get_pointer();
				state = x.get_state();
			}
			return (*this);
		}

		operator tree_primitive_iterator<Tree, true>(void) const noexcept
		{
			return tree_primitive_iterator<Tree, true>(node);
		}

		// tree_primitive_iterator operations:

		node_pointer get_parent(void) noexcept
		{
			return node->parent;
		}
		const node_pointer get_parent(void) const noexcept
		{
			return node->parent;
		}

		node_pointer get_pointer(void) noexcept
		{
			return node;
		}
		const node_pointer get_pointer(void) const noexcept
		{
			return node;
		}

		tree_node_state get_state(void) const noexcept
		{
			return state;
		}

		intptr_t get_depth(void) const noexcept
		{
			return static_cast<intptr_t>(state >> 4);
		}

		reference operator*(void) const noexcept
		{
			return node->data;
		}
		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		tree_primitive_iterator<Tree, IsConst>& operator++(void) noexcept
		{
			if (state != tree_state_parent)
			{
				if (node->first_child != nullptr)
				{
					node = node->first_child;
					state = tree_state_child;
				}
				else if (node->next_sibling != nullptr)
				{
					node = node->next_sibling;
					state = tree_state_sibling;
				}
				else
				{
					node = node->parent;
					state = tree_state_parent;
				}
			}
			else
			{
				if (node->next_sibling != nullptr)
				{
					node = node->next_sibling;
					state = tree_state_sibling;
				}
				else
					node = node->parent;
			}
			return *this;
		}
		tree_primitive_iterator<Tree, IsConst>& operator--(void) noexcept
		{
			if (state != tree_state_parent)
			{
				if (node->last_child != nullptr)
				{
					node = node->last_child;
					state = tree_state_child;
				}
				else if (node->prev_sibling != nullptr)
				{
					node = node->prev_sibling;
					state = tree_state_sibling;
				}
				else
				{
					node = node->parent;
					state = tree_state_parent;
				}
			}
			else
			{
				if (node->prev_sibling != nullptr)
				{
					node = node->prev_sibling;
					state = tree_state_sibling;
				}
				else
					node = node->parent;
			}
			return *this;
		}
		tree_primitive_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		tree_primitive_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const tree_primitive_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const tree_primitive_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer    node;
		tree_node_state state;
	};

	// Template class tree_sibling_iterator
	template <class Tree, bool IsConst>
	class tree_sibling_iterator
	{
	public:
		// types:

		typedef tree_sibling_iterator<Tree, IsConst>                      iterator_type;
		typedef ::std::bidirectional_iterator_tag                         iterator_category;

		typedef typename tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		tree_sibling_iterator(void) noexcept
			: parent(nullptr)
			, node(nullptr)
		{}
		explicit tree_sibling_iterator(const node_pointer ptr) noexcept
			: parent(ptr->parent)
			, node(ptr)
		{}
		explicit tree_sibling_iterator(const node_pointer parent_ptr, const node_pointer ptr) noexcept
			: parent(parent_ptr)
			, node(ptr)
		{}
		tree_sibling_iterator(const tree_sibling_iterator<Tree, IsConst>& rhs) noexcept
			: parent(rhs.get_parent())
			, node(rhs.get_pointer())
		{}

		tree_sibling_iterator<Tree, IsConst>& operator=(const tree_sibling_iterator<Tree, IsConst>& x)
		{
			if (this != &x)
			{
				parent = x.get_parent();
				node = x.get_pointer();
			}
			return (*this);
		}

		operator tree_sibling_iterator<Tree, true>(void) const noexcept
		{
			return tree_sibling_iterator<Tree, true>(node);
		}

		// tree_sibling_iterator operations:

		node_pointer get_parent(void) noexcept
		{
			return parent;
		}
		const node_pointer get_parent(void) const noexcept
		{
			return parent;
		}

		node_pointer get_pointer(void) noexcept
		{
			return node;
		}
		const node_pointer get_pointer(void) const noexcept
		{
			return node;
		}

		bool is_end(void) const noexcept
		{
			return (node == parent);
		}

		reference operator*(void) const noexcept
		{
			return node->data;
		}
		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		tree_sibling_iterator<Tree, IsConst>& operator++(void) noexcept
		{
			if (node != parent)
				node = (node->next_sibling != nullptr) ? node->next_sibling : parent;
			else
				node = (parent->first_child != nullptr) ? parent->first_child : parent;
			return *this;
		}
		tree_sibling_iterator<Tree, IsConst>& operator--(void) noexcept
		{
			if (node != parent)
				node = (node->prev_sibling != nullptr) ? node->prev_sibling : parent;
			else
				node = (parent->last_child != nullptr) ? parent->last_child : parent;
			return *this;
		}
		tree_sibling_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		tree_sibling_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const tree_sibling_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const tree_sibling_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer parent;
		node_pointer node;
	};

	// Template class tree_leaf_iterator
	template <class Tree, bool IsConst>
	class tree_leaf_iterator
	{
	public:
		// types:

		typedef tree_leaf_iterator<Tree, IsConst>                         iterator_type;
		typedef ::std::bidirectional_iterator_tag                         iterator_category;

		typedef typename tree_type_traits<Tree, IsConst>::node_type       node_type;
		typedef typename tree_type_traits<Tree, IsConst>::node_pointer    node_pointer;
		typedef typename tree_type_traits<Tree, IsConst>::value_type      value_type;
		typedef typename tree_type_traits<Tree, IsConst>::pointer         pointer;
		typedef typename tree_type_traits<Tree, IsConst>::reference       reference;
		typedef typename tree_type_traits<Tree, IsConst>::size_type       size_type;
		typedef typename tree_type_traits<Tree, IsConst>::difference_type difference_type;

		// construct/copy/destroy:

		tree_leaf_iterator(void) noexcept
			: node(nullptr)
		{}
		explicit tree_leaf_iterator(const node_pointer ptr) noexcept
			: node(ptr)
		{}
		tree_leaf_iterator(const tree_leaf_iterator<Tree, IsConst>& x) noexcept
			: node(x.get_pointer())
		{}

		tree_leaf_iterator<Tree, IsConst>& operator=(const tree_leaf_iterator<Tree, IsConst>& x) noexcept
		{
			if (this != &x)
				node = x.get_pointer();
			return (*this);
		}

		operator tree_leaf_iterator<Tree, true>(void) const noexcept
		{
			return tree_leaf_iterator<Tree, true>(node);
		}

		// tree_leaf_iterator operations:

		node_pointer get_parent(void) noexcept
		{
			return node->parent;
		}
		const node_pointer get_parent(void) const noexcept
		{
			return node->parent;
		}

		node_pointer get_pointer(void) noexcept
		{
			return node;
		}
		const node_pointer get_pointer(void) const noexcept
		{
			return node;
		}

		reference operator*(void) const noexcept
		{
			return node->data;
		}
		pointer operator->(void) const noexcept
		{
			return &(operator*());
		}

		// increment / decrement

		tree_leaf_iterator<Tree, IsConst>& operator++(void) noexcept
		{
			if (node->first_child != nullptr)
			{
				while (node->first_child != nullptr)
					node = node->first_child;
			}
			else
			{
				while (node->next_sibling == nullptr)
					node = node->parent;
				node = node->next_sibling;
				if (node->parent != nullptr)
				{
					while (node->first_child != nullptr)
						node = node->first_child;
				}
			}
			return *this;
		}
		tree_leaf_iterator<Tree, IsConst>& operator--(void) noexcept
		{
			while (node->prev_sibling == nullptr)
				node = node->parent;
			node = node->prev_sibling;
			while (node->last_child != nullptr)
				node = node->last_child;
			return *this;
		}
		tree_leaf_iterator<Tree, IsConst> operator++(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator++();
			return tmp;
		}
		tree_leaf_iterator<Tree, IsConst> operator--(int) noexcept
		{
			iterator_type tmp(*this);
			this->operator--();
			return tmp;
		}

		// relational operators:

		template <bool b>
		bool operator==(const tree_leaf_iterator<Tree, b>& rhs) const noexcept
		{
			return (node == rhs.get_pointer());
		}
		template <bool b>
		bool operator!=(const tree_leaf_iterator<Tree, b>& rhs) const noexcept
		{
			return (node != rhs.get_pointer());
		}
	private:
		node_pointer node;
	};

	// Template class tree_node_allocator
	template <class T, class Allocator>
	class tree_node_allocator : public Allocator
	{
	public:
		// types:

		typedef Allocator                                                  allocator_type;
		typedef typename tree_node<T>::node_type                           tree_node_type;
		typedef typename Allocator::template rebind<tree_node_type>::other node_allocator_type;
		typedef ::std::allocator_traits<allocator_type>                    allocator_traits_type;
		typedef ::std::allocator_traits<node_allocator_type>               node_allocator_traits_type;

		typedef typename node_allocator_traits_type::size_type             node_size_type;
		typedef typename node_allocator_traits_type::difference_type       node_difference_type;
		typedef typename node_allocator_traits_type::value_type            node_type;
		typedef typename node_allocator_traits_type::pointer               node_pointer;

		// construct/copy/destroy:

		tree_node_allocator(void)
			: Allocator()
		{}
		explicit tree_node_allocator(const Allocator& alloc)
			: Allocator(alloc)
		{}
		explicit tree_node_allocator(Allocator&& alloc)
			: Allocator(::std::forward<Allocator>(alloc))
		{}
		~tree_node_allocator(void)
		{}

		// tree_node_allocator operations:

		allocator_type get_allocator(void) const noexcept
		{
			return *static_cast<const allocator_type*>(this);
		}
		node_size_type max_size(void) const noexcept
		{
			return node_alloc.max_size();
		}
	protected:
		template<class ...Args>
		node_pointer create_node(Args&&... args)
		{
			node_pointer p = node_alloc.allocate(1);
			get_allocator().construct(::std::addressof(p->data), ::std::forward<Args>(args)...);
			return p;
		}
		void destroy_node(const node_pointer p)
		{
			get_allocator().destroy(::std::addressof(p->data));
			node_alloc.deallocate(p, 1);
		}
	private:
		node_allocator_type node_alloc;
	};

	// Template class tree
	template <class T, class Allocator = ::std::allocator<T> >
	class tree : public tree_node_allocator<T, Allocator>
	{
	public:
		// types:

		typedef Allocator                                         allocator_type;
		typedef tree_node_allocator<T, Allocator>                 node_allocator_type;
		typedef tree<T, Allocator>                                tree_type;
		typedef typename node_allocator_type::node_type           node_type;
		typedef typename node_allocator_type::node_pointer        node_pointer;
		typedef typename allocator_type::value_type               value_type;
		typedef typename allocator_type::pointer                  pointer;
		typedef typename allocator_type::const_pointer            const_pointer;
		typedef typename allocator_type::reference                reference;
		typedef typename allocator_type::const_reference          const_reference;
		typedef typename allocator_type::size_type                size_type;
		typedef typename allocator_type::difference_type          difference_type;

		typedef tree_iterator<tree_type, false>                   iterator;
		typedef tree_iterator<tree_type, true>                    const_iterator;
		typedef tree_primitive_iterator<tree_type, false>         primitive_iterator;
		typedef tree_primitive_iterator<tree_type, true>          const_primitive_iterator;
		typedef tree_sibling_iterator<tree_type, false>           sibling_iterator;
		typedef tree_sibling_iterator<tree_type, true>            const_sibling_iterator;
		typedef tree_leaf_iterator<tree_type, false>              leaf_iterator;
		typedef tree_leaf_iterator<tree_type, true>               const_leaf_iterator;
		typedef ::std::reverse_iterator<iterator>                 reverse_iterator;
		typedef ::std::reverse_iterator<const_iterator>           const_reverse_iterator;
		typedef ::std::reverse_iterator<primitive_iterator>       reverse_primitive_iterator;
		typedef ::std::reverse_iterator<const_primitive_iterator> const_reverse_primitive_iterator;
		typedef ::std::reverse_iterator<sibling_iterator>         reverse_sibling_iterator;
		typedef ::std::reverse_iterator<const_sibling_iterator>   const_reverse_sibling_iterator;
		typedef ::std::reverse_iterator<leaf_iterator>            reverse_leaf_iterator;
		typedef ::std::reverse_iterator<const_leaf_iterator>      const_reverse_leaf_iterator;

		// construct/copy/destroy:

		explicit tree(const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
		{
			create_root();
		}
		explicit tree(size_type n, const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
		{
			create_root();
			for (; n > 0; --n)
				push_back(value_type());
		}
		tree(size_type n, const T& value, const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
		{
			create_root();
			for (; n > 0; --n)
				push_back(value);
		}
		tree(const_sibling_iterator first, const_sibling_iterator last, const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
		{
			create_root();
			for (; first != last; ++first)
				push_back(*first);
		}
		tree(const tree<T, Allocator>& x)
			: node_allocator_type(x.get_allocator())
		{
			create_root();
			copy_child_back(csend(), x.csbegin(), x.csend());
		}
		tree(const tree<T, Allocator>& x, const Allocator& alloc)
			: node_allocator_type(alloc)
		{
			create_root();
			copy_child_back(csend(), x.csbegin(), x.csend());
		}
		tree(tree<T, Allocator>&& x)
			: node_allocator_type(x.get_allocator())
		{
			create_root();
			if (this != &x)
			{
				node_pointer p = root;
				root = x.root;
				x.root = p;
			}
		}
		tree(tree<T, Allocator>&& x, const Allocator& alloc)
			: node_allocator_type(alloc)
		{
			create_root();
			if (this != &x)
			{
				node_pointer p = root;
				root = x.root;
				x.root = p;
			}
		}
		tree(::std::initializer_list<T> init, const Allocator& alloc = Allocator())
			: node_allocator_type(alloc)
		{
			create_root();
			typename ::std::initializer_list<T>::iterator i = init.begin();
			while (i != init.end())
				push_back(*i++);
		}
		~tree(void) noexcept
		{
			clear();
			destroy_root();
		}
		tree<T, Allocator>& operator=(const tree<T, Allocator>& x)
		{
			if (this != &x)
				copy_child_back(csend(), x.csbegin(), x.csend());
			return (*this);
		}
		tree<T, Allocator>& operator=(tree<T, Allocator>&& x)
		{
			if (this != &x)
			{
				node_pointer p = root;
				root = x.root;
				x.root = p;
			}
			return (*this);
		}
		tree& operator=(::std::initializer_list<T> init)
		{
			clear();
			typename ::std::initializer_list<T>::iterator i = init.begin();
			while (i != init.end())
				push_back(*i++);
			return (*this);
		}
		void assign(const_sibling_iterator first, const_sibling_iterator last)
		{
			clear();
			for (; first != last; ++first)
				push_back(*first);
		}
		void assign(size_type n, const T& value)
		{
			clear();
			for (; n > 0; --n)
				push_back(value);
		}
		void assign(::std::initializer_list<T> init)
		{
			clear();
			typename ::std::initializer_list<T>::iterator i = init.begin();
			while (i != init.end())
				push_back(*i++);
		}

		// iterators:

		iterator begin(void) noexcept
		{
			return iterator(get_first(root));
		}
		const_iterator begin(void) const noexcept
		{
			return const_iterator(get_first(root));
		}
		const_iterator cbegin(void) const noexcept
		{
			return const_iterator(get_first(root));
		}
		iterator end(void) noexcept
		{
			return iterator(root);
		}
		const_iterator end(void) const noexcept
		{
			return const_iterator(root);
		}
		const_iterator cend(void) const noexcept
		{
			return const_iterator(root);
		}
		reverse_iterator rbegin(void) noexcept
		{
			return reverse_iterator(end());
		}
		const_reverse_iterator rbegin(void) const noexcept
		{
			return const_reverse_iterator(end());
		}
		const_reverse_iterator crbegin(void) const noexcept
		{
			return const_reverse_iterator(cend());
		}
		reverse_iterator rend(void) noexcept
		{
			return reverse_iterator(begin());
		}
		const_reverse_iterator rend(void) const noexcept
		{
			return const_reverse_iterator(begin());
		}
		const_reverse_iterator crend(void) const noexcept
		{
			return const_reverse_iterator(cbegin());
		}

		primitive_iterator pbegin(void) noexcept
		{
			return primitive_iterator(get_first(root));
		}
		const_primitive_iterator pbegin(void) const noexcept
		{
			return const_primitive_iterator(get_first(root));
		}
		const_primitive_iterator cpbegin(void) const noexcept
		{
			return const_primitive_iterator(get_first(root));
		}
		primitive_iterator pend(void) noexcept
		{
			return primitive_iterator(root);
		}
		const_primitive_iterator pend(void) const noexcept
		{
			return const_primitive_iterator(root);
		}
		const_primitive_iterator cpend(void) const noexcept
		{
			return const_primitive_iterator(root);
		}
		reverse_primitive_iterator rpbegin(void) noexcept
		{
			return reverse_primitive_iterator(pend());
		}
		const_reverse_primitive_iterator rpbegin(void) const noexcept
		{
			return const_reverse_primitive_iterator(pend());
		}
		const_reverse_primitive_iterator crpbegin(void) const noexcept
		{
			return const_reverse_primitive_iterator(cpend());
		}
		reverse_primitive_iterator rpend(void) noexcept
		{
			return reverse_primitive_iterator(pbegin());
		}
		const_reverse_primitive_iterator rpend(void) const noexcept
		{
			return const_reverse_primitive_iterator(pbegin());
		}
		const_reverse_primitive_iterator crpend(void) const noexcept
		{
			return const_reverse_primitive_iterator(cpbegin());
		}
		template <class Iterator>
		primitive_iterator pbegin(Iterator node) noexcept
		{
			return primitive_iterator(get_first(node.get_pointer()));
		}
		template <class Iterator>
		const_primitive_iterator pbegin(Iterator node) const noexcept
		{
			return const_primitive_iterator(get_first(node.get_pointer()));
		}
		template <class Iterator>
		const_primitive_iterator cpbegin(Iterator node) const noexcept
		{
			return const_primitive_iterator(get_first(node.get_pointer()));
		}
		template <class Iterator>
		primitive_iterator pend(Iterator node) noexcept
		{
			return primitive_iterator(node.get_pointer());
		}
		template <class Iterator>
		const_primitive_iterator pend(Iterator node) const noexcept
		{
			return const_primitive_iterator(node.get_pointer());
		}
		template <class Iterator>
		const_primitive_iterator cpend(Iterator node) const noexcept
		{
			return const_primitive_iterator(node.get_pointer());
		}
		template <class Iterator>
		reverse_primitive_iterator rpbegin(Iterator node) noexcept
		{
			return reverse_primitive_iterator(pend(node));
		}
		template <class Iterator>
		const_reverse_primitive_iterator rpbegin(Iterator node) const noexcept
		{
			return const_reverse_primitive_iterator(pend(node));
		}
		template <class Iterator>
		const_reverse_primitive_iterator crpbegin(Iterator node) const noexcept
		{
			return const_reverse_primitive_iterator(cpend(node));
		}
		template <class Iterator>
		reverse_primitive_iterator rpend(Iterator node) noexcept
		{
			return reverse_primitive_iterator(pbegin(node));
		}
		template <class Iterator>
		const_reverse_primitive_iterator rpend(Iterator node) const noexcept
		{
			return const_reverse_primitive_iterator(pbegin(node));
		}
		template <class Iterator>
		const_reverse_primitive_iterator crpend(Iterator node) const noexcept
		{
			return const_reverse_primitive_iterator(cpbegin(node));
		}

		sibling_iterator sbegin(void) noexcept
		{
			return sibling_iterator(root, get_first(root));
		}
		const_sibling_iterator sbegin(void) const noexcept
		{
			return const_sibling_iterator(root, get_first(root));
		}
		const_sibling_iterator csbegin(void) const noexcept
		{
			return const_sibling_iterator(root, get_first(root));
		}
		sibling_iterator send(void) noexcept
		{
			return sibling_iterator(root, root);
		}
		const_sibling_iterator send(void) const noexcept
		{
			return const_sibling_iterator(root, root);
		}
		const_sibling_iterator csend(void) const noexcept
		{
			return const_sibling_iterator(root, root);
		}
		reverse_sibling_iterator rsbegin(void) noexcept
		{
			return reverse_sibling_iterator(send());
		}
		const_reverse_sibling_iterator rsbegin(void) const noexcept
		{
			return const_reverse_sibling_iterator(send());
		}
		const_reverse_sibling_iterator crsbegin(void) const noexcept
		{
			return const_reverse_sibling_iterator(csend());
		}
		reverse_sibling_iterator rsend(void) noexcept
		{
			return reverse_sibling_iterator(sbegin());
		}
		const_reverse_sibling_iterator rsend(void) const noexcept
		{
			return const_reverse_sibling_iterator(sbegin());
		}
		const_reverse_sibling_iterator crsend(void) const noexcept
		{
			return const_reverse_sibling_iterator(csbegin());
		}
		template <class Iterator>
		sibling_iterator sbegin(Iterator node) noexcept
		{
			return sibling_iterator(node.get_pointer(), get_first(node.get_pointer()));
		}
		template <class Iterator>
		const_sibling_iterator sbegin(Iterator node) const noexcept
		{
			return const_sibling_iterator(node.get_pointer(), get_first(node.get_pointer()));
		}
		template <class Iterator>
		const_sibling_iterator csbegin(Iterator node) const noexcept
		{
			return const_sibling_iterator(node.get_pointer(), get_first(node.get_pointer()));
		}
		template <class Iterator>
		sibling_iterator send(Iterator node) noexcept
		{
			return sibling_iterator(node.get_pointer(), node.get_pointer());
		}
		template <class Iterator>
		const_sibling_iterator send(Iterator node) const noexcept
		{
			return const_sibling_iterator(node.get_pointer(), node.get_pointer());
		}
		template <class Iterator>
		const_sibling_iterator csend(Iterator node) const noexcept
		{
			return const_sibling_iterator(node.get_pointer(), node.get_pointer());
		}
		template <class Iterator>
		reverse_sibling_iterator rsbegin(Iterator node) noexcept
		{
			return reverse_sibling_iterator(send(node));
		}
		template <class Iterator>
		const_reverse_sibling_iterator rsbegin(Iterator node) const noexcept
		{
			return const_reverse_sibling_iterator(send(node));
		}
		template <class Iterator>
		const_reverse_sibling_iterator crsbegin(Iterator node) const noexcept
		{
			return const_reverse_sibling_iterator(csend(node));
		}
		template <class Iterator>
		reverse_sibling_iterator rsend(Iterator node) noexcept
		{
			return reverse_sibling_iterator(sbegin(node));
		}
		template <class Iterator>
		const_reverse_sibling_iterator rsend(Iterator node) const noexcept
		{
			return const_reverse_sibling_iterator(sbegin(node));
		}
		template <class Iterator>
		const_reverse_sibling_iterator crsend(Iterator node) const noexcept
		{
			return const_reverse_sibling_iterator(csbegin(node));
		}

		leaf_iterator lbegin(void) noexcept
		{
			return leaf_iterator(get_leftmost(root));
		}
		const_leaf_iterator lbegin(void) const noexcept
		{
			return const_leaf_iterator(get_leftmost(root));
		}
		const_leaf_iterator clbegin(void) const noexcept
		{
			return const_leaf_iterator(get_leftmost(root));
		}
		leaf_iterator lend(void) noexcept
		{
			return leaf_iterator(root);
		}
		const_leaf_iterator lend(void) const noexcept
		{
			return const_leaf_iterator(root);
		}
		const_leaf_iterator clend(void) const noexcept
		{
			return const_leaf_iterator(root);
		}
		reverse_leaf_iterator rlbegin(void) noexcept
		{
			return reverse_leaf_iterator(lend());
		}
		const_reverse_leaf_iterator rlbegin() const noexcept
		{
			return const_reverse_leaf_iterator(lend());
		}
		const_reverse_leaf_iterator crlbegin() const noexcept
		{
			return const_reverse_leaf_iterator(clend());
		}
		reverse_leaf_iterator rlend(void) noexcept
		{
			return reverse_leaf_iterator(lbegin());
		}
		const_reverse_leaf_iterator rlend(void) const noexcept
		{
			return const_reverse_leaf_iterator(lbegin());
		}
		const_reverse_leaf_iterator crlend(void) const noexcept
		{
			return const_reverse_leaf_iterator(clbegin());
		}

		// capacity:

		bool empty(void) const noexcept
		{
			return (root->first_child == nullptr);
		}

		size_type size(void) const noexcept
		{
			size_type count = 0;
			const_iterator first = cbegin();
			const_iterator last = cend();
			for (; first != last; ++first)
				++count;
			return count;
		}

		// modifiers:

		template <class... Args>
		sibling_iterator emplace_front(Args&&... args)
		{
			return sibling_iterator(prepend_child_node(root, ::std::forward<Args>(args)...));
		}
		template <class... Args>
		sibling_iterator emplace_front(const_sibling_iterator position, Args&&... args)
		{
			return sibling_iterator(prepend_child_node(position.get_parent(), ::std::forward<Args>(args)...));
		}
		sibling_iterator push_front(const value_type& value)
		{
			return sibling_iterator(prepend_child_node(root, value));
		}
		sibling_iterator push_front(const_sibling_iterator position, const value_type& value)
		{
			return sibling_iterator(prepend_child_node(position.get_parent(), value));
		}
		sibling_iterator push_front(value_type&& value)
		{
			return sibling_iterator(prepend_child_node(root, ::std::forward<value_type>(value)));
		}
		sibling_iterator push_front(const_sibling_iterator position, value_type&& value)
		{
			return sibling_iterator(prepend_child_node(position.get_parent(), ::std::forward<value_type>(value)));
		}
		void pop_front(void)
		{
			erase_first_child(root);
		}
		void pop_front(const_sibling_iterator position)
		{
			erase_first_child(position.get_parent());
		}

		template <class... Args>
		sibling_iterator emplace_back(Args&&... args)
		{
			return sibling_iterator(append_child_node(root, ::std::forward<Args>(args)...));
		}
		template <class... Args>
		sibling_iterator emplace_back(const_sibling_iterator position, Args&&... args)
		{
			return sibling_iterator(append_child_node(position.get_parent(), ::std::forward<Args>(args)...));
		}
		sibling_iterator push_back(const value_type& value)
		{
			return sibling_iterator(append_child_node(root, value));
		}
		sibling_iterator push_back(const_sibling_iterator position, const value_type& value)
		{
			return sibling_iterator(append_child_node(position.get_parent(), value));
		}
		sibling_iterator push_back(value_type&& value)
		{
			return sibling_iterator(append_child_node(root, ::std::forward<value_type>(value)));
		}
		sibling_iterator push_back(const_sibling_iterator position, value_type&& value)
		{
			return sibling_iterator(append_child_node(position.get_parent(), ::std::forward<value_type>(value)));
		}
		void pop_back(void)
		{
			erase_last_child(root);
		}
		void pop_back(const_sibling_iterator position)
		{
			erase_last_child(position.get_parent());
		}

		template<class Iterator, class... Args>
		sibling_iterator emplace_child_front(Iterator position, Args&&... args)
		{
			return sibling_iterator(prepend_child_node(position.get_pointer(), ::std::forward<Args>(args)...));
		}
		template<class Iterator>
		sibling_iterator prepend_child(Iterator position, const value_type& value)
		{
			return sibling_iterator(prepend_child_node(position.get_pointer(), value));
		}
		template<class Iterator>
		sibling_iterator prepend_child(Iterator position, value_type&& value)
		{
			return sibling_iterator(prepend_child_node(position.get_pointer(), ::std::forward<value_type>(value)));
		}
		template<class Iterator>
		sibling_iterator prepend_child(Iterator position, size_type n, const value_type& value)
		{
			sibling_iterator i = sbegin(position);
			for (; n > 0; --n)
				prepend_child_node(position.get_pointer(), value);
			return i;
		}
		template<class Iterator, class InputIterator>
		sibling_iterator prepend_child(Iterator position, InputIterator first, InputIterator last)
		{
			sibling_iterator i = sbegin(position);
			for (; first != last; ++first)
				prepend_child_node(position.get_pointer(), *first);
			return i;
		}
		template<class Iterator>
		sibling_iterator prepend_child(Iterator position, ::std::initializer_list<value_type> init)
		{
			return prepend_child(position, init.begin(), init.end());
		}
		template<class Iterator>
		void pop_front(Iterator position)
		{
			erase_first_child(position.get_pointer());
		}

		template <class Iterator, class... Args>
		sibling_iterator emplace_child_back(Iterator position, Args&&... args)
		{
			return sibling_iterator(append_child_node(position.get_pointer(), ::std::forward<Args>(args)...));
		}
		template<class Iterator>
		sibling_iterator append_child(Iterator position, const value_type& value)
		{
			return sibling_iterator(append_child_node(position.get_pointer(), value));
		}
		template<class Iterator>
		sibling_iterator append_child(Iterator position, value_type&& value)
		{
			return sibling_iterator(append_child_node(position.get_pointer(), ::std::forward<value_type>(value)));
		}
		template<class Iterator>
		sibling_iterator append_child(Iterator position, size_type n, const value_type& value)
		{
			sibling_iterator i = send(position);
			for (; n > 0; --n)
				append_child_node(position.get_pointer(), value);
			return i;
		}
		template<class Iterator, class InputIterator>
		sibling_iterator append_child(Iterator position, InputIterator first, InputIterator last)
		{
			sibling_iterator i = send(position);
			for (; first != last; ++first)
				append_child_node(position.get_pointer(), *first);
			return i;
		}
		template<class Iterator>
		sibling_iterator append_child(Iterator position, ::std::initializer_list<value_type> init)
		{
			return append_child(position, init.begin(), init.end());
		}
		template<class Iterator>
		void pop_back(Iterator position)
		{
			erase_last_child(position.get_pointer());
		}

		template <class... Args>
		sibling_iterator emplace(const_sibling_iterator position, Args&&... args)
		{
			return sibling_iterator(insert_node(position.get_parent(), position.get_pointer(), ::std::forward<Args>(args)...));
		}
		sibling_iterator insert(const_sibling_iterator position, const value_type& value)
		{
			return sibling_iterator(insert_node(position.get_parent(), position.get_pointer(), value));
		}
		sibling_iterator insert(const_sibling_iterator position, value_type&& value)
		{
			return sibling_iterator(insert_node(position.get_parent(), position.get_pointer(), ::std::forward<value_type>(value)));
		}
		sibling_iterator insert(const_sibling_iterator position, size_type n, const value_type& value)
		{
			sibling_iterator i(position.get_parent(), position.get_pointer());
			--i;
			for (; n > 0; --n)
				insert_node(position.get_parent(), position.get_pointer(), value);
			++i;
			return i;
		}
		template <class InputIterator>
		sibling_iterator insert(const_sibling_iterator position, InputIterator first, InputIterator last)
		{
			sibling_iterator i(position.get_parent(), position.get_pointer());
			--i;
			for (; first != last; ++first)
				insert_node(position.get_parent(), position.get_pointer(), *first);
			++i;
			return i;
		}
		sibling_iterator insert(const_sibling_iterator position, ::std::initializer_list<value_type> init)
		{
			return insert(position, init.begin(), init.end());
		}

		template <class Iterator, class InputIterator>
		Iterator copy_child_front(Iterator position, InputIterator node)
		{
			node_pointer p = prepend_child_node(position.get_pointer(), *node);
			copy_children(p, node.get_pointer());
			return Iterator(p);
		}
		template <class Iterator>
		reverse_sibling_iterator copy_child_front(Iterator position, const_reverse_sibling_iterator first, const_reverse_sibling_iterator last)
		{
			reverse_sibling_iterator i = rsend(position);
			--i;
			for (; first != last; ++first)
			{
				node_pointer p = prepend_child_node(position, *first);
				copy_children(p, first.get_pointer());
			}
			++i;
			return i;
		}

		template <class Iterator, class InputIterator>
		Iterator copy_child_back(Iterator position, InputIterator node)
		{
			node_pointer p = append_child_node(position.get_pointer(), *node);
			copy_children(p, node.get_pointer());
			return Iterator(p);
		}
		template <class Iterator>
		sibling_iterator copy_child_back(Iterator position, const_sibling_iterator first, const_sibling_iterator last)
		{
			sibling_iterator i = send(position);
			--i;
			for (; first != last; ++first)
			{
				node_pointer p = append_child_node(position.get_pointer(), *first);
				copy_children(p, first.get_pointer());
			}
			++i;
			return i;
		}

		template <class Iterator, class InputIterator>
		sibling_iterator copy_node(Iterator position, const_sibling_iterator first, const_sibling_iterator last)
		{
			sibling_iterator i(position.get_parent(), position.get_pointer());
			--i;
			for (; first != last; ++first)
			{
				node_pointer p = insert_node(position.get_parent(), position.get_pointer(), *first);
				copy_children(p, first.get_pointer());
			}
			++i;
			return i;
		}
		sibling_iterator copy_node(const_sibling_iterator position, const_sibling_iterator first, const_sibling_iterator last)
		{
			sibling_iterator i(position.get_parent(), position.get_pointer());
			--i;
			for (; first != last; ++first)
			{
				node_pointer p = insert_node(position.get_parent(), position.get_pointer(), *first);
				copy_children(p, first.get_pointer());
			}
			++i;
			return i;
		}

		template <class Iterator>
		sibling_iterator erase(Iterator node)
		{
			sibling_iterator i(node.get_parent(), node.get_pointer());
			++i;
			erase_node(node.get_pointer());
			return i;
		}
		sibling_iterator erase(sibling_iterator first, sibling_iterator last)
		{
			if (first == sbegin() && last == send())
				clear();
			else
				for (; first != last; ++first)
					erase_node(first.get_pointer());
			return sibling_iterator(last.get_pointer());
		}

		template <class Iterator>
		void remove_children(Iterator node)
		{
			erase_children(node.get_pointer());
		}

		void swap(tree<T, Allocator>& rhs) noexcept
		{
			if (this != &rhs)
			{
				node_pointer p = root;
				root = rhs.root;
				rhs.root = p;
			}
		}

		void clear(void) noexcept
		{
			node_pointer next = nullptr;
			node_pointer cur = root->first_child;
			while (cur != nullptr)
			{
				next = cur->next_sibling;
				erase_children(cur);
				this->destroy_node(cur);
				cur = next;
			}
			root->first_child = nullptr;
			root->last_child = nullptr;
		}

		// Operations:

		template <class InputIterator>
		void splice(const_sibling_iterator position, InputIterator node)
		{
			splice_node(position.get_parent(), position.get_pointer(), node.get_pointer());
		}
		void splice(const_sibling_iterator position, const_sibling_iterator first, const_sibling_iterator last)
		{
			splice_node(position.get_parent(), position.get_pointer(), first.get_pointer(), last.get_pointer());
		}

		template <class Iterator, class InputIterator>
		void merge(Iterator position, InputIterator node)
		{
			merge_node(position.get_pointer(), node.get_pointer(), ::std::less<T>());
		}

		template <class Iterator, class InputIterator, class Compare>
		void merge(Iterator position, InputIterator node, Compare comp)
		{
			merge_node(position.get_pointer(), node.get_pointer(), comp);
		}

		void sort(bool deep)
		{
			sort_node(root, deep, ::std::less<T>());
		}
		template <class Compare>
		void sort(bool deep, Compare comp)
		{
			sort_node(root, deep, comp);
		}
		template <class Iterator>
		void sort(Iterator node, bool deep)
		{
			sort_node(node.get_pointer(), deep, ::std::less<T>());
		}
		template <class Iterator, class Compare>
		void sort(Iterator node, bool deep, Compare comp)
		{
			sort_node(node.get_pointer(), deep, comp);
		}
	private:
		node_pointer get_first(const node_pointer parent) const noexcept
		{
			return (parent->first_child != nullptr ? parent->first_child : parent);
		}
		node_pointer get_last(const node_pointer parent) const noexcept
		{
			return (parent->last_child != nullptr ? parent->last_child : parent);
		}

		node_pointer get_leftmost(const node_pointer parent) const noexcept
		{
			node_pointer leftmost = parent;
			while (leftmost->first_child)
				leftmost = leftmost->first_child;
			return leftmost;
		}
		node_pointer get_rightmost(const node_pointer parent) const noexcept
		{
			node_pointer rightmost = parent;
			while (rightmost->last_child)
				rightmost = rightmost->last_child;
			return rightmost;
		}

		void create_root(void)
		{
			root = this->create_node(value_type());
			root->parent = nullptr;
			root->prev_sibling = root;
			root->next_sibling = root;
			root->first_child = nullptr;
			root->last_child = nullptr;
		}

		void destroy_root(void)
		{
			this->destroy_node(root);
		}

		template<class ...Args>
		node_pointer append_child_node(const node_pointer parent, Args&&... args)
		{
			node_pointer node = this->create_node(::std::forward<Args>(args)...);
			node->parent = parent;
			node->prev_sibling = parent->last_child;
			node->next_sibling = nullptr;
			node->first_child = nullptr;
			node->last_child = nullptr;
			if (parent->last_child != nullptr)
				parent->last_child->next_sibling = node;
			else
				parent->first_child = node;
			parent->last_child = node;
			return node;
		}

		template<class ...Args>
		node_pointer prepend_child_node(const node_pointer parent, Args&&... args)
		{
			node_pointer node = this->create_node(::std::forward<Args>(args)...);
			node->parent = parent;
			node->prev_sibling = nullptr;
			node->next_sibling = parent->first_child;
			node->first_child = nullptr;
			node->last_child = nullptr;
			if (parent->first_child != nullptr)
				parent->first_child->prev_sibling = node;
			else
				parent->last_child = node;
			parent->first_child = node;
			return node;
		}

		template<class ...Args>
		node_pointer insert_node(const node_pointer parent, const node_pointer position, Args&&... args)
		{
			node_pointer node = this->create_node(::std::forward<Args>(args)...);
			node->parent = parent;
			if (position == parent)
			{
				node->prev_sibling = parent->last_child;
				node->next_sibling = nullptr;
				node->first_child = nullptr;
				node->last_child = nullptr;
				if (parent->last_child != nullptr)
					parent->last_child->next_sibling = node;
				else
					parent->first_child = node;
				parent->last_child = node;
			}
			else
			{
				node->prev_sibling = position->prev_sibling;
				node->next_sibling = position;
				node->first_child = nullptr;
				node->last_child = nullptr;
				if (position->prev_sibling != nullptr)
					position->prev_sibling->next_sibling = node;
				else
					parent->first_child = node;
				position->prev_sibling = node;
			}
			return node;
		}

		void copy_children(node_pointer dst, const node_pointer src)
		{
			node_pointer cur = src;
			do
			{
				if (cur->first_child != nullptr)
				{
					cur = cur->first_child;
					dst = append_child_node(dst, cur->data);
				}
				else
				{
					if (cur->next_sibling != nullptr)
					{
						cur = cur->next_sibling;
						append_child_node(dst->parent, cur->data);
					}
					else
					{
						cur = cur->parent;
						dst = dst->parent;
					}
				}
			} while (cur != src);
		}

		void erase_children(const node_pointer parent)
		{
			node_pointer cur = parent;
			node_pointer next = nullptr;
			do
			{
				while (cur->first_child != nullptr)
					cur = cur->first_child;
				if (cur->next_sibling != nullptr)
				{
					next = cur->next_sibling;
					this->destroy_node(cur);
					cur = next;
				}
				else
				{
					next = cur->parent;
					this->destroy_node(cur);
					cur = next;
				}
			} while (cur != parent);
			parent->first_child = nullptr;
			parent->last_child = nullptr;
		}

		void erase_first_child(const node_pointer parent)
		{
			node_pointer child = parent->first_child;
			if (child != nullptr)
			{
				parent->first_child = child->next_sibling;
				if (child->next_sibling != nullptr)
					child->next_sibling->prev_sibling = nullptr;
				else
					parent->last_child = nullptr;
				erase_children(child);
				this->destroy_node(child);
			}
		}

		void erase_last_child(const node_pointer parent)
		{
			node_pointer child = parent->last_child;
			if (child != nullptr)
			{
				parent->last_child = child->prev_sibling;
				if (child->prev_sibling != nullptr)
					child->prev_sibling->next_sibling = nullptr;
				else
					parent->first_child = nullptr;
				erase_children(child);
				this->destroy_node(child);
			}
		}

		void erase_node(const node_pointer position)
		{
			if (position->prev_sibling != nullptr)
				position->prev_sibling->next_sibling = position->next_sibling;
			else
				position->parent->first_child = position->next_sibling;
			if (position->next_sibling != nullptr)
				position->next_sibling->prev_sibling = position->prev_sibling;
			else
				position->parent->last_child = position->prev_sibling;
			erase_children(position);
			this->destroy_node(position);
		}

		void transfer_node(const node_pointer parent, const node_pointer position, const node_pointer node)
		{
			if (position != node && position != node->next_sibling)
			{
				if (node->prev_sibling != nullptr)
					node->prev_sibling->next_sibling = node->next_sibling;
				else
					node->parent->first_child = node->next_sibling;
				if (node->next_sibling != nullptr)
					node->next_sibling->prev_sibling = node->prev_sibling;
				else
					node->parent->last_child = node->prev_sibling;
				if (position == parent)
				{
					node->prev_sibling = parent->last_child;
					node->next_sibling = nullptr;
					if (parent->last_child != nullptr)
						parent->last_child->next_sibling = node;
					else
						parent->first_child = node;
					parent->last_child = node;
				}
				else
				{
					node->prev_sibling = position->prev_sibling;
					node->next_sibling = position;
					if (position->prev_sibling != nullptr)
						position->prev_sibling->next_sibling = node;
					else
						parent->first_child = node;
					position->prev_sibling = node;
				}
			}
		}
		void transfer_node(const node_pointer parent, const node_pointer position, const node_pointer first, const node_pointer last)
		{
			node_pointer node = (last != first->parent) ? last->prev_sibling : last->last_child;
			if (position != first && position != node->next_sibling)
			{
				if (first->prev_sibling != nullptr)
					first->prev_sibling->next_sibling = node->next_sibling;
				else
					first->parent->first_child = node->next_sibling;
				if (node->next_sibling != nullptr)
					node->next_sibling->prev_sibling = first->prev_sibling;
				else
					first->parent->last_child = first->prev_sibling;
				if (position == parent)
				{
					first->prev_sibling = parent->last_child;
					node->next_sibling = nullptr;
					if (parent->last_child != nullptr)
						parent->last_child->next_sibling = first;
					else
						parent->first_child = first;
					parent->last_child = node;
				}
				else
				{
					first->prev_sibling = position->prev_sibling;
					node->next_sibling = position;
					if (position->prev_sibling != nullptr)
						position->prev_sibling->next_sibling = first;
					else
						parent->first_child = first;
					position->prev_sibling = node;
				}
			}
		}

		void splice_node(const node_pointer parent, const node_pointer position, const node_pointer node)
		{
			transfer_node(parent, position, node);
			node->parent = parent;
		}
		void splice_node(const node_pointer parent, const node_pointer position, const node_pointer first, const node_pointer last)
		{
			transfer_node(parent, position, first, last);
			if (parent != first->parent)
				for (node_pointer p = first; p != nullptr && p != last; p = p->next_sibling)
					p->parent = parent;
		}

		template <class Compare>
		void merge_node(const node_pointer position, const node_pointer node, Compare comp)
		{
			if (position != node)
			{
				node_pointer dst = position->first_child;
				node_pointer src = node->first_child;

				while (dst != nullptr && src != nullptr)
				{
					if (comp(src->data, dst->data))
					{
						node_pointer next = src->next_sibling;
						splice_node(position, dst, src);
						src = next;
					}
					else
						dst = dst->next_sibling;
				}
				if (src != nullptr)
					splice_node(position, position, src, node);
			}
		}

		template <class Compare>
		void sort_node(const node_pointer parent, bool deep, Compare comp)
		{
			tree<T, Allocator> carry;
			tree<T, Allocator> counter[64];
			if (deep)
			{
				node_pointer node = parent;
				do
				{
					if (node->first_child != nullptr)
						node = node->first_child;
					else
					{
						while (node != parent && node->next_sibling == nullptr)
						{
							node = node->parent;
							sort_single_node(carry, counter, node, comp);
						}
						node = node->next_sibling;
					}
				} while (node != parent);
			}
			else
				sort_single_node(carry, counter, parent, comp);
		}
		template <class Compare>
		void sort_single_node(tree<T, Allocator>&carry, tree<T, Allocator>* counter, const node_pointer parent, Compare comp)
		{
			if (parent->first_child == nullptr || parent->first_child == parent->last_child)
				return;
			int fill = 0;
			while (parent->first_child != nullptr)
			{
				int i = 0;
				carry.splice_node(carry.root, carry.root, parent->first_child);
				while (i < fill && !counter[i].empty())
				{
					counter[i].merge_node(counter[i].root, carry.root, comp);
					carry.swap(counter[i++]);
				}
				carry.swap(counter[i]);
				if (i == fill)
					++fill;
			}
			for (int i = 1; i < fill; ++i)
				counter[i].merge_node(counter[i].root, counter[i - 1].root, comp);
			splice_node(parent, parent, counter[fill - 1].root->first_child, counter[fill - 1].root);
		}
	private:
		node_pointer root;
	};

} // namespace core

#endif
