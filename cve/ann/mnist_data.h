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

#ifndef __ANN_MNIST_DATA_H__
#define __ANN_MNIST_DATA_H__

#include <iostream>
#include <fstream>
#include "sample_base.h"

namespace ann
{
	// The MNIST database of handwritten digits
	// Data from: http://yann.lecun.com/exdb/mnist/

	static constexpr unsigned int mnist_identity_label = 0x01080000; /* 2049 */
	static constexpr unsigned int mnist_identity_image = 0x03080000; /* 2051 */

#	pragma pack(push, 4)
	// Struct mnist_image_header
	struct mnist_label_header
	{
		unsigned int         magic;    /* magic number */
		unsigned int         items;    /* number of images */
	//	unsigned char        label;    /* labels value are 0 to 9. */
	};

	// Struct mnist_image_header
	struct mnist_image_header
	{
		unsigned int         magic;    /* magic number */
		unsigned int         images;   /* number of images */
		unsigned int         height;   /* number of rows */
		unsigned int         width;    /* number of width */
	//	unsigned char        pixel;    /* pixel value are 0 to 255. */
	};
#	pragma pack(pop)


	// Template class mnist_data
	template <class Allocator = ::core::allocator<unsigned char> >
	class mnist_data : public sample_base<Allocator>
	{
	public:
		// construct/copy/destroy:

		mnist_data(void)
		{}

		// Load MNIST data set

		bool load(::std::ifstream &image, ::std::ifstream &label)
		{
			if (image.is_open())
			{
				load_images(image);
				image.close();
			}
			if (label.is_open())
			{
				load_labels(label);
				label.close();
			}
			if (images.empty() || labels.empty() || images.batch() != labels.length())
			{
				images.clear();
				labels.clear();
				return false;
			}
			this->assign(labels.length());
			return true;
		}

		bool load(const char* image, const char* label)
		{
			::std::ifstream image_stream(image, ::std::ios::in | ::std::ios::binary);
			::std::ifstream label_stream(label, ::std::ios::in | ::std::ios::binary);
			return load(image_stream, label_stream);
		}

		bool load(const ::std::string& image, const ::std::string& label)
		{
			::std::ifstream image_stream(image, ::std::ios::in | ::std::ios::binary);
			::std::ifstream label_stream(label, ::std::ios::in | ::std::ios::binary);
			return load(image_stream, label_stream);
		}

		// Not recommended. (Olny provided for Microsoft Visual Studio)
#		if defined(_MSC_VER)
		bool load(const wchar_t* image, const wchar_t* label)
		{
			::std::ifstream image_stream(image, ::std::ios::in | ::std::ios::binary);
			::std::ifstream label_stream(label, ::std::ios::in | ::std::ios::binary);
			return load(image_stream, label_stream);
		}

		bool load(const ::std::wstring& image, const ::std::wstring& label)
		{
			::std::ifstream image_stream(image, ::std::ios::in | ::std::ios::binary);
			::std::ifstream label_stream(label, ::std::ios::in | ::std::ios::binary);
			return load(image_stream, label_stream);

		}
#		endif

		// Return a batch of data

		template<class T1, class T2, class A1, class A2>
		void next_batch(::core::tensor<T1, A1> &batch_images, ::core::vector<T2, A2> &batch_labels)
		{
			if (batch_images.batch() != batch_labels.length())
				throw ::std::invalid_argument(::core::sample_unequal_number);

			size_t batch_size = batch_images.batch();
			for (size_t i = 0; i < batch_size; ++i)
			{
				size_t index = this->next();
				::core::get_element(batch_images.at(i), images, index);
				::core::get_element(batch_labels.at(i), labels, index);
			}
		}

		template<class T2, class A1, class A2>
		void next_batch(::core::tensor<float, A1> &batch_images, ::core::vector<T2, A2> &batch_labels, float scale)
		{
			if (batch_images.batch() != batch_labels.length())
				throw ::std::invalid_argument(::core::sample_unequal_number);

			size_t batch_size = batch_images.batch();
			for (size_t i = 0; i < batch_size; ++i)
			{
				size_t index = this->next();
				::core::get_element(batch_images.at(i), images, index, scale);
				::core::get_element(batch_labels.at(i), labels, index);
			}
		}

		template<class T2, class A1, class A2>
		void next_batch(::core::tensor<double, A1> &batch_images, ::core::vector<T2, A2> &batch_labels, double scale)
		{
			if (batch_images.batch() != batch_labels.length())
				throw ::std::invalid_argument(::core::sample_unequal_number);

			size_t batch_size = batch_images.batch();
			for (size_t i = 0; i < batch_size; ++i)
			{
				size_t index = this->next();
				::core::get_element(batch_images.at(i), images, index, scale);
				::core::get_element(batch_labels.at(i), labels, index);
			}
		}
	private:
		unsigned int reverse_uint32(unsigned int number) const
		{
			return (((number & 0x000000ffU) << 24)
				| ((number & 0x0000ff00U) << 8)
				| ((number & 0x00ff0000U) >> 8)
				| ((number & 0xff000000U) >> 24));
		}
		// Load images
		bool load_images(::std::istream &input)
		{
			mnist_image_header header = { 0 };
			input.read(reinterpret_cast<char*>(&header), sizeof(mnist_image_header));
			if (!input.good())
				return false;
			if (header.magic != mnist_identity_image)
				return false;
			size_t batch     = static_cast<size_t>(reverse_uint32(header.images));
			size_t rows      = static_cast<size_t>(reverse_uint32(header.height));
			size_t columns   = static_cast<size_t>(reverse_uint32(header.width));
			size_t dimension = 1;
			images.assign(batch, rows, columns, dimension);
			input.read(reinterpret_cast<char*>(images.data()), images.size());
			if (!input.good())
			{
				images.clear();
				return false;
			}
			return true;
		}
		// Load labels
		bool load_labels(::std::istream &input)
		{
			mnist_label_header header = { 0 };
			input.read(reinterpret_cast<char*>(&header), sizeof(mnist_label_header));
			if (!input.good())
				return false;
			if (header.magic != mnist_identity_label)
				return false;
			size_t length = static_cast<size_t>(reverse_uint32(header.items));
			size_t dimension = 1;
			labels.assign(length, dimension);
			input.read(reinterpret_cast<char*>(labels.data()), labels.size());
			if (!input.good())
			{
				labels.clear();
				return false;
			}
			return true;
		}
	public:
		::core::tensor<unsigned char, Allocator> images;
		::core::vector<unsigned char, Allocator> labels;
	};

} // namespace ann

#endif
