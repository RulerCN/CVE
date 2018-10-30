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

#ifndef __NN_MNIST_DATA_H__
#define __NN_MNIST_DATA_H__

#include <iostream>
#include <fstream>
#include "sample_set.h"

namespace nn
{
	// The MNIST database of handwritten digits
	// Data from: http://yann.lecun.com/exdb/mnist/

	static constexpr unsigned int mnist_identity_label = 0x01080000; /* 2049 */
	static constexpr unsigned int mnist_identity_image = 0x03080000; /* 2051 */

#	pragma pack(push, 4)

	// Struct mnist_image_header
	struct mnist_image_header
	{
		unsigned int         magic;    /* magic number */
		unsigned int         images;   /* number of images */
		unsigned int         height;   /* number of rows */
		unsigned int         width;    /* number of width */
	//	unsigned char        pixel;    /* pixel value are 0 to 255. */
	};

	// Struct mnist_image_header
	struct mnist_label_header
	{
		unsigned int         magic;    /* magic number */
		unsigned int         items;    /* number of images */
	//	unsigned char        label;    /* labels value are 0 to 9. */
	};

#	pragma pack(pop)

	// Class template mnist_data
	template <class Allocator = ::core::allocator<void> >
	class mnist_data : public sample_set<unsigned char, unsigned char, Allocator>
	{
	public:
		// construct/copy/destroy:

		mnist_data(void)
		{}
		mnist_data(::std::ifstream &image, ::std::ifstream &label)
		{
			load(image, label);
		}
		mnist_data(const char* image, const char* label)
		{
			load(image, label);
		}
		mnist_data(const ::std::string& image, const ::std::string& label)
		{
			load(image, label);
		}
		// Not recommended. (Olny provided for Microsoft Visual Studio)
#		if defined(_MSC_VER)
		mnist_data(const wchar_t* image, const wchar_t* label)
		{
			load(image, label);
		}
#		endif

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
			this->data.assign(batch, rows, columns, dimension);
			input.read(reinterpret_cast<char*>(this->data.data()), this->data.size());
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
			this->labels.assign(length, dimension);
			input.read(reinterpret_cast<char*>(this->labels.data()), this->labels.size());
			if (!input.good())
			{
				labels.clear();
				return false;
			}
			return true;
		}
	};

} // namespace nn

#endif
