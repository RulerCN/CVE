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

#ifndef __IMAGE_BITMAP_H__
#define __IMAGE_BITMAP_H__

#include <iostream>
#include <fstream>
#include "../core/matrix.h"

namespace img
{
	static constexpr unsigned short bitmap_identity_icon   = 0x4349; /* 'IC' */
	static constexpr unsigned short bitmap_identity_bitmap = 0x4d42; /* 'BM' */
	static constexpr unsigned short bitmap_identity_cursor = 0x5450; /* 'PT' */

#	pragma pack(push, 2)

	// Struct bitmap_file_header
	struct bitmap_file_header
	{
		unsigned short       type;             /* ID field BM (0x4d42) */
		unsigned int         size;             /* Size of the BMP file */
		unsigned short       reserved1;        /* Application specific */
		unsigned short       reserved2;        /* Application specific */
		unsigned int         offset;           /* Offset to bitmap data */
	};

	// Struct bitmap_dib_header
	struct bitmap_dib_header
	{
		unsigned int         size;             /* Size of the DIB header */
		int                  width;            /* Width of the bitmap in pixels */
		int                  height;           /* Height of the bitmap in pixels */
		unsigned short       planes;           /* Number of color planes */
		unsigned short       bits;             /* Number of bits per pixel */
		unsigned int         compression;      /* Compression type */
		unsigned int         image_size;       /* Size of the raw bitmap data */
		int                  resolution_x;     /* pixels/meter horizontal */
		int                  resolution_y;     /* pixels/meter vertical */
		unsigned int         colors;           /* Number of colors in the palette */
		unsigned int         important_colors; /* Important colors */
	};

	// Struct bitmap_palette_entry
	struct bitmap_palette_entry
	{
		unsigned char        blue;
		unsigned char        green;
		unsigned char        red;
		unsigned char        reserved;
	};

	// Struct bitmap_header
	struct bitmap_header
	{
		bitmap_file_header   file;
		bitmap_dib_header    dib;
	};

	// Struct bitmap_palette
	struct bitmap_palette
	{
		unsigned int         number;
		bitmap_palette_entry color[256];
	};

#	pragma pack(pop)

	// Class bitmap
	class bitmap
	{
	public:
		// Bitmap encode

		template <class Allocator>
		static bool encode(std::ostream &output, const ::core::matrix<unsigned char, Allocator> &input, float dpi = 72)
		{
			// Write header information
			if (!write_header(output, input, dpi))
				return false;
			// Write bitmap data
			if (!write_data(output, input))
				return false;
			output.flush();
			return true;
		}

		template <class Allocator>
		static bool encode(const char* file_name, const ::core::matrix<unsigned char, Allocator> &input, float dpi = 72)
		{
			bool rst = false;
			std::ofstream output(file_name, std::ios::out | std::ios::binary);
			if (output.is_open())
			{
				rst = encode(output, input, dpi);
				output.close();
			}
			return rst;
		}

		template <class Allocator>
		static bool encode(const std::string& file_name, const ::core::matrix<unsigned char, Allocator> &input, float dpi = 72)
		{
			bool rst = false;
			std::ofstream output(file_name, std::ios::out | std::ios::binary);
			if (output.is_open())
			{
				rst = encode(output, input, dpi);
				output.close();
			}
			return rst;
		}

		// Not recommended. (Olny provided for Microsoft Visual Studio)
#		if defined(_MSC_VER)
		template <class Allocator>
		static bool encode(const wchar_t* file_name, const ::core::matrix<unsigned char, Allocator> &input, float dpi = 72)
		{
			bool rst = false;
			std::ofstream output(file_name, std::ios::out | std::ios::binary);
			if (output.is_open())
			{
				rst = encode(output, input, dpi);
				output.close();
			}
			return rst;
		}

		template <class Allocator>
		static bool encode(const std::wstring& file_name, const ::core::matrix<unsigned char, Allocator> &input, float dpi = 72)
		{
			bool rst = false;
			std::ofstream output(file_name, std::ios::out | std::ios::binary);
			if (output.is_open())
			{
				rst = encode(output, input, dpi);
				output.close();
			}
			return rst;
		}
#		endif

		// Bitmap decode

		template <class Allocator>
		static bool decode(std::istream &input, ::core::matrix<unsigned char, Allocator> &output)
		{
			bitmap_header header = { 0 };
			bitmap_palette palette = { 0 };
			// Read header information
			if (!read_header(input, header))
				return false;
			// Read palette information
			input.seekg(static_cast<std::streamoff>(header.dib.size), std::ios::beg);
			if (!read_palette(input, header, palette))
				return false;
			// Read bitmap data
			input.seekg(static_cast<std::streamoff>(header.file.offset), std::ios::beg);
			return read_data(input, header, palette, output);
		}

		template <class Allocator>
		static bool decode(const char* file_name, ::core::matrix<unsigned char, Allocator> &output)
		{
			bool rst = false;
			std::ifstream input(file_name, std::ios::in | std::ios::binary);
			if (input.is_open())
			{
				rst = decode(input, output);
				input.close();
			}
			return rst;
		}

		template <class Allocator>
		static bool decode(const std::string& file_name, ::core::matrix<unsigned char, Allocator> &output)
		{
			bool rst = false;
			std::ifstream input(file_name, std::ios::in | std::ios::binary);
			if (input.is_open())
			{
				rst = decode(input, output);
				input.close();
			}
			return rst;
		}

		// Not recommended. (Olny provided for Microsoft Visual Studio)
#		if defined(_MSC_VER)
		template <class Allocator>
		static bool decode(const wchar_t* file_name, ::core::matrix<unsigned char, Allocator> &output)
		{
			bool rst = false;
			std::ifstream input(file_name, std::ios::in | std::ios::binary);
			if (input.is_open())
			{
				rst = decode(input, output);
				input.close();
			}
			return rst;
		}

		template <class Allocator>
		static bool decode(const std::wstring& file_name, ::core::matrix<unsigned char, Allocator> &output)
		{
			bool rst = false;
			std::ifstream input(file_name, std::ios::in | std::ios::binary);
			if (input.is_open())
			{
				rst = decode(input, output);
				input.close();
			}
			return rst;
		}
#		endif
	private:
		// Write header information
		template <class Allocator>
		static bool write_header(std::ostream &output, const ::core::matrix<unsigned char, Allocator> &input, float dpi)
		{
			bitmap_header header;
			size_t stride = (input.row_size() + 3) & ~3;
			// file header
			header.file.type = bitmap_identity_bitmap;
			header.file.size = static_cast<unsigned int>(sizeof(bitmap_header) + stride * input.rows());
			header.file.reserved1 = 0;
			header.file.reserved2 = 0;
			header.file.offset = sizeof(bitmap_header);
			// DIB header
			header.dib.size = sizeof(bitmap_dib_header);
			header.dib.width = static_cast<int>(input.columns());
			header.dib.height = -static_cast<int>(input.rows());
			header.dib.planes = 1;
			header.dib.bits = static_cast<unsigned short>(input.dimension() << 3);
			header.dib.compression = 0;
			header.dib.image_size = static_cast<unsigned int>(stride * input.rows());
			header.dib.resolution_x = static_cast<int>(dpi * 39.37f + 0.5f);
			header.dib.resolution_y = static_cast<int>(dpi * 39.37f + 0.5f);
			header.dib.colors = (input.dimension() == 1) ? 256 : 0;
			header.dib.important_colors = 0;
			// Write file header
			output.write(reinterpret_cast<char*>(&header.file), sizeof(bitmap_file_header));
			// Write DIB header
			output.write(reinterpret_cast<char*>(&header.dib), sizeof(bitmap_dib_header));
			return true;
		}

		// Write gray palette information
		static bool write_palette(std::ostream &output)
		{
			bitmap_palette palette;
			palette.number = 256;
			for (unsigned int i = 0; i < palette.number; ++i)
			{
				palette.color[i].blue = static_cast<unsigned char>(i);
				palette.color[i].green = static_cast<unsigned char>(i);
				palette.color[i].red = static_cast<unsigned char>(i);
				palette.color[i].reserved = 0;
			}
			output.write(reinterpret_cast<const char*>(&palette), palette.number * sizeof(bitmap_palette_entry));
			return true;
		}

		// Write bitmap data
		template <class Allocator>
		static bool write_data(std::ostream &output, const ::core::matrix<unsigned char, Allocator> &input)
		{
			bool rst = false;
			const char zero[3] = { 0, 0, 0 };
			size_t row_size = input.row_size();
			size_t padding_size = (4 - (row_size & 3)) & 3;
			switch (input.dimension())
			{
			case 1:
				// Write gray palette information
				write_palette(output);
				// Write color data
				for (auto vi = input.cvbegin(); vi != input.cvend(); ++vi)
				{
					output.write(reinterpret_cast<const char*>(vi.operator->()), row_size);
					output.write(zero, padding_size);
				}
				rst = true;
				break;
			case 3:
			case 4:
				// Write color data
				for (auto vi = input.cvbegin(); vi != input.cvend(); ++vi)
				{
					output.write(reinterpret_cast<const char*>(vi.operator->()), row_size);
					output.write(zero, padding_size);
				}
				rst = true;
				break;
			}
			return rst;
		}

		// Read header information
		static bool read_header(std::istream &input, bitmap_header &header)
		{
			bool rst = false;
			// Read file header
			input.read(reinterpret_cast<char*>(&header.file), sizeof(bitmap_file_header));
			if (input.good())
			{
				switch (header.file.type)
				{
				case bitmap_identity_bitmap:
					// Read DIB header
					input.read(reinterpret_cast<char*>(&header.dib), sizeof(bitmap_dib_header));
					rst = input.good();
					break;
				}
			}
			return rst;
		}

		// Read palette information
		static bool read_palette(std::istream &input, const bitmap_header &header, bitmap_palette &palette)
		{
			bool rst = false;
			if (header.dib.colors == 0)
			{
				palette.number = 0;
				rst = true;
			}
			else if (header.dib.colors > 0 && header.dib.colors <= 256)
			{
				palette.number = header.dib.colors;
				input.read(reinterpret_cast<char*>(&palette), palette.number * sizeof(bitmap_palette_entry));
				rst = input.good();
			}
			return rst;
		}

		// Read bitmap data
		template <class Allocator>
		static bool read_data(std::istream &input, const bitmap_header &header, const bitmap_palette &palette, ::core::matrix<unsigned char, Allocator> &output)
		{
			bool rst = true;
			int channel = (palette.number > 0) ? 3 : (header.dib.bits >> 3);
			int width = header.dib.width;
			int height = header.dib.height;
			int stride = ((header.dib.width * header.dib.bits + 31) & ~31) >> 3;
			// Create matrix
			output.assign(height, width, channel);
			// Read bitmap data
			switch (header.dib.compression)
			{
			case 0:
				switch (header.dib.bits)
				{
				case 1:
					// Read 1-bit color image
					rst = read_data_1bit(input, width, height, stride, palette, output);
					break;
				case 4:
					// Read 4-bit color image
					rst = read_data_4bit(input, width, height, stride, palette, output);
					break;
				case 8:
					// Read 8-bit color image
					rst = read_data_8bit(input, width, height, stride, palette, output);
					break;
				case 24:
					// Read 24-bit color image
					rst = read_data_24bit(input, width, height, stride, output);
					break;
				case 32:
					// Read 32-bit color image
					rst = read_data_32bit(input, width, height, stride, output);
					break;
				}
				break;
			case 1:
				break;
			case 4:
				break;
			}
			return rst;
		}

		// Read 1-bit color image
		template <class Allocator>
		static bool read_data_1bit(std::istream &input, int width, int height, int stride, const bitmap_palette &palette, ::core::matrix<unsigned char, Allocator> &output)
		{
			if (palette.number == 0)
			{
				if (height < 0)
				{
					for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
					{
						char value = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int x = 0; x < width;)
						{
							input.get(value);
							dst[x++] = (value & 0x80);
							dst[x++] = (value & 0x40) << 1;
							dst[x++] = (value & 0x20) << 2;
							dst[x++] = (value & 0x10) << 3;
							dst[x++] = (value & 0x08) << 4;
							dst[x++] = (value & 0x04) << 5;
							dst[x++] = (value & 0x02) << 6;
							dst[x++] = (value & 0x01) << 7;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
				else
				{
					for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
					{
						char value = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int x = 0; x < width;)
						{
							input.get(value);
							dst[x++] = (value & 0x80);
							dst[x++] = (value & 0x40) << 1;
							dst[x++] = (value & 0x20) << 2;
							dst[x++] = (value & 0x10) << 3;
							dst[x++] = (value & 0x08) << 4;
							dst[x++] = (value & 0x04) << 5;
							dst[x++] = (value & 0x02) << 6;
							dst[x++] = (value & 0x01) << 7;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
			}
			else
			{
				if (height < 0)
				{
					for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
					{
						char value = 0;
						unsigned char index = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int n = 0, x = 0; x < width; x += 8)
						{
							input.get(value);
							for (int i = 7; i >= 0; --i)
							{
								index = static_cast<unsigned char>((value >> i) & 0x01);
								dst[n++] = palette.color[index].blue;
								dst[n++] = palette.color[index].green;
								dst[n++] = palette.color[index].red;
							}
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
				else
				{
					for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
					{
						char value = 0;
						unsigned char index = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int n = 0, x = 0; x < width; x += 8)
						{
							input.get(value);
							for (int i = 7; i >= 0; --i)
							{
								index = static_cast<unsigned char>((value >> i) & 0x01);
								dst[n++] = palette.color[index].blue;
								dst[n++] = palette.color[index].green;
								dst[n++] = palette.color[index].red;
							}
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
			}
			return true;
		}

		// Read 4-bit color image
		template <class Allocator>
		static bool read_data_4bit(std::istream &input, int width, int height, int stride, const bitmap_palette &palette, ::core::matrix<unsigned char, Allocator> &output)
		{
			if (palette.number == 0)
			{
				if (height < 0)
				{
					for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
					{
						char value = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int x = 0; x < width;)
						{
							input.get(value);
							dst[x++] = (value & 0xf0);
							dst[x++] = (value & 0x0f) << 4;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
				else
				{
					for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
					{
						char value = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int x = 0; x < width;)
						{
							input.get(value);
							dst[x++] = (value & 0xf0);
							dst[x++] = (value & 0x0f) << 4;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
			}
			else
			{
				if (height < 0)
				{
					for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
					{
						char value = 0;
						unsigned char index = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int n = 0, x = 0; x < width; x += 2)
						{
							input.get(value);
							index = static_cast<unsigned char>((value >> 4) & 0x0f);
							dst[n++] = palette.color[index].blue;
							dst[n++] = palette.color[index].green;
							dst[n++] = palette.color[index].red;
							index = static_cast<unsigned char>(value & 0x0f);
							dst[n++] = palette.color[index].blue;
							dst[n++] = palette.color[index].green;
							dst[n++] = palette.color[index].red;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
				else
				{
					for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
					{
						char value = 0;
						unsigned char index = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int n = 0, x = 0; x < width; x += 2)
						{
							input.get(value);
							index = static_cast<unsigned char>((value >> 4) & 0x0f);
							dst[n++] = palette.color[index].blue;
							dst[n++] = palette.color[index].green;
							dst[n++] = palette.color[index].red;
							index = static_cast<unsigned char>(value & 0x0f);
							dst[n++] = palette.color[index].blue;
							dst[n++] = palette.color[index].green;
							dst[n++] = palette.color[index].red;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
			}
			return true;
		}

		// Read 8-bit color image
		template <class Allocator>
		static bool read_data_8bit(std::istream &input, int width, int height, int stride, const bitmap_palette &palette, ::core::matrix<unsigned char, Allocator> &output)
		{
			if (palette.number == 0)
			{
				if (height < 0)
				{
					for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
					{
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						input.read(dst, width);
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
				else
				{
					for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
					{
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						input.read(dst, width);
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
			}
			else
			{
				if (height < 0)
				{
					for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
					{
						char index = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int n = 0, x = 0; x < width; ++x)
						{
							input.get(index);
							dst[n++] = palette.color[static_cast<unsigned char>(index)].blue;
							dst[n++] = palette.color[static_cast<unsigned char>(index)].green;
							dst[n++] = palette.color[static_cast<unsigned char>(index)].red;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
				else
				{
					for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
					{
						char index = 0;
						char* dst = reinterpret_cast<char*>(vi.operator->());
						std::streampos position = input.tellg();
						for (int n = 0, x = 0; x < width; ++x)
						{
							input.get(index);
							dst[n++] = palette.color[static_cast<unsigned char>(index)].blue;
							dst[n++] = palette.color[static_cast<unsigned char>(index)].green;
							dst[n++] = palette.color[static_cast<unsigned char>(index)].red;
						}
						input.seekg(position + static_cast<std::streamoff>(stride));
					}
				}
			}
			return true;
		}

		// Read 24-bit color image
		template <class Allocator>
		static bool read_data_24bit(std::istream &input, int width, int height, int stride, ::core::matrix<unsigned char, Allocator> &output)
		{
			if (height < 0)
			{
				for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
				{
					char* dst = reinterpret_cast<char*>(vi.operator->());
					std::streampos position = input.tellg();
					input.read(dst, width * 3);
					input.seekg(position + static_cast<std::streamoff>(stride));
				}
			}
			else
			{
				for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
				{
					char* dst = reinterpret_cast<char*>(vi.operator->());
					std::streampos position = input.tellg();
					input.read(dst, width * 3);
					input.seekg(position + static_cast<std::streamoff>(stride));
				}
			}
			return true;
		}

		// Read 32-bit color image
		template <class Allocator>
		static bool read_data_32bit(std::istream &input, int width, int height, int stride, ::core::matrix<unsigned char, Allocator> &output)
		{
			if (height < 0)
			{
				for (auto vi = output.vbegin(); vi != output.vend(); ++vi)
				{
					char* dst = reinterpret_cast<char*>(vi.operator->());
					std::streampos position = input.tellg();
					input.read(dst, width * 4);
					input.seekg(position + static_cast<std::streamoff>(stride));
				}
			}
			else
			{
				for (auto vi = output.rvbegin(); vi != output.rvend(); ++vi)
				{
					char* dst = reinterpret_cast<char*>(vi.operator->());
					std::streampos position = input.tellg();
					input.read(dst, width * 4);
					input.seekg(position + static_cast<std::streamoff>(stride));
				}
			}
			return true;
		}
	};

} // namespace img

#endif
