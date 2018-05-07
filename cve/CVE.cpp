// CVE.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <iomanip>

#include "core\cpu\cpu_convert.h"
#include "core\cpu\cpu_convert_scale.h"
#include "core\cpu\cpu_reduce.h"
#include "core\cpu\cpu_transpose.h"
#include "core\cpu\cpu_border.h"
#include "core\cpu\cpu_multiply.h"
#include "image\bitmap.h"
#include "ann\mnist.h"

// Print vector
template<class Allocator>
void print(const char *name, const core::vector<signed char, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<unsigned char, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<signed short, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<unsigned short, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<int, Allocator> &vec);
template<class Allocator>
void print(const char *name, const core::vector<float, Allocator> &vec);
// Print matrix
template<class Allocator>
void print(const char *name, const core::matrix<signed char, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned char, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<signed short, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned short, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<int, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<float, Allocator> &mat);
template<class Allocator>
void print(const char *name, const core::matrix<unsigned long long, Allocator> &mat);

int main()
{
	core::cpu::enable_simd(true);
	try
	{
		size_t row = 15;
		size_t p = 16;
		size_t col = 17;
		size_t dim = 1;
		core::matrix<float> a(row, p, dim);
		core::matrix<float> b(p, col, dim);
		core::matrix<float> bt(col, p, dim);
		core::matrix<float> c(row, col, dim, 0.0F);
		core::matrix<float> d(row, col, dim, 0.0F);
		// Initialization matrix
		a.linear_fill(1.0F, 1.0F);
		b.linear_fill(1.0F, 1.0F);
		core::cpu_transpose(bt, b);
		// Matrix-matrix multiplication
		core::cpu_multiply(c, a, b);
		print("a", a);
		print("b", b);
		print("c=a*b", c);
		core::cpu_multiply(d, a, bt, true);
		print("d=a*b^t", d);
	}
	catch (std::exception err)
	{
		std::cout << err.what() << std::endl;
	}
/*
	 23256.  23392.  23528.  23664.  23800.  23936.  24072.  24208.  24344.   24480.  24616.  24752.  24888.  25024.  25160.  25296.  25432.
	 56152.  56544.  56936.  57328.  57720.  58112.  58504.  58896.  59288.   59680.  60072.  60464.  60856.  61248.  61640.  62032.  62424.
	 89048.  89696.  90344.  90992.  91640.  92288.  92936.  93584.  94232.   94880.  95528.  96176.  96824.  97472.  98120.  98768.  99416.
	121944. 122848. 123752. 124656. 125560. 126464. 127368. 128272. 129176.  130080. 130984. 131888. 132792. 133696. 134600. 135504. 136408.
	154840. 156000. 157160. 158320. 159480. 160640. 161800. 162960. 164120.  165280. 166440. 167600. 168760. 169920. 171080. 172240. 173400.
	187736. 189152. 190568. 191984. 193400. 194816. 196232. 197648. 199064.  200480. 201896. 203312. 204728. 206144. 207560. 208976. 210392.
	220632. 222304. 223976. 225648. 227320. 228992. 230664. 232336. 234008.  235680. 237352. 239024. 240696. 242368. 244040. 245712. 247384.
	253528. 255456. 257384. 259312. 261240. 263168. 265096. 267024. 268952.  270880. 272808. 274736. 276664. 278592. 280520. 282448. 284376.
	286424. 288608. 290792. 292976. 295160. 297344. 299528. 301712. 303896.  306080. 308264. 310448. 312632. 314816. 317000. 319184. 321368.
	319320. 321760. 324200. 326640. 329080. 331520. 333960. 336400. 338840.  341280. 343720. 346160. 348600. 351040. 353480. 355920. 358360.
	352216. 354912. 357608. 360304. 363000. 365696. 368392. 371088. 373784.  376480. 379176. 381872. 384568. 387264. 389960. 392656. 395352.
	385112. 388064. 391016. 393968. 396920. 399872. 402824. 405776. 408728.  411680. 414632. 417584. 420536. 423488. 426440. 429392. 432344.
	418008. 421216. 424424. 427632. 430840. 434048. 437256. 440464. 443672.  446880. 450088. 453296. 456504. 459712. 462920. 466128. 469336.
	450904. 454368. 457832. 461296. 464760. 468224. 471688. 475152. 478616.  482080. 485544. 489008. 492472. 495936. 499400. 502864. 506328.
	483800. 487520. 491240. 494960. 498680. 502400. 506120. 509840. 513560.  517280. 521000. 524720. 528440. 532160. 535880. 539600. 543320.
*/
	return 0;

	try
	{
		std::string input_image = "data/test.bmp";
		std::string border_wrap = "data/wrap.bmp";

		core::matrix<unsigned char> input;
		if (img::bitmap::decode(input_image, input))
		{
			size_t left = 482 * 1 + 20;
			size_t top = 272 * 1 + 20;
			size_t right = 482 * 1 + 20;
			size_t bottom = 272 * 1 + 20;
			size_t rows = input.rows();
			size_t columns = input.columns();
			size_t dimension = input.dimension();
			core::matrix<size_t> index(top + rows + bottom, left + columns + right, dimension);
			core::cpu_border(index, left, top, right, bottom, core::border_wrap);
			core::matrix<unsigned char> output(input.data(), index);
			img::bitmap::encode(border_wrap, output);
		}
		else
			std::cout << "Can't load image file '" << input_image.data() << "'." << std::endl;
	}
	catch (std::exception err)
	{
		std::cout << err.what() << std::endl;
	}
	return 0;
/*
	const size_t batch     = 10;
	const size_t rows      = 28;
	const size_t columns   = 28;
	const size_t length    = 10;
	const size_t dimension = 1;

	ann::mnist<> mnist("data/mnist");
	core::tensor<float> train_images_flt(batch, rows, columns, dimension);
	core::tensor<unsigned char> train_images(batch, rows, columns, dimension);
	core::vector<unsigned char> train_labels(length, dimension);
	core::tensor<unsigned char> test_images(batch, rows, columns, dimension);
	core::vector<unsigned char> test_labels(length, dimension);

	mnist.train.shuffle(1U);
	mnist.train.next_batch(train_images_flt, train_labels);
	core::cpu_convert(train_images, train_images_flt);

	img::bitmap::encode("data/train/1.bmp", train_images[0]);
	img::bitmap::encode("data/train/2.bmp", train_images[1]);
	img::bitmap::encode("data/train/3.bmp", train_images[2]);
	img::bitmap::encode("data/train/4.bmp", train_images[3]);
	img::bitmap::encode("data/train/5.bmp", train_images[4]);
	img::bitmap::encode("data/train/6.bmp", train_images[5]);
	img::bitmap::encode("data/train/7.bmp", train_images[6]);
	img::bitmap::encode("data/train/8.bmp", train_images[7]);

	mnist.test.shuffle(1U);
	mnist.test.next_batch(test_images, test_labels);
	img::bitmap::encode("data/test/1.bmp", test_images[0]);
	img::bitmap::encode("data/test/2.bmp", test_images[1]);
	img::bitmap::encode("data/test/3.bmp", test_images[2]);
	img::bitmap::encode("data/test/4.bmp", test_images[3]);
	img::bitmap::encode("data/test/5.bmp", test_images[4]);
	img::bitmap::encode("data/test/6.bmp", test_images[5]);
	img::bitmap::encode("data/test/7.bmp", test_images[6]);
	img::bitmap::encode("data/test/8.bmp", test_images[7]);
*/
	//size_t row = 13;
	//size_t col = 17;
	//size_t dim = 1;
	//core::matrix<signed char> x(row, col, dim);
	//core::matrix<signed char> t(col, row, dim);
	//core::vector<signed char> col_min(col, dim, static_cast<signed char>(core::int8_max));
	//core::vector<signed char> col_max(col, dim, static_cast<signed char>(core::int8_min));
	//core::vector<signed int> col_sum(col, dim, static_cast<signed int>(core::int32_zero));
	//core::vector<signed char> row_min(row, dim, static_cast<signed char>(core::int8_max));
	//core::vector<signed char> row_max(row, dim, static_cast<signed char>(core::int8_min));
	//core::vector<signed int> row_sum(row, dim, static_cast<signed int>(core::int32_zero));
	//// Initialization matrix
	//x.linear_fill(static_cast<signed char>(1), static_cast<signed char>(2), static_cast<signed char>(1));
	//// Matrix operation
	//core::transpose(t, x);
	//core::reduce(col_min, x, core::reduce_col_min);
	//core::reduce(col_max, x, core::reduce_col_max);
	//core::reduce(col_sum, x, core::reduce_col_sum);
	//core::reduce(row_min, x, core::reduce_row_min);
	//core::reduce(row_max, x, core::reduce_row_max);
	//core::reduce(row_sum, x, core::reduce_row_sum);
	//print("X", x);
	//print("T", t);
	//print("col_min", col_min);
	//print("col_max", col_max);
	//print("col_sum", col_sum);
	//print("row_min", row_min);
	//print("row_max", row_max);
	//print("row_sum", row_sum);
	return 0;
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed char, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(static_cast<unsigned char>(*i)) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<unsigned char, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*i) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed short, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(static_cast<unsigned short>(*i)) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<unsigned short, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(*i) << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<signed int, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << "0x" << std::hex << std::setfill('0') << std::setw(8) << *i << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const ::core::vector<float, Allocator> &vec)
{
	std::cout << name << "[" << vec.length() << "] =\n";
	std::cout << "    ";
	for (auto i = vec.begin(); i != vec.end(); ++i)
		std::cout << std::setfill(' ') << std::setw(6) << *i << ",";
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<signed char, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(static_cast<unsigned char>(*i)) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned char, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*i) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<signed short, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(static_cast<unsigned short>(*i)) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned short, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(*i) << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<int, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << "0x" << std::hex << std::setfill('0') << std::setw(8) << *i << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<float, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			std::cout << std::setfill(' ') << std::setw(6) << *i << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<class Allocator>
void print(const char *name, const core::matrix<unsigned long long, Allocator> &mat)
{
	std::cout << name << "[" << mat.rows() << "][" << mat.columns() << "] =\n";
	for (auto j = mat.vbegin(); j != mat.vend(); ++j)
	{
		std::cout << "    ";
		for (auto i = mat.begin(j); i != mat.end(j); ++i)
			for (size_t n = 0; n < mat.dimension(); ++n)
				std::cout << std::setfill(' ') << std::setw(3) << i[n] << ",";
		std::cout << "\n";
	}
	std::cout << "\n";
}
