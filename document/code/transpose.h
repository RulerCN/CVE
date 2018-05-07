#include <intrin.h>

template <class T>
void transpose(int m, int n, const T **a, T **b)
{
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			b[j][i] = a[i][j];
}

template <class T>
void block_transpose(int m, int n, const T **a, T **b)
{
	for (int i = 0; i < m; i += 4)
	{
		for (int j = 0; j < n; j += 4)
		{
			b[j][i] = a[i][j];
			b[j + 1][i] = a[i][j + 1];
			b[j + 2][i] = a[i][j + 2];
			b[j + 3][i] = a[i][j + 3];
			b[j][i + 1] = a[i + 1][j];
			b[j + 1][i + 1] = a[i + 1][j + 1];
			b[j + 2][i + 1] = a[i + 1][j + 2];
			b[j + 3][i + 1] = a[i + 1][j + 3];
			b[j][i + 2] = a[i + 2][j];
			b[j + 1][i + 2] = a[i + 2][j + 1];
			b[j + 2][i + 2] = a[i + 2][j + 2];
			b[j + 3][i + 2] = a[i + 2][j + 3];
			b[j][i + 3] = a[i + 3][j];
			b[j + 1][i + 3] = a[i + 3][j + 1];
			b[j + 2][i + 3] = a[i + 3][j + 2];
			b[j + 3][i + 3] = a[i + 3][j + 3];
		}
	}
}

void block_transpose_sse2(int m, int n, const char **a, char **b)
{
	__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
	__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;

	for (int i = 0; i < m; i += 16)
	{
		for (int j = 0; j < n; j += 8)
		{
			// load data from memory
			xmm_a0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i][j]));
			xmm_a1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 1][j]));
			xmm_a2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 2][j]));
			xmm_a3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 3][j]));
			xmm_a4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 4][j]));
			xmm_a5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 5][j]));
			xmm_a6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 6][j]));
			xmm_a7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 7][j]));
			xmm_t0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 8][j]));
			xmm_t1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 9][j]));
			xmm_t2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 10][j]));
			xmm_t3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 11][j]));
			xmm_t4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 12][j]));
			xmm_t5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 13][j]));
			xmm_t6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 14][j]));
			xmm_t7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&a[i + 15][j]));
			// matrix transposed
			xmm_a0 = _mm_unpacklo_epi8(xmm_a0, xmm_a1);
			xmm_a1 = _mm_unpacklo_epi8(xmm_a2, xmm_a3);
			xmm_a2 = _mm_unpacklo_epi8(xmm_a4, xmm_a5);
			xmm_a3 = _mm_unpacklo_epi8(xmm_a6, xmm_a7);
			xmm_a4 = _mm_unpacklo_epi8(xmm_t0, xmm_t1);
			xmm_a5 = _mm_unpacklo_epi8(xmm_t2, xmm_t3);
			xmm_a6 = _mm_unpacklo_epi8(xmm_t4, xmm_t5);
			xmm_a7 = _mm_unpacklo_epi8(xmm_t6, xmm_t7);
			xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
			xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
			xmm_t2 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
			xmm_t3 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
			xmm_t4 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
			xmm_t5 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
			xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
			xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
			xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
			xmm_a1 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
			xmm_a2 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
			xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
			xmm_a4 = _mm_unpacklo_epi32(xmm_t4, xmm_t5);
			xmm_a5 = _mm_unpackhi_epi32(xmm_t4, xmm_t5);
			xmm_a6 = _mm_unpacklo_epi32(xmm_t6, xmm_t7);
			xmm_a7 = _mm_unpackhi_epi32(xmm_t6, xmm_t7);
			xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a4);
			xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a4);
			xmm_t2 = _mm_unpacklo_epi64(xmm_a1, xmm_a5);
			xmm_t3 = _mm_unpackhi_epi64(xmm_a1, xmm_a5);
			xmm_t4 = _mm_unpacklo_epi64(xmm_a2, xmm_a6);
			xmm_t5 = _mm_unpackhi_epi64(xmm_a2, xmm_a6);
			xmm_t6 = _mm_unpacklo_epi64(xmm_a3, xmm_a7);
			xmm_t7 = _mm_unpackhi_epi64(xmm_a3, xmm_a7);
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j][i]), xmm_t0);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 1][i]), xmm_t1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 2][i]), xmm_t2);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 3][i]), xmm_t3);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 4][i]), xmm_t4);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 5][i]), xmm_t5);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 6][i]), xmm_t6);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 7][i]), xmm_t7);
		}
	}
}

void block_transpose_sse2(int m, int n, const short **a, short **b)
{
	__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3, xmm_a4, xmm_a5, xmm_a6, xmm_a7;
	__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3, xmm_t4, xmm_t5, xmm_t6, xmm_t7;

	for (int i = 0; i < m; i += 8)
	{
		for (int j = 0; j < n; j += 8)
		{
			// load data from memory
			xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i][j]));
			xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 1][j]));
			xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 2][j]));
			xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 3][j]));
			xmm_a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 4][j]));
			xmm_a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 5][j]));
			xmm_a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 6][j]));
			xmm_a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 7][j]));
			// matrix transposed
			xmm_t0 = _mm_unpacklo_epi16(xmm_a0, xmm_a1);
			xmm_t1 = _mm_unpacklo_epi16(xmm_a2, xmm_a3);
			xmm_t2 = _mm_unpacklo_epi16(xmm_a4, xmm_a5);
			xmm_t3 = _mm_unpacklo_epi16(xmm_a6, xmm_a7);
			xmm_t4 = _mm_unpackhi_epi16(xmm_a0, xmm_a1);
			xmm_t5 = _mm_unpackhi_epi16(xmm_a2, xmm_a3);
			xmm_t6 = _mm_unpackhi_epi16(xmm_a4, xmm_a5);
			xmm_t7 = _mm_unpackhi_epi16(xmm_a6, xmm_a7);
			xmm_a0 = _mm_unpacklo_epi32(xmm_t0, xmm_t1);
			xmm_a1 = _mm_unpacklo_epi32(xmm_t2, xmm_t3);
			xmm_a2 = _mm_unpackhi_epi32(xmm_t0, xmm_t1);
			xmm_a3 = _mm_unpackhi_epi32(xmm_t2, xmm_t3);
			xmm_a4 = _mm_unpacklo_epi32(xmm_t4, xmm_t5);
			xmm_a5 = _mm_unpacklo_epi32(xmm_t6, xmm_t7);
			xmm_a6 = _mm_unpackhi_epi32(xmm_t4, xmm_t5);
			xmm_a7 = _mm_unpackhi_epi32(xmm_t6, xmm_t7);
			xmm_t0 = _mm_unpacklo_epi64(xmm_a0, xmm_a1);
			xmm_t1 = _mm_unpackhi_epi64(xmm_a0, xmm_a1);
			xmm_t2 = _mm_unpacklo_epi64(xmm_a2, xmm_a3);
			xmm_t3 = _mm_unpackhi_epi64(xmm_a2, xmm_a3);
			xmm_t4 = _mm_unpacklo_epi64(xmm_a4, xmm_a5);
			xmm_t5 = _mm_unpackhi_epi64(xmm_a4, xmm_a5);
			xmm_t6 = _mm_unpacklo_epi64(xmm_a6, xmm_a7);
			xmm_t7 = _mm_unpackhi_epi64(xmm_a6, xmm_a7);
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j][i]), xmm_t0);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 1][i]), xmm_t1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 2][i]), xmm_t2);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 3][i]), xmm_t3);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 4][i]), xmm_t4);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 5][i]), xmm_t5);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 6][i]), xmm_t6);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 7][i]), xmm_t7);
		}
	}
}

void block_transpose_sse2(int m, int n, const int **a, int **b)
{
	__m128i xmm_a0, xmm_a1, xmm_a2, xmm_a3;
	__m128i xmm_t0, xmm_t1, xmm_t2, xmm_t3;

	for (int i = 0; i < m; i += 4)
	{
		for (int j = 0; j < n; j += 4)
		{
			// load data from memory
			xmm_a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i][j]));
			xmm_a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 1][j]));
			xmm_a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 2][j]));
			xmm_a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 3][j]));
			// matria transposed
			xmm_t0 = _mm_unpacklo_epi32(xmm_a0, xmm_a1);
			xmm_t1 = _mm_unpacklo_epi32(xmm_a2, xmm_a3);
			xmm_t2 = _mm_unpackhi_epi32(xmm_a0, xmm_a1);
			xmm_t3 = _mm_unpackhi_epi32(xmm_a2, xmm_a3);
			xmm_a0 = _mm_unpacklo_epi64(xmm_t0, xmm_t1);
			xmm_a1 = _mm_unpackhi_epi64(xmm_t0, xmm_t1);
			xmm_a2 = _mm_unpacklo_epi64(xmm_t2, xmm_t3);
			xmm_a3 = _mm_unpackhi_epi64(xmm_t2, xmm_t3);
			// store data into memory
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j][i]), xmm_a0);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 1][i]), xmm_a1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 2][i]), xmm_a2);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&b[j + 3][i]), xmm_a3);
		}
	}
}

void block_transpose_sse(int m, int n, const float **a, float **b)
{
	__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
	__m128 xmm_t0, xmm_t1, xmm_t2, xmm_t3;

	for (int i = 0; i < m; i += 4)
	{
		for (int j = 0; j < n; j += 4)
		{
			// load data from memory
			xmm_a0 = _mm_loadu_ps(&a[i][j]);
			xmm_a1 = _mm_loadu_ps(&a[i + 1][j]);
			xmm_a2 = _mm_loadu_ps(&a[i + 2][j]);
			xmm_a3 = _mm_loadu_ps(&a[i + 3][j]);
			// matria transposed
			xmm_t0 = _mm_shuffle_ps(xmm_a0, xmm_a1, _MM_SHUFFLE(1, 0, 1, 0));
			xmm_t1 = _mm_shuffle_ps(xmm_a2, xmm_a3, _MM_SHUFFLE(1, 0, 1, 0));
			xmm_t2 = _mm_shuffle_ps(xmm_a0, xmm_a1, _MM_SHUFFLE(3, 2, 3, 2));
			xmm_t3 = _mm_shuffle_ps(xmm_a2, xmm_a3, _MM_SHUFFLE(3, 2, 3, 2));
			xmm_a0 = _mm_shuffle_ps(xmm_t0, xmm_t1, _MM_SHUFFLE(2, 0, 2, 0));
			xmm_a1 = _mm_shuffle_ps(xmm_t0, xmm_t1, _MM_SHUFFLE(3, 1, 3, 1));
			xmm_a2 = _mm_shuffle_ps(xmm_t2, xmm_t3, _MM_SHUFFLE(2, 0, 2, 0));
			xmm_a3 = _mm_shuffle_ps(xmm_t2, xmm_t3, _MM_SHUFFLE(3, 1, 3, 1));
			// store data into memory
			_mm_storeu_ps(&b[j][i]), xmm_a0);
			_mm_storeu_ps(&b[j + 1][i]), xmm_a1);
			_mm_storeu_ps(&b[j + 2][i]), xmm_a2);
			_mm_storeu_ps(&b[j + 3][i]), xmm_a3);
		}
	}
}

void block_transpose_sse2(int m, int n, const double **a, double **b)
{
	__m128d xmm_a0, xmm_a1;
	__m128d xmm_t0, xmm_t1;

	for (int i = 0; i < m; i += 2)
	{
		for (int j = 0; j < n; j += 2)
		{
			// load data from memory
			xmm_a0 = _mm_loadu_pd(&a[i][j]);
			xmm_a1 = _mm_loadu_pd(&a[i + 1][j]);
			// matria transposed
			xmm_t0 = _mm_shuffle_pd(xmm_a0, xmm_a1, _MM_SHUFFLE(0, 0, 0, 0));
			xmm_t1 = _mm_shuffle_pd(xmm_a0, xmm_a1, _MM_SHUFFLE(0, 0, 3, 3));
			// store data into memory
			_mm_storeu_pd(&b[j][i]), xmm_t0);
			_mm_storeu_pd(&b[j + 1][i]), xmm_t1);
		}
	}
}

void block_transpose_avx2(int m, int n, const char **a, char **b)
{
	__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7, ymm_a8, ymm_a9, ymm_aa, ymm_ab, ymm_ac, ymm_ad, ymm_ae, ymm_af;
	__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7, ymm_t8, ymm_t9, ymm_ta, ymm_tb, ymm_tc, ymm_td, ymm_te, ymm_tf;

	for (int i = 0; i < m; i += 32)
	{
		for (int j = 0; j < n; j += 16)
		{
			// load data from memory
			ymm_a0 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i][j])));
			ymm_a1 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 1][j])));
			ymm_a2 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 2][j])));
			ymm_a3 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 3][j])));
			ymm_a4 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 4][j])));
			ymm_a5 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 5][j])));
			ymm_a6 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 6][j])));
			ymm_a7 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 7][j])));
			ymm_a8 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 8][j])));
			ymm_a9 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 9][j])));
			ymm_aa = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 10][j])));
			ymm_ab = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 11][j])));
			ymm_ac = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 12][j])));
			ymm_ad = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 13][j])));
			ymm_ae = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 14][j])));
			ymm_af = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 15][j])));
			ymm_a0 = _mm256_insertf128_si256(ymm_a0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 16][j])), 1);
			ymm_a1 = _mm256_insertf128_si256(ymm_a1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 17][j])), 1);
			ymm_a2 = _mm256_insertf128_si256(ymm_a2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 18][j])), 1);
			ymm_a3 = _mm256_insertf128_si256(ymm_a3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 19][j])), 1);
			ymm_a4 = _mm256_insertf128_si256(ymm_a4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 20][j])), 1);
			ymm_a5 = _mm256_insertf128_si256(ymm_a5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 21][j])), 1);
			ymm_a6 = _mm256_insertf128_si256(ymm_a6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 22][j])), 1);
			ymm_a7 = _mm256_insertf128_si256(ymm_a7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 23][j])), 1);
			ymm_a8 = _mm256_insertf128_si256(ymm_a8, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 24][j])), 1);
			ymm_a9 = _mm256_insertf128_si256(ymm_a9, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 25][j])), 1);
			ymm_aa = _mm256_insertf128_si256(ymm_aa, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 26][j])), 1);
			ymm_ab = _mm256_insertf128_si256(ymm_ab, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 27][j])), 1);
			ymm_ac = _mm256_insertf128_si256(ymm_ac, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 28][j])), 1);
			ymm_ad = _mm256_insertf128_si256(ymm_ad, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 29][j])), 1);
			ymm_ae = _mm256_insertf128_si256(ymm_ae, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 30][j])), 1);
			ymm_af = _mm256_insertf128_si256(ymm_af, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 31][j])), 1);
			// matria transposed
			ymm_t0 = _mm256_unpacklo_epi8(ymm_a0, ymm_a1);
			ymm_t1 = _mm256_unpacklo_epi8(ymm_a2, ymm_a3);
			ymm_t2 = _mm256_unpacklo_epi8(ymm_a4, ymm_a5);
			ymm_t3 = _mm256_unpacklo_epi8(ymm_a6, ymm_a7);
			ymm_t4 = _mm256_unpacklo_epi8(ymm_a8, ymm_a9);
			ymm_t5 = _mm256_unpacklo_epi8(ymm_aa, ymm_ab);
			ymm_t6 = _mm256_unpacklo_epi8(ymm_ac, ymm_ad);
			ymm_t7 = _mm256_unpacklo_epi8(ymm_ae, ymm_af);
			ymm_t8 = _mm256_unpackhi_epi8(ymm_a0, ymm_a1);
			ymm_t9 = _mm256_unpackhi_epi8(ymm_a2, ymm_a3);
			ymm_ta = _mm256_unpackhi_epi8(ymm_a4, ymm_a5);
			ymm_tb = _mm256_unpackhi_epi8(ymm_a6, ymm_a7);
			ymm_tc = _mm256_unpackhi_epi8(ymm_a8, ymm_a9);
			ymm_td = _mm256_unpackhi_epi8(ymm_aa, ymm_ab);
			ymm_te = _mm256_unpackhi_epi8(ymm_ac, ymm_ad);
			ymm_tf = _mm256_unpackhi_epi8(ymm_ae, ymm_af);
			ymm_a0 = _mm256_unpacklo_epi16(ymm_t0, ymm_t1);
			ymm_a1 = _mm256_unpacklo_epi16(ymm_t2, ymm_t3);
			ymm_a2 = _mm256_unpacklo_epi16(ymm_t4, ymm_t5);
			ymm_a3 = _mm256_unpacklo_epi16(ymm_t6, ymm_t7);
			ymm_a4 = _mm256_unpackhi_epi16(ymm_t0, ymm_t1);
			ymm_a5 = _mm256_unpackhi_epi16(ymm_t2, ymm_t3);
			ymm_a6 = _mm256_unpackhi_epi16(ymm_t4, ymm_t5);
			ymm_a7 = _mm256_unpackhi_epi16(ymm_t6, ymm_t7);
			ymm_a8 = _mm256_unpacklo_epi16(ymm_t8, ymm_t9);
			ymm_a9 = _mm256_unpacklo_epi16(ymm_ta, ymm_tb);
			ymm_aa = _mm256_unpacklo_epi16(ymm_tc, ymm_td);
			ymm_ab = _mm256_unpacklo_epi16(ymm_te, ymm_tf);
			ymm_ac = _mm256_unpackhi_epi16(ymm_t8, ymm_t9);
			ymm_ad = _mm256_unpackhi_epi16(ymm_ta, ymm_tb);
			ymm_ae = _mm256_unpackhi_epi16(ymm_tc, ymm_td);
			ymm_af = _mm256_unpackhi_epi16(ymm_te, ymm_tf);
			ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
			ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
			ymm_t2 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
			ymm_t3 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
			ymm_t4 = _mm256_unpacklo_epi32(ymm_a4, ymm_a5);
			ymm_t5 = _mm256_unpacklo_epi32(ymm_a6, ymm_a7);
			ymm_t6 = _mm256_unpackhi_epi32(ymm_a4, ymm_a5);
			ymm_t7 = _mm256_unpackhi_epi32(ymm_a6, ymm_a7);
			ymm_t8 = _mm256_unpacklo_epi32(ymm_a8, ymm_a9);
			ymm_t9 = _mm256_unpacklo_epi32(ymm_aa, ymm_ab);
			ymm_ta = _mm256_unpackhi_epi32(ymm_a8, ymm_a9);
			ymm_tb = _mm256_unpackhi_epi32(ymm_aa, ymm_ab);
			ymm_tc = _mm256_unpacklo_epi32(ymm_ac, ymm_ad);
			ymm_td = _mm256_unpacklo_epi32(ymm_ae, ymm_af);
			ymm_te = _mm256_unpackhi_epi32(ymm_ac, ymm_ad);
			ymm_tf = _mm256_unpackhi_epi32(ymm_ae, ymm_af);
			ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
			ymm_a1 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
			ymm_a2 = _mm256_unpacklo_epi64(ymm_t2, ymm_t3);
			ymm_a3 = _mm256_unpackhi_epi64(ymm_t2, ymm_t3);
			ymm_a4 = _mm256_unpacklo_epi64(ymm_t4, ymm_t5);
			ymm_a5 = _mm256_unpackhi_epi64(ymm_t4, ymm_t5);
			ymm_a6 = _mm256_unpacklo_epi64(ymm_t6, ymm_t7);
			ymm_a7 = _mm256_unpackhi_epi64(ymm_t6, ymm_t7);
			ymm_a8 = _mm256_unpacklo_epi64(ymm_t8, ymm_t9);
			ymm_a9 = _mm256_unpackhi_epi64(ymm_t8, ymm_t9);
			ymm_aa = _mm256_unpacklo_epi64(ymm_ta, ymm_tb);
			ymm_ab = _mm256_unpackhi_epi64(ymm_ta, ymm_tb);
			ymm_ac = _mm256_unpacklo_epi64(ymm_tc, ymm_td);
			ymm_ad = _mm256_unpackhi_epi64(ymm_tc, ymm_td);
			ymm_ae = _mm256_unpacklo_epi64(ymm_te, ymm_tf);
			ymm_af = _mm256_unpackhi_epi64(ymm_te, ymm_tf);
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j][i]), ymm_a0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 1][i]), ymm_a1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 2][i]), ymm_a2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 3][i]), ymm_a3);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 4][i]), ymm_a4);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 5][i]), ymm_a5);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 6][i]), ymm_a6);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 7][i]), ymm_a7);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 8][i]), ymm_a8);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 9][i]), ymm_a9);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 10][i]), ymm_aa);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 11][i]), ymm_ab);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 12][i]), ymm_ac);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 13][i]), ymm_ad);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 14][i]), ymm_ae);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 15][i]), ymm_af);
		}
	}
}

void block_transpose_avx2(int m, int n, const short **a, short **b)
{
	__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
	__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

	for (int i = 0; i < m; i += 16)
	{
		for (int j = 0; j < n; j += 8)
		{
			// load data from memory
			ymm_a0 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i][j])));
			ymm_a1 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 1][j])));
			ymm_a2 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 2][j])));
			ymm_a3 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 3][j])));
			ymm_a4 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 4][j])));
			ymm_a5 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 5][j])));
			ymm_a6 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 6][j])));
			ymm_a7 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 7][j])));
			ymm_a0 = _mm256_insertf128_si256(ymm_a0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 8][j])), 1);
			ymm_a1 = _mm256_insertf128_si256(ymm_a1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 9][j])), 1);
			ymm_a2 = _mm256_insertf128_si256(ymm_a2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 10][j])), 1);
			ymm_a3 = _mm256_insertf128_si256(ymm_a3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 11][j])), 1);
			ymm_a4 = _mm256_insertf128_si256(ymm_a4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 12][j])), 1);
			ymm_a5 = _mm256_insertf128_si256(ymm_a5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 13][j])), 1);
			ymm_a6 = _mm256_insertf128_si256(ymm_a6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 14][j])), 1);
			ymm_a7 = _mm256_insertf128_si256(ymm_a7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i + 15][j])), 1);
			// matrix transposed
			ymm_t0 = _mm256_unpacklo_epi16(ymm_a0, ymm_a1);
			ymm_t1 = _mm256_unpacklo_epi16(ymm_a2, ymm_a3);
			ymm_t2 = _mm256_unpacklo_epi16(ymm_a4, ymm_a5);
			ymm_t3 = _mm256_unpacklo_epi16(ymm_a6, ymm_a7);
			ymm_t4 = _mm256_unpackhi_epi16(ymm_a0, ymm_a1);
			ymm_t5 = _mm256_unpackhi_epi16(ymm_a2, ymm_a3);
			ymm_t6 = _mm256_unpackhi_epi16(ymm_a4, ymm_a5);
			ymm_t7 = _mm256_unpackhi_epi16(ymm_a6, ymm_a7);
			ymm_a0 = _mm256_unpacklo_epi32(ymm_t0, ymm_t1);
			ymm_a1 = _mm256_unpacklo_epi32(ymm_t2, ymm_t3);
			ymm_a2 = _mm256_unpackhi_epi32(ymm_t0, ymm_t1);
			ymm_a3 = _mm256_unpackhi_epi32(ymm_t2, ymm_t3);
			ymm_a4 = _mm256_unpacklo_epi32(ymm_t4, ymm_t5);
			ymm_a5 = _mm256_unpacklo_epi32(ymm_t6, ymm_t7);
			ymm_a6 = _mm256_unpackhi_epi32(ymm_t4, ymm_t5);
			ymm_a7 = _mm256_unpackhi_epi32(ymm_t6, ymm_t7);
			ymm_t0 = _mm256_unpacklo_epi64(ymm_a0, ymm_a1);
			ymm_t1 = _mm256_unpackhi_epi64(ymm_a0, ymm_a1);
			ymm_t2 = _mm256_unpacklo_epi64(ymm_a2, ymm_a3);
			ymm_t3 = _mm256_unpackhi_epi64(ymm_a2, ymm_a3);
			ymm_t4 = _mm256_unpacklo_epi64(ymm_a4, ymm_a5);
			ymm_t5 = _mm256_unpackhi_epi64(ymm_a4, ymm_a5);
			ymm_t6 = _mm256_unpacklo_epi64(ymm_a6, ymm_a7);
			ymm_t7 = _mm256_unpackhi_epi64(ymm_a6, ymm_a7);
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j][i]), ymm_t0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 1][i]), ymm_t1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 2][i]), ymm_t2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 3][i]), ymm_t3);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 4][i]), ymm_t4);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 5][i]), ymm_t5);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 6][i]), ymm_t6);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 7][i]), ymm_t7);
		}
	}
}

void block_transpose_avx2(int m, int n, const int **a, int **b)
{
	__m256i ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
	__m256i ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

	for (int i = 0; i < m; i += 8)
	{
		for (int j = 0; j < n; j += 8)
		{
			// load data from memory
			ymm_a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i][j]));
			ymm_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i + 1][j]));
			ymm_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i + 2][j]));
			ymm_a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i + 3][j]));
			ymm_a4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i + 4][j]));
			ymm_a5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i + 5][j]));
			ymm_a6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i + 6][j]));
			ymm_a7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i + 7][j]));
			// matrix transposed
			ymm_t0 = _mm256_unpacklo_epi32(ymm_a0, ymm_a1);
			ymm_t1 = _mm256_unpacklo_epi32(ymm_a2, ymm_a3);
			ymm_t2 = _mm256_unpacklo_epi32(ymm_a4, ymm_a5);
			ymm_t3 = _mm256_unpacklo_epi32(ymm_a6, ymm_a7);
			ymm_t4 = _mm256_unpackhi_epi32(ymm_a0, ymm_a1);
			ymm_t5 = _mm256_unpackhi_epi32(ymm_a2, ymm_a3);
			ymm_t6 = _mm256_unpackhi_epi32(ymm_a4, ymm_a5);
			ymm_t7 = _mm256_unpackhi_epi32(ymm_a6, ymm_a7);
			ymm_a0 = _mm256_unpacklo_epi64(ymm_t0, ymm_t1);
			ymm_a1 = _mm256_unpacklo_epi64(ymm_t2, ymm_t3);
			ymm_a2 = _mm256_unpackhi_epi64(ymm_t0, ymm_t1);
			ymm_a3 = _mm256_unpackhi_epi64(ymm_t2, ymm_t3);
			ymm_a4 = _mm256_unpacklo_epi64(ymm_t4, ymm_t5);
			ymm_a5 = _mm256_unpacklo_epi64(ymm_t6, ymm_t7);
			ymm_a6 = _mm256_unpackhi_epi64(ymm_t4, ymm_t5);
			ymm_a7 = _mm256_unpackhi_epi64(ymm_t6, ymm_t7);
			ymm_t0 = _mm256_permute2f128_si256(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t1 = _mm256_permute2f128_si256(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t2 = _mm256_permute2f128_si256(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t3 = _mm256_permute2f128_si256(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t4 = _mm256_permute2f128_si256(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_t5 = _mm256_permute2f128_si256(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_t6 = _mm256_permute2f128_si256(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_t7 = _mm256_permute2f128_si256(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 3, 0, 1));
			// store data into memory
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j][i]), ymm_t0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 1][i]), ymm_t1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 2][i]), ymm_t2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 3][i]), ymm_t3);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 4][i]), ymm_t4);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 5][i]), ymm_t5);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 6][i]), ymm_t6);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(&b[j + 7][i]), ymm_t7);
		}
	}
}

void block_transpose_avx(int m, int n, const float **a, float **b)
{
	__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
	__m256 ymm_t0, ymm_t1, ymm_t2, ymm_t3, ymm_t4, ymm_t5, ymm_t6, ymm_t7;

	for (int i = 0; i < m; i += 8)
	{
		for (int j = 0; j < n; j += 8)
		{
			// load data from memory
			ymm_a0 = _mm256_loadu_ps(&a[i][j]);
			ymm_a1 = _mm256_loadu_ps(&a[i + 1][j]);
			ymm_a2 = _mm256_loadu_ps(&a[i + 2][j]);
			ymm_a3 = _mm256_loadu_ps(&a[i + 3][j]);
			ymm_a4 = _mm256_loadu_ps(&a[i + 4][j]);
			ymm_a5 = _mm256_loadu_ps(&a[i + 5][j]);
			ymm_a6 = _mm256_loadu_ps(&a[i + 6][j]);
			ymm_a7 = _mm256_loadu_ps(&a[i + 7][j]);
			// matrix transposed
			ymm_t0 = _mm256_shuffle_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_t1 = _mm256_shuffle_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_t2 = _mm256_shuffle_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_t3 = _mm256_shuffle_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(1, 0, 1, 0));
			ymm_t4 = _mm256_shuffle_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_t5 = _mm256_shuffle_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_t6 = _mm256_shuffle_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_t7 = _mm256_shuffle_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(3, 2, 3, 2));
			ymm_a0 = _mm256_shuffle_ps(ymm_t0, ymm_t1, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_a1 = _mm256_shuffle_ps(ymm_t2, ymm_t3, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_a2 = _mm256_shuffle_ps(ymm_t0, ymm_t1, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_a3 = _mm256_shuffle_ps(ymm_t2, ymm_t3, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_a4 = _mm256_shuffle_ps(ymm_t4, ymm_t5, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_a5 = _mm256_shuffle_ps(ymm_t6, ymm_t7, _MM_SHUFFLE(2, 0, 2, 0));
			ymm_a6 = _mm256_shuffle_ps(ymm_t4, ymm_t5, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_a7 = _mm256_shuffle_ps(ymm_t6, ymm_t7, _MM_SHUFFLE(3, 1, 3, 1));
			ymm_t0 = _mm256_permute2f128_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t1 = _mm256_permute2f128_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t2 = _mm256_permute2f128_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t3 = _mm256_permute2f128_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_t4 = _mm256_permute2f128_ps(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_t5 = _mm256_permute2f128_ps(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_t6 = _mm256_permute2f128_ps(ymm_a4, ymm_a5, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_t7 = _mm256_permute2f128_ps(ymm_a6, ymm_a7, _MM_SHUFFLE(0, 3, 0, 1));
			// store data into memory
			_mm256_storeu_ps(&b[j][i], ymm_t0);
			_mm256_storeu_ps(&b[j + 1][i], ymm_t1);
			_mm256_storeu_ps(&b[j + 2][i], ymm_t2);
			_mm256_storeu_ps(&b[j + 3][i], ymm_t3);
			_mm256_storeu_ps(&b[j + 4][i], ymm_t4);
			_mm256_storeu_ps(&b[j + 5][i], ymm_t5);
			_mm256_storeu_ps(&b[j + 6][i], ymm_t6);
			_mm256_storeu_ps(&b[j + 7][i], ymm_t7);
		}
	}
}

void block_transpose_avx(int m, int n, const double **a, double **b)
{
	__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
	__m256d ymm_t0, ymm_t1, ymm_t2, ymm_t3;

	for (int i = 0; i < m; i += 4)
	{
		for (int j = 0; j < n; j += 4)
		{
			// load data from memory
			ymm_a0 = _mm256_loadu_pd(&a[i][j]);
			ymm_a1 = _mm256_loadu_pd(&a[i + 1][j]);
			ymm_a2 = _mm256_loadu_pd(&a[i + 2][j]);
			ymm_a3 = _mm256_loadu_pd(&a[i + 3][j]);
			// matrix transposed
			ymm_t0 = _mm256_shuffle_pd(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 0, 0, 0));
			ymm_t1 = _mm256_shuffle_pd(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 0, 0, 0));
			ymm_t2 = _mm256_shuffle_pd(ymm_a0, ymm_a1, _MM_SHUFFLE(0, 0, 3, 3));
			ymm_t3 = _mm256_shuffle_pd(ymm_a2, ymm_a3, _MM_SHUFFLE(0, 0, 3, 3));
			ymm_a0 = _mm256_permute2f128_pd(ymm_t0, ymm_t1, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_a1 = _mm256_permute2f128_pd(ymm_t2, ymm_t3, _MM_SHUFFLE(0, 2, 0, 0));
			ymm_a2 = _mm256_permute2f128_pd(ymm_t0, ymm_t1, _MM_SHUFFLE(0, 3, 0, 1));
			ymm_a3 = _mm256_permute2f128_pd(ymm_t2, ymm_t3, _MM_SHUFFLE(0, 3, 0, 1));
			// store data into memory
			_mm256_storeu_pd(&b[j][i], ymm_a0);
			_mm256_storeu_pd(&b[j + 1][i], ymm_a1);
			_mm256_storeu_pd(&b[j + 2][i], ymm_a2);
			_mm256_storeu_pd(&b[j + 3][i], ymm_a3);
		}
	}
}
