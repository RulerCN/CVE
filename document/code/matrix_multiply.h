#include <intrin.h>

template <class T>
void mat_mul(int m, int p, int n, const T **a, const T **b, T **c)
{
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			for (int k = 0; k < p; ++k)
				c[i][j] += a[i][k] * b[k][j];
}

template <class T>
void mat_mul_v1(int m, int p, int n, const T **a, const T **b, T **c)
{
	for (int i = 0; i < m; ++i)
	{
		for (int k = 0; k < p; ++k)
		{
			T tmp = a[i][k];
			for (int j = 0; j < n; ++j)
				c[i][j] += tmp * b[k][j];
		}
	}
}

template <class T>
void block_mat_mul(int m, int p, int n, const T **a, const T **b, T **c)
{
	for (int i = 0; i < m; ++i)
	{
		for (int k = 0; k < p; k += 4)
		{
			T a0 = a[i][k];
			T a1 = a[i][k + 1];
			T a2 = a[i][k + 2];
			T a3 = a[i][k + 3];
			for (int j = 0; j < n; j += 4)
			{
				T c0 = a0 * b[k][j];
				T c1 = a0 * b[k][j + 1];
				T c2 = a0 * b[k][j + 2];
				T c3 = a0 * b[k][j + 3];
				c0 += a1 * b[k + 1][j];
				c1 += a1 * b[k + 1][j + 1];
				c2 += a1 * b[k + 1][j + 2];
				c3 += a1 * b[k + 1][j + 3];
				c0 += a2 * b[k + 2][j];
				c1 += a2 * b[k + 2][j + 1];
				c2 += a2 * b[k + 2][j + 2];
				c3 += a2 * b[k + 2][j + 3];
				c0 += a3 * b[k + 3][j];
				c1 += a3 * b[k + 3][j + 1];
				c2 += a3 * b[k + 3][j + 2];
				c3 += a3 * b[k + 3][j + 3];
				c[i][j] += c0;
				c[i][j + 1] += c1;
				c[i][j + 2] += c2;
				c[i][j + 3] += c3;
			}
		}
	}
}

void block_mat_mul_sse(int m, int p, int n, const float **a, const float **b, float **c)
{
	__m128 xmm_a0, xmm_a1, xmm_a2, xmm_a3;
	__m128 xmm_b0, xmm_b1, xmm_b2, xmm_b3;
	__m128 xmm_c0, xmm_c1, xmm_c2, xmm_c3;

	for (int i = 0; i < m; ++i)
	{
		for (int k = 0; k < p; k += 4)
		{
			xmm_a0 = _mm_set1_ps(a[i][k]);
			xmm_a1 = _mm_set1_ps(a[i][k + 1]);
			xmm_a2 = _mm_set1_ps(a[i][k + 2]);
			xmm_a3 = _mm_set1_ps(a[i][k + 3]);
			for (int j = 0; j < n; j += 4)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_ps(&b[k][j]);
				xmm_b1 = _mm_loadu_ps(&b[k + 1][j]);
				xmm_b2 = _mm_loadu_ps(&b[k + 2][j]);
				xmm_b3 = _mm_loadu_ps(&b[k + 3][j]);
				// return the weighted sum
				xmm_c0 = _mm_mul_ps(xmm_a0, xmm_b0);
				xmm_c1 = _mm_mul_ps(xmm_a1, xmm_b1);
				xmm_c2 = _mm_mul_ps(xmm_a2, xmm_b2);
				xmm_c3 = _mm_mul_ps(xmm_a3, xmm_b3);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_c1);
				xmm_c2 = _mm_add_ps(xmm_c2, xmm_c3);
				xmm_c0 = _mm_add_ps(xmm_c0, xmm_c2);
				// store data into memory
				_mm_storeu_ps(&c[i][j], _mm_add_ps(_mm_loadu_ps(&c[i][j]), xmm_c0));
			}
		}
	}
}

void block_mat_mul_sse2(int m, int p, int n, const double **a, const double **b, double **c)
{
	__m128d xmm_a0, xmm_a1;
	__m128d xmm_b0, xmm_b1;
	__m128d xmm_c0, xmm_c1;

	for (int i = 0; i < m; ++i)
	{
		for (int k = 0; k < p; k += 2)
		{
			xmm_a0 = _mm_set1_pd(a[i][k]);
			xmm_a1 = _mm_set1_pd(a[i][k + 1]);
			for (int j = 0; j < n; j += 2)
			{
				// load data from memory
				xmm_b0 = _mm_loadu_pd(&b[k][j]);
				xmm_b1 = _mm_loadu_pd(&b[k + 1][j]);
				// return the weighted sum
				xmm_c0 = _mm_mul_pd(xmm_a0, xmm_b0);
				xmm_c1 = _mm_mul_pd(xmm_a1, xmm_b1);
				xmm_c0 = _mm_add_pd(xmm_c0, xmm_c1);
				// store data into memory
				_mm_storeu_pd(&c[i][j], _mm_add_pd(_mm_loadu_pd(&c[i][j]), xmm_c0));
			}
		}
	}
}

void block_mat_mul_avx(int m, int p, int n, const float **a, const float **b, float **c)
{
	__m256 ymm_a0, ymm_a1, ymm_a2, ymm_a3, ymm_a4, ymm_a5, ymm_a6, ymm_a7;
	__m256 ymm_b0, ymm_b1, ymm_b2, ymm_b3, ymm_b4, ymm_b5, ymm_b6, ymm_b7;
	__m256 ymm_c0, ymm_c1, ymm_c2, ymm_c3, ymm_c4, ymm_c5, ymm_c6, ymm_c7;

	for (int i = 0; i < m; ++i)
	{
		for (int k = 0; k < p; k += 8)
		{
			ymm_a0 = _mm256_set1_ps(a[i][k]);
			ymm_a1 = _mm256_set1_ps(a[i][k + 1]);
			ymm_a2 = _mm256_set1_ps(a[i][k + 2]);
			ymm_a3 = _mm256_set1_ps(a[i][k + 3]);
			ymm_a4 = _mm256_set1_ps(a[i][k + 4]);
			ymm_a5 = _mm256_set1_ps(a[i][k + 5]);
			ymm_a6 = _mm256_set1_ps(a[i][k + 6]);
			ymm_a7 = _mm256_set1_ps(a[i][k + 7]);
			for (int j = 0; j < n; j += 8)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_ps(&b[k][j]);
				ymm_b1 = _mm256_loadu_ps(&b[k + 1][j]);
				ymm_b2 = _mm256_loadu_ps(&b[k + 2][j]);
				ymm_b3 = _mm256_loadu_ps(&b[k + 3][j]);
				ymm_b4 = _mm256_loadu_ps(&b[k + 4][j]);
				ymm_b5 = _mm256_loadu_ps(&b[k + 5][j]);
				ymm_b6 = _mm256_loadu_ps(&b[k + 6][j]);
				ymm_b7 = _mm256_loadu_ps(&b[k + 7][j]);
				// return the weighted sum
				ymm_c0 = _mm256_mul_ps(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_mul_ps(ymm_a1, ymm_b1);
				ymm_c2 = _mm256_mul_ps(ymm_a2, ymm_b2);
				ymm_c3 = _mm256_mul_ps(ymm_a3, ymm_b3);
				ymm_c4 = _mm256_mul_ps(ymm_a4, ymm_b4);
				ymm_c5 = _mm256_mul_ps(ymm_a5, ymm_b5);
				ymm_c6 = _mm256_mul_ps(ymm_a6, ymm_b6);
				ymm_c7 = _mm256_mul_ps(ymm_a7, ymm_b7);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c1);
				ymm_c2 = _mm256_add_ps(ymm_c2, ymm_c3);
				ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c5);
				ymm_c6 = _mm256_add_ps(ymm_c6, ymm_c7);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c2);
				ymm_c4 = _mm256_add_ps(ymm_c4, ymm_c6);
				ymm_c0 = _mm256_add_ps(ymm_c0, ymm_c4);
				// store data into memory
				_mm256_storeu_ps(&c[i][j], _mm256_add_ps(_mm256_loadu_ps(&c[i][j]), ymm_c0));
			}
		}
	}
}

void block_mat_mul_avx(int m, int p, int n, const double **a, const double **b, double **c)
{
	__m256d ymm_a0, ymm_a1, ymm_a2, ymm_a3;
	__m256d ymm_b0, ymm_b1, ymm_b2, ymm_b3;
	__m256d ymm_c0, ymm_c1, ymm_c2, ymm_c3;

	for (int i = 0; i < m; ++i)
	{
		for (int k = 0; k < p; k += 4)
		{
			ymm_a0 = _mm256_set1_pd(a[i][k]);
			ymm_a1 = _mm256_set1_pd(a[i][k + 1]);
			ymm_a2 = _mm256_set1_pd(a[i][k + 2]);
			ymm_a3 = _mm256_set1_pd(a[i][k + 3]);
			for (int j = 0; j < n; j += 4)
			{
				// load data from memory
				ymm_b0 = _mm256_loadu_pd(&b[k][j]);
				ymm_b1 = _mm256_loadu_pd(&b[k + 1][j]);
				ymm_b2 = _mm256_loadu_pd(&b[k + 2][j]);
				ymm_b3 = _mm256_loadu_pd(&b[k + 3][j]);
				// return the weighted sum
				ymm_c0 = _mm256_mul_pd(ymm_a0, ymm_b0);
				ymm_c1 = _mm256_mul_pd(ymm_a1, ymm_b1);
				ymm_c2 = _mm256_mul_pd(ymm_a2, ymm_b2);
				ymm_c3 = _mm256_mul_pd(ymm_a3, ymm_b3);
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c1);
				ymm_c2 = _mm256_add_pd(ymm_c2, ymm_c3);
				ymm_c0 = _mm256_add_pd(ymm_c0, ymm_c2);
				// store data into memory
				_mm256_storeu_pd(&c[i][j], _mm256_add_pd(_mm256_loadu_ps(&c[i][j]), ymm_c0));
			}
		}
	}
}
