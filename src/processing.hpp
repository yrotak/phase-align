#include "dr_wav.h"
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <atomic>

#ifndef PROCESSING_HPP
#define PROCESSING_HPP

struct AudioFile
{
    unsigned int channels;
    unsigned int sampleRate;
    drwav_uint64 totalSampleCount;
    float *samples;
    std::string name;
    std::string filename;
};

struct Complex
{
    float re, im;

    Complex(float r = 0.0f, float i = 0.0f) : re(r), im(i) {}

    Complex operator+(const Complex &other) const { return {re + other.re, im + other.im}; }
    Complex operator-(const Complex &other) const { return {re - other.re, im - other.im}; }
    Complex operator*(const Complex &other) const
    {
        return {re * other.re - im * other.im, re * other.im + im * other.re};
    }

    Complex operator/(float scalar) const { return {re / scalar, im / scalar}; }

    float norm() const { return std::sqrt(re * re + im * im); }
};

class Processing
{
public:
    static std::vector<float> generateFilter(int f_low, int f_high, int sampleRate, int order)
    {
        if (order % 2 == 0)
            order++;
        const int M = (order - 1) / 2;
        std::vector<float> coeffs(order);

        const float T = 1.0f / sampleRate;
        const float twoPiT = 2.0f * M_PI * T;
        const float invPi = 1.0f / M_PI;
        const float centerCoeff = 2.0f * (f_high - f_low) * T;
        const float windowScale = (order > 1) ? 2.0f * M_PI / (order - 1) : 0.0f;

        for (int n = 0; n < order; ++n)
        {
            const int k = n - M;
            float h;

            if (k == 0)
            {
                h = centerCoeff;
            }
            else
            {
                const float angle = k * twoPiT;
                const float term1 = sin(angle * f_high);
                const float term2 = sin(angle * f_low);
                h = (term1 - term2) * (invPi / k);
            }
            coeffs[n] = h * (0.54f - 0.46f * std::cos(windowScale * n));
        }

        return coeffs;
    }

    static std::vector<float> applyFilter(const std::vector<float> &signal, const std::vector<float> &coeffs)
    {
        int order = coeffs.size();
        std::vector<float> output(signal.size(), 0.0f);
        for (size_t n = 0; n < signal.size(); ++n)
        {
            for (int k = 0; k < order; ++k)
            {
                if (n >= k)
                {
                    output[n] += coeffs[k] * signal[n - k];
                }
            }
        }
        return output;
    }

    static void fft(std::vector<Complex> &a, bool inverse = false)
    {
        int n = a.size();
        if (n <= 1)
            return;

        std::vector<Complex> even(n / 2), odd(n / 2);
        for (int i = 0; i < n / 2; i++)
        {
            even[i] = a[2 * i];
            odd[i] = a[2 * i + 1];
        }

        fft(even, inverse);
        fft(odd, inverse);

        for (int k = 0; k < n / 2; k++)
        {
            float angle = 2 * M_PI * k / n * (inverse ? 1 : -1);
            Complex twiddle(std::cos(angle), std::sin(angle));
            Complex t = twiddle * odd[k];
            a[k] = even[k] + t;
            a[k + n / 2] = even[k] - t;
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                a[i] = a[i] / 2.0f;
            }
        }
    }

    static std::vector<float> hilbertEnvelope(const std::vector<float> &input)
    {
        int n = input.size();
        std::vector<Complex> spectrum(n);
        for (int i = 0; i < n; ++i)
            spectrum[i] = Complex(input[i], 0.0f);

        fft(spectrum);

        for (int i = 0; i < n; ++i)
        {
            if (i == 0 || (n % 2 == 0 && i == n / 2))
            {
                spectrum[i] = spectrum[i];
            }
            else if (i < n / 2)
            {
                spectrum[i] = spectrum[i] * Complex(2.0f, 0.0f);
            }
            else
            {
                spectrum[i] = Complex(0.0f, 0.0f);
            }
        }

        fft(spectrum, true);

        std::vector<float> envelope(n);
        for (int i = 0; i < n; ++i)
            envelope[i] = spectrum[i].norm();

        return envelope;
    }

    static int findStart(AudioFile &ref, float threshold = 0.1f)
    {
        std::vector<float> signal(ref.samples, ref.samples + ref.totalSampleCount);
        std::vector<float> coeffs = generateFilter(300, 3000, ref.sampleRate, 101);
        std::vector<float> filteredSignal = applyFilter(signal, coeffs);

        std::vector<float> envelope = hilbertEnvelope(filteredSignal);

        float maxEnv = *std::max_element(envelope.begin(), envelope.end());

        for (int i = 0; i < envelope.size(); i++)
        {
            if (envelope[i] > maxEnv * threshold)
            {
                return i;
            }
        }

        return -1;
    }

    static int findReferencePosition(const std::vector<float> &target,
                                     const std::vector<float> &ref,
                                     float *maxCorr = nullptr)
    {
        const size_t N = target.size();
        const size_t M = ref.size();

        if (N < M || M == 0)
        {
            return -1;
        }

        float ref_norm = 0.0f;
        for (float v : ref)
            ref_norm += v * v;
        ref_norm = std::sqrt(ref_norm);
        if (ref_norm < 1e-9f)
            ref_norm = 1e-9f;

        int bestIndex = 0;
        float bestCorr = -1.0f;

        for (size_t i = 0; i <= N - M; ++i)
        {
            float dot = 0.0f;
            float tgt_norm = 0.0f;

            for (size_t j = 0; j < M; ++j)
            {
                float a = ref[j];
                float b = target[i + j];
                dot += a * b;
                tgt_norm += b * b;
            }

            tgt_norm = std::sqrt(tgt_norm);
            if (tgt_norm < 1e-9f)
                tgt_norm = 1e-9f;

            float corr = dot / (ref_norm * tgt_norm);
            if (corr > bestCorr)
            {
                bestCorr = corr;
                bestIndex = static_cast<int>(i);
            }
        }

        if (maxCorr)
            *maxCorr = bestCorr;

        return bestIndex;
    }

    static std::vector<float> downsample(const std::vector<float> &signal, int factor)
    {
        if (factor <= 1)
            return signal;

        std::vector<float> result;
        result.reserve(signal.size() / factor);

        for (size_t i = 0; i < signal.size(); i += factor)
        {
            result.push_back(signal[i]);
        }

        return result;
    }

    static std::pair<int, float> findCorrelationShift(
        const std::vector<float> &ref,
        const std::vector<float> &signal)
    {
        if (ref.empty() || signal.empty() || ref.size() > signal.size())
        {
            return {-1, 0.0f};
        }

        const size_t n = ref.size();
        const size_t m = signal.size();
        const int max_shift = m - n;

        const float ref_norm = std::sqrt(std::inner_product(ref.begin(), ref.end(), ref.begin(), 0.0f));

        int best_shift = 0;
        float max_corr = -1.0f;

        for (int shift = 0; shift <= max_shift; ++shift)
        {
            float dot_product = 0.0f;
            float signal_norm = 0.0f;

            for (size_t i = 0; i < n; ++i)
            {
                const float s = signal[shift + i];
                dot_product += ref[i] * s;
                signal_norm += s * s;
            }

            signal_norm = std::sqrt(signal_norm);

            if (signal_norm < 1e-10f || ref_norm < 1e-10f)
            {
                continue;
            }

            const float corr = dot_product / (ref_norm * signal_norm);

            if (corr > max_corr)
            {
                max_corr = corr;
                best_shift = shift;
            }
        }

        return {best_shift, max_corr};
    }

    static std::pair<int, float> findCorrelationShiftOptimized(
        const std::vector<float> &ref,
        const std::vector<float> &signal,
        int step = 10)
    {
        if (ref.empty() || signal.empty() || ref.size() > signal.size())
        {
            return {-1, 0.0f};
        }

        auto coarse_result = findCorrelationShift(
            downsample(ref, step),
            downsample(signal, step));

        const int coarse_shift = coarse_result.first * step;
        const int search_radius = step * 2;
        const int start = std::max(0, coarse_shift - search_radius);
        const int end = std::min(static_cast<int>(signal.size() - ref.size()), coarse_shift + search_radius);

        int best_shift = coarse_shift;
        float max_corr = coarse_result.second;

        for (int shift = start; shift <= end; ++shift)
        {
            const auto [current_shift, current_corr] = findCorrelationShift(
                ref,
                std::vector<float>(signal.begin() + shift, signal.begin() + shift + ref.size()));

            if (current_corr > max_corr)
            {
                max_corr = current_corr;
                best_shift = shift + current_shift;
            }
        }

        return {best_shift, max_corr};
    }

    static std::vector<float> alignSignalToRef(const std::vector<float> &signal, const std::vector<float> &ref, int radius = 100)
    {
        int n = signal.size();
        int m = ref.size();

        if (n == 0 || m == 0)
            return signal;

        std::vector<std::vector<char>> directions;
        std::vector<float> prev_row(2 * radius + 1, std::numeric_limits<float>::max());
        std::vector<float> curr_row(2 * radius + 1, std::numeric_limits<float>::max());

        int j_min = std::max(0, -radius);
        int j_max = std::min(m - 1, radius);
        directions.emplace_back(2 * radius + 1, 'X');

        for (int j = j_min; j <= j_max; ++j)
        {
            int k = j + radius;
            float cost = std::abs(signal[0] - ref[j]);
            if (j == j_min)
            {
                curr_row[k] = cost;
                directions[0][k] = 'S';
            }
            else
            {
                curr_row[k] = curr_row[k - 1] + cost;
                directions[0][k] = 'H';
            }
        }
        std::swap(prev_row, curr_row);

        for (int i = 1; i < n; ++i)
        {
            j_min = std::max(0, i - radius);
            j_max = std::min(m - 1, i + radius);
            directions.emplace_back(2 * radius + 1, 'X');
            std::fill(curr_row.begin(), curr_row.end(), std::numeric_limits<float>::max());

            for (int j = j_min; j <= j_max; ++j)
            {
                int k = j - i + radius;
                float cost = std::abs(signal[i] - ref[j]);
                float min_cost = std::numeric_limits<float>::max();
                char dir = 'X';

                if (j > j_min && curr_row[k - 1] < min_cost)
                {
                    min_cost = curr_row[k - 1];
                    dir = 'H';
                }
                if (prev_row[k] < min_cost)
                {
                    min_cost = prev_row[k];
                    dir = 'D';
                }
                if (k + 1 < prev_row.size() && prev_row[k + 1] < min_cost)
                {
                    min_cost = prev_row[k + 1];
                    dir = 'V';
                }

                if (min_cost != std::numeric_limits<float>::max())
                {
                    curr_row[k] = cost + min_cost;
                    directions[i][k] = dir;
                }
            }
            std::swap(prev_row, curr_row);
        }

        std::vector<int> aligned_indices(n, -1);
        int i = n - 1;
        int j = m - 1;

        while (i >= 0 && j >= 0)
        {
            int k = j - i + radius;
            if (k < 0 || k >= directions[i].size())
                break;

            aligned_indices[i] = j;
            switch (directions[i][k])
            {
            case 'H':
                j--;
                break;
            case 'D':
                i--;
                j--;
                break;
            case 'V':
                i--;
                break;
            case 'S':
                i = -1;
                break;
            default:
                i = -1;
                break;
            }
        }

        std::vector<float> aligned_signal;
        aligned_signal.reserve(n);
        for (i = 0; i < n; ++i)
        {
            if (aligned_indices[i] != -1)
            {
                aligned_signal.push_back(signal[i]);
            }
            else
            {
                aligned_signal.push_back(signal[i]);
            }
        }

        return aligned_signal;
    }
};

#endif