// Separable Keys bicubic antialiasing with the uint8 quantization of Pillow.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int precision = 22;
constexpr int half = 1 << (precision - 1);

struct Axis {
    std::vector<int> start;
    std::vector<int> count;
    std::vector<int32_t> weights;
    int taps;
};

double cubic(double distance) {
    const double x = std::abs(distance);
    if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5;
    return 0.0;
}

Axis coefficients(int input, int output) {
    const double scale = double(input) / output;
    const double filter_scale = std::max(1.0, scale);
    const double support = 2.0 * filter_scale;
    const double inverse = 1.0 / filter_scale;
    const int taps = 2 * int(std::ceil(support)) + 1;
    Axis axis{std::vector<int>(output), std::vector<int>(output),
              std::vector<int32_t>(output * taps), taps};
    std::vector<double> values(taps);
    for (int o = 0; o < output; ++o) {
        const double center = (o + 0.5) * scale;
        const int lo = std::max(0, int(center - support + 0.5));
        const int hi = std::min(input, int(center + support + 0.5));
        axis.start[o] = lo;
        axis.count[o] = hi - lo;
        double total = 0;
        for (int i = lo; i < hi; ++i) {
            values[i - lo] = cubic((i - center + 0.5) * inverse);
            total += values[i - lo];
        }
        for (int i = lo; i < hi; ++i) {
            const double w = (values[i - lo] / total) * (1 << precision);
            axis.weights[o * taps + i - lo] = int32_t(w + (w < 0 ? -0.5 : 0.5));
        }
    }
    return axis;
}

uint8_t quantize(int32_t value) {
    return uint8_t(std::clamp(value >> precision, 0, 255));
}
}

extern "C" void va_crop_bicubic(const uint8_t* input, uint8_t* output,
                                 int64_t step_c, int64_t step_y, int64_t step_x,
                                 int height, int width, int size, bool flip) {
    const Axis horizontal = coefficients(width, size);
    const Axis vertical = coefficients(height, size);
    std::vector<uint8_t> intermediate(height * size * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < size; ++x) {
            const auto* row = input + y * step_y + horizontal.start[x] * step_x;
            const auto* w = horizontal.weights.data() + x * horizontal.taps;
            int32_t r = half, g = half, b = half;
            for (int k = 0; k < horizontal.count[x]; ++k) {
                const auto* pixel = row + k * step_x;
                r += pixel[0] * w[k];
                g += pixel[step_c] * w[k];
                b += pixel[2 * step_c] * w[k];
            }
            const int dst = (y * 3) * size + x;
            intermediate[dst] = quantize(r);
            intermediate[dst + size] = quantize(g);
            intermediate[dst + 2 * size] = quantize(b);
        }
    }

    std::vector<int32_t> accum(size * 3);
    for (int y = 0; y < size; ++y) {
        std::fill(accum.begin(), accum.end(), half);
        const auto* w = vertical.weights.data() + y * vertical.taps;
        for (int k = 0; k < vertical.count[y]; ++k) {
            const auto* line = intermediate.data() + (vertical.start[y] + k) * size * 3;
            for (int x = 0; x < 3 * size; ++x) accum[x] += line[x] * w[k];
        }
        for (int c = 0; c < 3; ++c) {
            auto* dst = output + (c * size + y) * size;
            for (int x = 0; x < size; ++x)
                dst[flip ? size - 1 - x : x] = quantize(accum[c * size + x]);
        }
    }
}
