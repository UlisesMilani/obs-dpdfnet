// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <kiss_fftr.h>

#include <vector>

inline constexpr double kPi = 3.141592653589793238462643383279502884;

// Streaming 50%-overlap STFT/ISTFT with the normalization expected by the
// DPDFNet 48 kHz HR reference implementation. analysis() writes the normalized
// real/imaginary spectrum directly into the model's input tensor, and
// synthesis() applies the inverse normalization before overlap-add.
class StreamingStft {
public:
  StreamingStft(int n_fft, int hop_size);
  ~StreamingStft();

  StreamingStft(const StreamingStft &) = delete;
  StreamingStft &operator=(const StreamingStft &) = delete;

  void reset();
  // spec is freq_bins * 2 interleaved floats; the caller guarantees the size.
  void analysis(const std::vector<float> &frame, float *spec);
  void synthesis(const float *spec, std::vector<float> &hop);

  int freq_bins() const { return freq_bins_; }

private:
  int n_fft_ = 0;
  int hop_size_ = 0;
  int freq_bins_ = 0;

  kiss_fftr_cfg forward_ = nullptr;
  kiss_fftr_cfg inverse_ = nullptr;

  float spectrum_scale_ = 1.0f;
  float synthesis_scale_ = 1.0f;

  std::vector<float> window_;
  std::vector<float> time_frame_;
  std::vector<float> ola_buffer_;
};
