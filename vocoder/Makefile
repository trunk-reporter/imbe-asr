VOCODER_DIR = ../trunk-recorder/lib/op25_repeater/lib/imbe_vocoder

SRCS = \
	$(VOCODER_DIR)/imbe_vocoder.cc \
	$(VOCODER_DIR)/aux_sub.cc \
	$(VOCODER_DIR)/basicop2.cc \
	$(VOCODER_DIR)/ch_decode.cc \
	$(VOCODER_DIR)/ch_encode.cc \
	$(VOCODER_DIR)/dc_rmv.cc \
	$(VOCODER_DIR)/decode.cc \
	$(VOCODER_DIR)/dsp_sub.cc \
	$(VOCODER_DIR)/encode.cc \
	$(VOCODER_DIR)/math_sub.cc \
	$(VOCODER_DIR)/pe_lpf.cc \
	$(VOCODER_DIR)/pitch_est.cc \
	$(VOCODER_DIR)/pitch_ref.cc \
	$(VOCODER_DIR)/qnt_sub.cc \
	$(VOCODER_DIR)/rand_gen.cc \
	$(VOCODER_DIR)/sa_decode.cc \
	$(VOCODER_DIR)/sa_encode.cc \
	$(VOCODER_DIR)/sa_enh.cc \
	$(VOCODER_DIR)/tbls.cc \
	$(VOCODER_DIR)/uv_synt.cc \
	$(VOCODER_DIR)/v_synt.cc \
	$(VOCODER_DIR)/v_uv_det.cc \
	imbe_capi.cpp

CXX = g++
CXXFLAGS = -O2 -fPIC -I$(VOCODER_DIR) -std=c++11 -w
TARGET = libimbe.so

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ -lm

clean:
	rm -f $(TARGET)

.PHONY: clean
