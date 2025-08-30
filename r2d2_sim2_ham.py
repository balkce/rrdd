from rrdd import r2d2

import numpy as np

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

# Communication configuration
bit_bandwidth = 400.0
s1_start_f = 1000.0

# Create byte arrays to encode
s1_end_f = s1_start_f + (bit_bandwidth*7)

s2_start_f = s1_end_f + (bit_bandwidth/2)
s2_end_f = s2_start_f + (bit_bandwidth*7)

if s2_end_f >= 7700:
  print("The end frequency for s2 ("+str(s2_end_f)+") is larger than or to close to the nyquist-shannon frequency (8000).")
  exit()

print("--- bit bandwidth: "+str(bit_bandwidth))
print("--- s1:")
print("       start_f: "+str(s1_start_f))
print("       end_f  : "+str(s1_end_f))
print("--- s2:")
print("       start_f: "+str(s2_start_f))
print("       end_f  : "+str(s2_end_f))

word_array1 = [[0,1,1,0], [0,1,0,0], [1,1,0,1], [1,0,0,0], [1,0,1,0], [1,1,0,1], [0,1,0,1], [0,1,0,1],
               [1,1,0,0], [0,0,0,1], [0,1,1,0], [0,0,0,1], [0,0,1,0], [0,1,0,0], [0,1,1,0], [0,1,0,0]]
word_array2 = [[0,0,0,0], [0,0,0,0], [1,1,1,1], [1,1,1,1], [0,0,0,0], [1,1,1,1], [1,1,1,1], [0,0,0,0],
               [1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1], [0,1,0,1], [0,1,0,1], [1,0,1,0], [1,0,1,0]]

xs1 = r2d2.encode_ham(word_array1, start_f=s1_start_f, bit_bandwidth=bit_bandwidth)
xs2 = r2d2.encode_ham(word_array2, start_f=s2_start_f, bit_bandwidth=bit_bandwidth)

# Generate white noise
r = 0.01 * np.random.normal(size=xs1.shape)

# Create encoded signals

# Create a rectangular room with two sources
rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
             box=np.asarray([10.0, 10.0, 2.5]),
             srcs=np.asarray([[2.0, 3.0, 1.0], [1.0, 5.0, 1.0], [8.0, 7.0, 1.5]]),
             origin=np.asarray([4.0, 5.0, 1.25]),
             alphas=0.8 * np.ones(6),
             c=343.0)

# Create room impulse responses
hy = rb.rir(rm)
hxs1 = hy[[0], :, :]
hxs2 = hy[[1], :, :]
hr = hy[[2], :, :]

# Combine input sources
y = np.concatenate([xs1,xs2,r], axis=0)

# Apply room impulse response
xs1s = rb.conv(hxs1, xs1)
xs2s = rb.conv(hxs2, xs2)
rs = rb.conv(hr, r)
ys = rb.conv(hy, y)

# Compute spectrograms
XS1s = fb.stft(xs1s)
XS2s = fb.stft(xs2s)
Rs = fb.stft(rs)
Ys = fb.stft(ys)

# Compute spatial correlation matrices
XXS1s = sp.scm(sp.xspec(XS1s))
XXS2s = sp.scm(sp.xspec(XS2s))
RRs = sp.scm(sp.xspec(Rs))

# Compute mvdr weights
w1s = bf.mvdr(XXS1s, XXS2s+RRs)
w2s = bf.mvdr(XXS2s, XXS1s+RRs)

# Perform beamforming
Z1s = bf.beam(Ys, w1s)
Z2s = bf.beam(Ys, w2s)

# Return to time domain
z1s = fb.istft(Z1s)
z2s = fb.istft(Z2s)

#vz.spex(Ys)

Xs1 = fb.stft(xs1, frame_size=512, hop_size=512)
vz.spex(Xs1)
vz.spex(Z1s)

Xs2 = fb.stft(xs2, frame_size=512, hop_size=512)
vz.spex(Xs2)
vz.spex(Z2s)

#vz.wave(ys)
#vz.wave(xs1)
#vz.wave(z1s)
#vz.wave(z2s)


print("--- word_array1")
word_array1_received = r2d2.decode_ham(z1s, start_f=s1_start_f, bit_bandwidth=bit_bandwidth)
min_len = np.min(np.array([len(word_array1)-1, len(word_array1_received)]))
for index_byte in range(min_len):
  print("Received:")
  print(word_array1_received[index_byte])
  print("Correct:")
  print(word_array1[index_byte+1])
  print("Error:")
  err = np.sum(np.abs(np.array(word_array1_received[index_byte]) - np.array(word_array1[index_byte+1])))
  print(err)
  print("")

print("--- word_array2")
word_array2_received = r2d2.decode_ham(z2s, start_f=s2_start_f, bit_bandwidth=bit_bandwidth)
min_len = np.min(np.array([len(word_array2)-1, len(word_array2_received)]))
for index_byte in range(min_len):
  print("Received:")
  print(word_array2_received[index_byte])
  print("Correct:")
  print(word_array2[index_byte+1])
  print("Error:")
  err = np.sum(np.abs(np.array(word_array2_received[index_byte]) - np.array(word_array2[index_byte+1])))
  print(err)
  print("")

