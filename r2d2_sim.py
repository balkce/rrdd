from rrdd import r2d2

import numpy as np

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

# Create byte arrays to encode
byte_array1 = [[0,1,1,0,0,1,0,0], [1,1,0,1,1,0,0,0], [1,0,1,0,1,1,0,1], [0,1,0,1,0,1,0,1],
              [1,1,0,0,0,0,0,1], [0,1,1,0,0,0,0,1], [0,0,1,0,0,1,0,0], [0,1,1,0,0,1,0,0]]

xs1 = r2d2.encode(byte_array1)

# Generate white noise
r = 0.01 * np.random.normal(size=xs1.shape)

# Create encoded signals

# Create a rectangular room with two sources
rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
             box=np.asarray([10.0, 10.0, 2.5]),
             srcs=np.asarray([[2.0, 3.0, 1.0], [8.0, 7.0, 1.5]]),
             origin=np.asarray([4.0, 5.0, 1.25]),
             alphas=0.8 * np.ones(6),
             c=343.0)

# Create room impulse responses
hy = rb.rir(rm)
hxs1 = hy[[0], :, :]
hr = hy[[1], :, :]

# Combine input sources
y = np.concatenate([xs1,r], axis=0)

# Apply room impulse response
xs1s = rb.conv(hxs1, xs1)
rs = rb.conv(hr, r)
ys = rb.conv(hy, y)

# Compute spectrograms
XS1s = fb.stft(xs1s)
Rs = fb.stft(rs)
Ys = fb.stft(ys)

# Compute spatial correlation matrices
XXS1s = sp.scm(sp.xspec(XS1s))
RRs = sp.scm(sp.xspec(Rs))

# Compute mvdr weights
w1s = bf.mvdr(XXS1s, RRs)

# Perform beamforming
Z1s = bf.beam(Ys, w1s)

# Return to time domain
z1s = fb.istft(Z1s)

#vz.spex(Ys)

Xs1 = fb.stft(xs1, frame_size=512, hop_size=512)
vz.spex(Xs1)
vz.spex(Z1s)

#vz.wave(ys)
#vz.wave(xs1)
#vz.wave(z1s)
#vz.wave(z2s)


print("--- byte_array1")
byte_array1_received = r2d2.decode(z1s,num_bits=8, fS=16000, start_f=1000.0)
print(len(byte_array1))
print(len(byte_array1_received))
for index_byte in range(len(byte_array1_received)):
  print("Received:")
  print(byte_array1_received[index_byte])
  print("Correct:")
  print(byte_array1[index_byte+1])
  print("Error:")
  err = np.sum(np.abs(np.array(byte_array1_received[index_byte]) - np.array(byte_array1[index_byte+1])))
  print(err)
  print("")

