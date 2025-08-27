import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import kissdsp.filterbank as fb
import kissdsp.visualize as vz
import kissdsp.io as io

def encode(byte_array, N=2048, fS=16000, start_f=1000.0, bit_bandwidth=750.0, interbit_bandwidth=0.0):
  assert(len(byte_array) > 0)
  
  t = np.arange(N)/fS
  t1 = N/fS
  num_bits = len(byte_array[0])
  
  bits_true = np.zeros((num_bits, N), dtype=np.float32)
  bits_false = np.zeros((num_bits, N), dtype=np.float32)
  
  for index_bit in range(num_bits):
    f0 = (start_f + interbit_bandwidth) + (bit_bandwidth * index_bit)
    f1 = (start_f + bit_bandwidth) + (bit_bandwidth * index_bit)
    fmed = (f0+f1)/2
    
    bits_true[index_bit, :] = sig.chirp(t=t, f0=fmed, t1=t1, f1=f1)
    bits_false[index_bit, :] = sig.chirp(t=t, f0=fmed, t1=t1, f1=f0)
  
  buffer = np.zeros((N*len(byte_array),), dtype=np.float32)
  
  for index_byte in range(len(byte_array)):
    time_indexes = np.arange(N*index_byte, N*index_byte+N)
    
    for index_bit in range(num_bits):
      if byte_array[index_byte][index_bit] == 1:
        buffer[time_indexes] += bits_true[index_bit, :]
      else:
        buffer[time_indexes] += bits_false[index_bit, :]
  
  xs = np.expand_dims(buffer, axis=0)
  xs /= np.max(xs)
  
  return xs

def decode(xs, num_bits, N=2048, fS=16000, start_f=1000.0, bit_bandwidth=750.0, interbit_bandwidth=0.0):
  assert(num_bits > 0)
  analysis_N = int(N/4)
  
  bit_freqs = np.zeros(num_bits)
  bit_f0s = np.zeros(num_bits)
  bit_f1s = np.zeros(num_bits)
  
  for index_bit in range(num_bits):
    f0 = (start_f + interbit_bandwidth) + (bit_bandwidth * index_bit)
    f1 = (start_f + bit_bandwidth) + (bit_bandwidth * index_bit)
    fmed = (f0+f1)/2
    
    bit_freqs[index_bit] = fmed
    bit_f0s[index_bit] = f0
    bit_f1s[index_bit] = f1
    
  bit_freqsbin = np.zeros(num_bits,dtype=int)
  bit_f0sbin = np.zeros(num_bits,dtype=int)
  bit_f1sbin = np.zeros(num_bits,dtype=int)
  
  #finding closest freq bins
  w = np.fft.rfftfreq(analysis_N,1/fS)
  for index_bit in range(num_bits):
    bit_freqsbin[index_bit] = np.argmin(np.abs(w-bit_freqs[index_bit]))
    bit_f0sbin[index_bit] = np.argmin(np.abs(w-bit_f0s[index_bit]))
    bit_f1sbin[index_bit] = np.argmin(np.abs(w-bit_f1s[index_bit]))
  
  width_avg = np.mean(bit_f1sbin-bit_f0sbin)
  width_avg_mid = int(width_avg/2)
  
  Xs = fb.stft(xs, frame_size=analysis_N, hop_size=analysis_N)
  
  byte_start = np.zeros(num_bits,dtype=int)
  byte_curr = np.zeros(num_bits,dtype=int)
  byte_last = np.zeros(num_bits,dtype=int)
  byte_new = np.zeros(num_bits,dtype=int)
  byte_array_received = np.zeros(num_bits,dtype=int)
  
  stream_start = True
  bytes_array = []
  byte_index = 0
  for t_index in range(Xs.shape[1]):
    if stream_start:
      for index_bit in range(num_bits):
        byte_start[index_bit] = np.argmax(Xs[0,t_index,bit_f0sbin[index_bit]:bit_f1sbin[index_bit]])
      
      #checking if we are at the start of a byte:
      #  occurs when all high energy frequencies are near (less than 2 bins)
      #  from the middle of their bandwidth
      if np.mean(np.abs(byte_start-width_avg_mid)) < 2:
        byte_start[np.abs(byte_start-width_avg_mid) > 2] = width_avg_mid
        byte_curr = np.copy(byte_start)
        stream_start = False
        print("Stream started...")
        print("")
      
    else:
      byte_last = np.copy(byte_curr)
      for index_bit in range(num_bits):
        byte_curr[index_bit] = np.argmax(Xs[0,t_index,bit_f0sbin[index_bit]:bit_f1sbin[index_bit]])
      
      diff_from_start = np.mean(np.abs(byte_curr - byte_start))
      diff_from_last = np.mean(np.abs(byte_curr - byte_last))
      
      #check for start of new byte:
      #  ocurs if the difference of current to start is less than
      #  current to last
      if diff_from_start >= diff_from_last:
        #still in the same byte
        #  we recreate the byte at each time step, for redundancy
        for index_bit in range(num_bits):
          #a bit value is defined by the difference in frequency bin number
          #compared between current and at the start of the byte time step
          if byte_curr[index_bit] > byte_start[index_bit]:
            byte_array_received[index_bit] = 1
          elif byte_curr[index_bit] < byte_start[index_bit]:
            byte_array_received[index_bit] = 0
          else:
            #if this value is the same as the last, store an UNKNOWN (-1)
            byte_array_received[index_bit] = -1
      else:
        #before adding this byte to the list, let's do a sane check
        for index_bit in range(num_bits):
          if byte_array_received[index_bit] == 1 and byte_last[index_bit] < width_avg_mid:
            print("Byte: "+str(byte_index)+". Sane check failed in bit "+str(index_bit)+" (Bit value: "+str(byte_array_received[index_bit])+", while freq bin with max value ("+str(byte_last[index_bit])+") is lower than mid bin ("+str(width_avg_mid)+"). Fixing to appropriate value.")
            byte_array_received[index_bit] = 0
          elif byte_array_received[index_bit] == 0 and byte_last[index_bit] > width_avg_mid:
            print("Byte: "+str(byte_index)+". Sane check failed in bit "+str(index_bit)+" (Bit value: "+str(byte_array_received[index_bit])+", while freq bin with max value ("+str(byte_last[index_bit])+") is higher than mid bin ("+str(width_avg_mid)+"). Fixing to appropriate value.")
            byte_array_received[index_bit] = 1
          elif byte_array_received[index_bit] == -1:
            if byte_last[index_bit] < width_avg_mid:
              print("Byte: "+str(byte_index)+". Sane check failed in bit "+str(index_bit)+" (Unknown bit value: "+str(byte_array_received[index_bit])+", while freq bin with max value ("+str(byte_last[index_bit])+") is lower than mid bin ("+str(width_avg_mid)+"). Fixing to appropriate value.")
              byte_array_received[index_bit] = 0
            elif byte_last[index_bit] > width_avg_mid:
              print("Byte: "+str(byte_index)+". Sane check failed in bit "+str(index_bit)+" (Unknown bit value: "+str(byte_array_received[index_bit])+", while freq bin with max value ("+str(byte_last[index_bit])+") is higher than mid bin ("+str(width_avg_mid)+"). Fixing to appropriate value.")
              byte_array_received[index_bit] = 1
            else:
              print("Byte: "+str(byte_index)+". Sane check failed in bit "+str(index_bit)+" (Unknown bit value: "+str(byte_array_received[index_bit])+", while freq bin with max value ("+str(byte_last[index_bit])+") is exactly as mid bin ("+str(width_avg_mid)+"). Unable to determine appropriate value. Fixing to 1 to avoid tranmission issues.")
              byte_array_received[index_bit] = 1
        
        #new byte
        bytes_array += [byte_array_received.tolist()]
        byte_index += 1
        
        #resetting start values for the next byte
        byte_curr[np.abs(byte_curr-width_avg_mid) > 2] = width_avg_mid
        byte_start = np.copy(byte_curr)
  
  #adding the last byte to the stream
  for index_bit in range(num_bits):
    if byte_curr[index_bit] > byte_start[index_bit]:
      byte_array_received[index_bit] = 1
    elif byte_curr[index_bit] < byte_start[index_bit]:
      byte_array_received[index_bit] = 0
    else:
      byte_array_received[index_bit] = -1
  bytes_array += [byte_array_received.tolist()]
  
  return bytes_array

