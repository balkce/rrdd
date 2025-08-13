from rrdd import r2d2
import numpy as np

byte_array = [[0,1,1,0,0,1,0,0], [1,1,0,1,1,0,0,0], [1,0,1,0,1,1,0,1], [0,1,0,1,0,1,0,1],
              [1,1,0,0,0,0,0,1], [0,1,1,0,0,0,0,1], [0,0,1,0,0,1,0,0], [0,1,1,0,0,1,0,0]]

xs = r2d2.encode(byte_array)

# do something with the audio data...

# then, decode...

byte_array_received = r2d2.decode(xs,num_bits=8)

# checking for errors:

for index_byte in range(len(byte_array)):
  print("Received:")
  print(byte_array_received[index_byte])
  print("Correct:")
  print(byte_array[index_byte])
  print("Error:")
  err = np.sum(np.abs(np.array(byte_array_received[index_byte]) - np.array(byte_array[index_byte])))
  print(err)
  print("")

