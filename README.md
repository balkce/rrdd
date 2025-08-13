# Remote Robotic Data Decoder (R2D2)

## Install

After cloning this repository, and `cd rrdd`, install the `kissdsp` library:

    git clone https://github.com/FrancoisGrondin/kissdsp.git
    cd kissdsp
    python setup.py install
    cd ..

Then, install dependencies:

    pip install -r requirements.txt


## Running

The `r2d2_test.py` provides a good starting point to running the whole system, but for completeness sake, here is a teardown of how to run the encoding/decoding procedure.

1.- Import the `r2d2` library:

    from rrdd import r2d2

2.- Create a list of lists of bits (it is preferable to use 8 bits to form a byte array, but `r2d2` is flexible enough to handle different bit list lengths):

    byte_array = [[0,1,1,0,0,1,0,0], [1,1,0,1,1,0,0,0], [1,0,1,0,1,1,0,1]]

3.- Then encode them into audio data (`xs`):

    xs = r2d2.encode(byte_array)

This function receives the following optional arguments:

- `N`: length (in samples) of byte time window (default: 2048 samples)
- `fS`: sampling rate (default: 16000 Hz)
- `start_f`: byte starting frequency (default: 1000 Hz)
- `bit_bandwidth`: frequency bandwidth for each bit (default: 750 Hz)
- `interbit_bandwidth`: bandwidth between bits (default: 0 Hz)

You then can use the resulting encoded audio data to do as you like.

4.- To decode the audio data, you should do:

    byte_array_received = r2d2.decode(xs,num_bits=8)

This function will return a list of lists of bits (`byte_array_received`) which is the digital information that was encoded into audio data (`xs`).

`rs.decode` receives the same optional arguments as `rs.encode`. It is **essential** that the same values for these optional arguments are used in both the decode and encode functions, as well as that the `num_bits` is the same length as the bit list length of the `byte_array`.
