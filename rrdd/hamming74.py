# based on code from https://github.com/shakthisachintha/hamming-7-4-python
import numpy

class Hamming:
    '''
    Hamming (7,4) error correction code implementation.
    Can be used to encode, parity check, error correct, decode and get the orginal message back.
    This can detect two bit errors and correct single bit errors.
    '''
    _gT = numpy.matrix([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [
        0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    _h = numpy.matrix(
        [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
    
    _R = numpy.matrix([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [
        0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
    
    
    def encode(self, message):
        '''
        @Input
        Numpy matrix column vector
        4 bit message to encode
          can be done with: numpy.matrix([1, 0, 0, 1]).transpose()
        @Output
        Numpy matrix column vector
        Encoded 7 bit binary string
        '''
        en = numpy.dot(self._gT, message) % 2
        return en
    
    
    def parityCheck(self, message):
        '''
        @Input
        Numpy matrix a column vector of length 7
        Accepts a binary column vector
        
        @Output
        Numpy row vector of length 3
        Returns the single bit error location as row vector
        '''
        z = numpy.dot(self._h, message) % 2
        return numpy.fliplr(z.transpose())
    
    
    def decode(self, message):
        '''
        @Input
        Numpy matrix a column vector of length 7
        Accepts a binary column vector
        
        @Output
        List of length 4
        Returns the single bit error location as row vector ()
        '''
        ep = self.parityCheck(message)
        pos = self._binatodeci(ep)
        if pos > 0:
            correctMessage = self._flipbit(message, pos)
        else:
            correctMessage = message
        
        origMessage = numpy.dot(self._R, correctMessage)
        return origMessage.transpose().tolist()
    
    
    def _flipbit(self, enc, bitpos):
        '''
        @Input
          enc:Numpy matrix a column vector of length 7
          Accepts a binary column vector
          
          bitpos: Integer value of the position to change
          flip the bit. Value should be on range 1-7
          
        @Output
          Numpy matrix a column vector of length 7
          Returns the bit flipped matrix
        '''
        enc = enc.transpose().tolist()
        bitpos = bitpos - 1
        if (enc[0][bitpos] == 1):
            enc[0][bitpos] = 0
        else:
                enc[0][bitpos] = 1
        return numpy.matrix(enc).transpose()
    
    
    def _binatodeci(self, binaryList):
        '''
        @Input
        Numpy matrix column or row one dimension
        
        @Output
        Decimal number equal to the binary matrix
        '''
        return sum(val*(2**idx) for idx, val in enumerate(reversed(binaryList.tolist()[0])))

