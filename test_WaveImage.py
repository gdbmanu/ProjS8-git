
import numpy as np
import scipy as sp

from waveimage import WaveImage

def test_init_zeros():
	w = WaveImage()

def test_isPuiss2():
	w = WaveImage()
	print w.isPuiss2()
	w = WaveImage(shape = (15,16))
	print w.isPuiss2()
	
def test_getData():
	w = WaveImage(shape = (8, 8))
	print w.getData()
	w = WaveImage(shape = (9,13))
	print w.getData()
	
def test_toString():
	w = WaveImage(shape = (16, 16))
	print w
	w = WaveImage(shape = (28, 28))
	print w
	
print 'Test 1 :'
test_init_zeros()

print 'Test 2 :'
test_isPuiss2()

print 'Test 3 :'
test_getData()

print 'Test 3 :'
test_toString()
