'''Optional script to merge multiple matrices computed as PBS jobs.
Will probably need some bodging to match whatever your file naming scheme is.
Currently it assumes separate files for word lists and matrices.'''

import sys
import numpy
import re
import shutil

infiles = sys.argv[1:]
wordfiles = [f for f in infiles if re.match(r".*_words\.npz$",f)]
matrixfiles = [f for f in infiles if re.match(r".*_matrix\.npy$",f)]

root = re.match("(.*)_[0-9]*_of_[0-9]*__words\.npz",wordfiles[0]).group(1)
outbase = root+".all"

# check all _words.npz match

zf = numpy.load(wordfiles[0])
wordsname = "index"
storycountname = "numstories"
first_words = zf[wordsname]
first_numstories = zf[storycountname]
del zf
for f in wordfiles[1:]:
	zf = numpy.load(f)
	assert (zf[wordsname]==first_words).all()
	assert zf[storycountname]==first_numstories
	del zf

# add all _matrix.npy

first_matrix = numpy.load(matrixfiles[0])
for f in matrixfiles[1:]:
	first_matrix += numpy.load(f)

numpy.save(outbase+"_matrix.npy",first_matrix)
shutil.copyfile(wordfiles[0],outbase+"_words.npz")
