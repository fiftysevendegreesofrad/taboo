'''Parses text corpus to make cross-correlation matrix
   Which documents how often word pairs appear in the same item together.

   Also output total number of items in corpus, and index of words to matrix cols/rows.
'''

import re
import numpy
import heapq
import sys,optparse
from collections import defaultdict
from bisect import bisect_right
import bz2

op = optparse.OptionParser("usage: %prog -i CORPUSFILE [options]")

op.add_option("-i",dest="infile",help="corpus input file, one story per line, plain text or .bz2")

op.add_option("--num-tasks",dest="num_tasks",help="number of tasks (for PBS batch job usage)",default=1)
op.add_option("--task",dest="task",help="number of current task, indexed from 0 (for PBS batch job usage)",default=0)
op.add_option("--thresh",dest="thresh",help="minumum number of times word can appear for inclusion (filters low frequency words)",default=5)
op.add_option("--max-prop",dest="prop",help="maximum proportion of stories word can appear in for inclusion (filters high frequency words)",default=0.11)
op.add_option("--max-size",dest="max_size",help="maximum number of words to include in matrix",default=20000)

(options,args) = op.parse_args()
if len(args)!=0:
    op.error("Trailing arguments on command line")
    
thresh = int(options.thresh)
prop = float(options.prop)
max_size = int(options.max_size)
subset = int(options.task)
numslices = int(options.num_tasks)
infile = options.infile

if infile is None:
    op.error("No input file supplied")

assert subset<numslices

if numslices!=1:
    print "running subset %d of %d"%(subset,numslices)

def get_lines():
    if infile[-4:].lower()==".bz2":
        return bz2.BZ2File(infile)
    else:
        return open(infile)

stripnonalpha = re.compile("[^A-Za-z ]*")
def prepare(story):
    return stripnonalpha.sub("",story).lower()

allwords = set()

# get word frequency counts
wordcounts = defaultdict(int)
numstories = 0
for line in get_lines():
    numstories += 1
    story = prepare(line)
    words = set(story.split())
    for w in words:
        wordcounts[w] += 1

allwords = sorted([(count,word) for word,count in wordcounts.iteritems()
                   if count>thresh and float(count)/numstories<prop])

allwords.reverse()

n = len(allwords)
print n,"of",len(wordcounts),"words occuring more than",thresh,"times in",numstories,"stories"

if n>max_size:
    print "trimming to ",max_size,"words"
    allwords = allwords[:max_size]

n = len(allwords)
allwords = sorted([w for _,w in allwords])

xcorrmatrix = numpy.zeros((n,n))

word_to_index = {}
wordcounts_from_index = []
for i,w in enumerate(allwords):
    word_to_index[w] = i
    wordcounts_from_index += [wordcounts[w]]

print "making correlation matrix"

for progress,line in enumerate(get_lines()):
    if progress % numslices != subset:
        continue
    if progress%100==0:
        print "computing %.2f%% complete\r"%(float(progress)/numstories*100),
    story = set(prepare(line).split()).intersection(allwords)
    # story presumed much shorter than allwords so iterate over story instead
    for w1 in story:
        for w2 in story:
            xcorrmatrix[word_to_index[w1],word_to_index[w2]] += 1
print


#print "saving..."
outbase = infile+"_%d_of_%d_"%(subset,numslices)
numpy.savez_compressed("%s_combined"%outbase,
                       index=allwords,
                       numstories=numstories,
                       matrix=xcorrmatrix)

# for large datasets is is sometimes necessary to save matrix separately
# due to numpy bug
# numpy.save("%s_matrix"%outbase,xcorrmatrix)
