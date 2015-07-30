"""Analyzes cross-correlation matrix output from taboo_just_matrix.py"""

# A note on notation.  Throughout the code, mathematical convention is followed for the order of
# arguments in conditional probabilities and variable names.  e.g:
# P(i|w2) means P of i given w2, functions such as P(i,w2) and variables like "pi2" reflect this.

# On the other hand,
# HTML outputs reverse this order to create a display that follows the intuition of "implies"
# or "causes", i.e. the precedent comes first and the consequent second
# which even for maths nerds is easier to read I think

import sys,optparse
import re,numpy,scipy,random
from scipy.sparse import csc_matrix,csr_matrix
import heapq
from bisect import bisect_right

op = optparse.OptionParser("usage: %prog -i INDEXFILE -m MATRIXFILE [options]")

# Filenames
op.add_option("-i",dest="indexfile",help="word index input file (npz)")
op.add_option("-m",dest="matrixfile",help="matrix input file (npy)")
op.add_option("-c",dest="combinedfile",help="combined input file (npy)")
op.add_option("-o",dest="outputbase",help="output base")

# Switches for different outputs
# Warning, not all possible combinations of the options below have been tested!
# If you're trying to do something I haven't, you may need to debug
op.add_option("--why",dest="why",help="interactive query",action="store_true",default=False)
op.add_option("--hist0",dest="histograms0",help="output histogram of preds for 0 outcome",action="store_true",default=False)
op.add_option("--hist1",dest="histograms1",help="output histogram of preds for 1 outcome",action="store_true",default=False)
op.add_option("--hist2",dest="histograms2",help="output bi-histogram of preds for 1 outcome",action="store_true",default=False)
op.add_option("--r0",dest="r0",help="regress zeros and output outliers",action="store_true",default=False)
op.add_option("--r1",dest="r1",help="regress nonzeros and output outliers",action="store_true",default=False)
op.add_option("--obq",dest="outliers_by_quantile",help="output outliers by quantile in 2d histogram",action="store_true",default=False)
op.add_option("--benchmark",dest="benchmark",help="benchmark",action="store_true",default=False)

# Other options
op.add_option("--sparse",dest="sparse",help="use sparse matrices.  Not necessarily faster or less demanding of memory!",action="store_true",default=False)
op.add_option("--greedy",dest="newalg",help="use greedy algorithm",action="store_true",default=False)
op.add_option("--maxlinks",dest="maxlinks",help="maximum links to compute",default=20)
op.add_option("--seed",dest="seed",help="random number seed",default=0)
op.add_option("--prop",dest="proportion",help="proportion of matrix to sample",default=0.00001)
op.add_option("--nq",dest="num_quantiles",help="number of quantiles for histograms",default=10)
op.add_option("--numoutliers",dest="numoutliers",help="number of low/high outliers to output",default=50)

(options,args) = op.parse_args()
if len(args)!=0:
    op.error("Trailing arguments on command line")

# Input is either a pair of files (indexfile/matrixfile) or a combined file
if options.combinedfile==None:
    if options.indexfile==None:
        op.error("Indexfile missing")
    if options.matrixfile==None:
        op.error("Matrixfile missing")

if options.outputbase==None:
    op.error("No output base specified")

if options.outliers_by_quantile:
    options.histograms2 = True

save_command = open(options.outputbase+"_command.txt","w")
save_command.write(" ".join(sys.argv)+"\n")
save_command.close()

maxlinks = int(options.maxlinks)
num_quantiles = int(options.num_quantiles)
numoutliers = int(options.numoutliers)

print "Loading"
if options.combinedfile:
    indexfile = options.combinedfile
else:
    indexfile = options.indexfile

data = numpy.load(indexfile)
allwords = data["index"] # list of all words which also serves as row and column labels for the matrix
numstories = int(data["numstories"]) # number of stories in the corpus

# load xcorrmatrix, the cross correlation matrix (in a loose sense)
# xcorrmatrix is actually a matrix covering the product space of the word list (allwords)
# it contains raw counts of the number of stories in which each pair of words appeared together
if options.combinedfile:
    xcorrmatrix = data["matrix"]
else:
    xcorrmatrix = numpy.load(options.matrixfile)
del data

word_to_index = dict((y,x) for x,y in enumerate(allwords))
wordcounts_from_index = numpy.diag(xcorrmatrix)
n = len(allwords)

print "computing conditional probabilities"
# normalize: divide each column by its diagonal element
# this turns xcorrmatrix into a matrix of conditional probabilities
# i.e. P(A|B) where A, B are the events of words A and B appearing in a story
xcorrmatrix /= wordcounts_from_index

if options.sparse:
    print "converting to sparse matrices"
    col_matrix = csc_matrix(xcorrmatrix)
    row_matrix = csr_matrix(xcorrmatrix)
    del xcorrmatrix
else:
    col_matrix = xcorrmatrix
    row_matrix = xcorrmatrix

infinity = float("inf")
def masked_array_to_enumerated_float_list(x):
    '''appends enumerator; converts masked values to float max so they fall to list end when sorting'''
    return [( (infinity if type(p)==numpy.ma.core.MaskedConstant else p) , i)
             for (i,p) in enumerate(x)]
    
def predict_a_given_b_naive(a,b,verbose=False):
    '''Predicts P(a|b) the probability of a appearing in a story if b is there
        Using naive algorithm
        If output is supplied an empty list, appends pairs of (link word id, probability) to explain reasoning'''
    ai = row_matrix[a,:] #pointer into xcorrow_matrix
    ib = col_matrix[:,b] #pointer into xcorrow_matrix
    # Failprobs are probabilities that a doesn't occur; for all x; 1-P(a|x)P(x|b)
    if options.sparse:
        failprobs = numpy.ma.masked_array(1-(ai.multiply(ib.transpose()).toarray()))[0] # not a pointer
    else:
        failprobs = numpy.ma.masked_array(1-ai*ib.transpose()) # not a pointer
    failprobs[a]=numpy.ma.masked
    failprobs[b]=numpy.ma.masked
    if verbose:
        output = masked_array_to_enumerated_float_list(failprobs)
        output.sort() #masked values go to end
        output = output[0:maxlinks]
        output = [(i,1-p) for (p,i) in output]
    failprobs.sort() #masked values go to end
    # sorting on conditional probabilities, not lifts is fine as they all relate to the same consequent
    # so the order will be the same
    result = 1-numpy.prod(failprobs[0:maxlinks])
    if not verbose:
        return result
    else:
        return result,output

def predict_a_given_b_greedy(a,b,verbose=False):
    '''Predicts P(a|b) the probability of a appearing in a story if b is there
        Using greedy algorithm
        If output is supplied an empty list, appends pairs of (link word id, probability) to explain reasoning'''
    ai = row_matrix[a,:]
    ib = col_matrix[:,b].copy() # we will modify the copy
    ib[b] = 0 # has same effect as ai[b] = 0
    ib[a] = 0
    prediction = 0
    linkprobs = ai*ib.transpose()
    if verbose:
        output = []
    for _ in range(maxlinks):
        best = linkprobs.argmax()
        prediction += linkprobs[best]
        if verbose:
            output += [(best,linkprobs[best])]
        linkprobs *= 1-col_matrix[:,best]
    if not verbose:
        return prediction
    else:
        return prediction,output

predict_a_given_b = predict_a_given_b_greedy if options.newalg else predict_a_given_b_naive

def why_inner(w1,w2):
    '''Compute how prediction of P(w1|w2) was arrived at'''
    i1 = word_to_index[w1]
    i2 = word_to_index[w2]
    pred,links = predict_a_given_b(i1,i2,True)
    corr = col_matrix[i1,i2]
    p1 = float(wordcounts_from_index[i1])/numstories
    summary_data = {'w2':w2,'w1':w1,'predp':pred,'predl':pred/p1,'actualp':corr,'actuall':corr/p1}
    link_data = []
    for i,link_prob in links: # i is index of intermediate word in link
        # pi2 = P(wi|w2), l1i = LIFT(w1|wi) = P(w1|wi)/P(w1)
        pi = float(wordcounts_from_index[i])/numstories
        pi2=col_matrix[i,i2]
        p1i=col_matrix[i1,i]
        li2=pi2/pi
        l1i=p1i/p1
        downward_adjust = link_prob/(p1i*pi2) # measure of what greedy algorithm does
        lift_contribution = link_prob/p1
        link_data += [{'liftcont':lift_contribution,'w2':w2,'pi2':pi2,
                       'li2':li2,'i':allwords[i],'p1i':p1i,'l1i':l1i,'adjust':downward_adjust}]
    return summary_data,link_data

def why(w1,w2):
    '''Explain how prediction of P(w1|w2) was arrived at (for interactive use)'''
    summary_data,link_data = why_inner(w1,w2)
    print "%(w2)s -> %(w1)s\n"\
          "pred: P:%(predp).2f, L:%(predl).2f\n"\
          "actual: P:%(actualp).2f, L:%(actuall).2f"%summary_data
    for link in link_data:
        print "%(liftcont).2f: %(w2)s -P%(pi2).2f-L%(li2).2f-> %(i)s "\
              "-P%(p1i).2f-L%(l1i).2f-> (%(adjust).2f)"%link

summarywidth = 160
linkwidth = 100

def html_why_header(title):
    return '''
        <!doctype html>
        <html lang="en">
        <head>
          <title>%(title)s</title>
          <meta charset="utf-8">
          <link rel="stylesheet" href="//code.jquery.com/ui/1.11.4/themes/smoothness/jquery-ui.css">
          <style>
            #accordion { width: %(totwidth)dpx }
            #accordion .ui-accordion-content {
                font-size: 10pt;
            }
            #accordion .ui-accordion-header {
                font-size: 10pt;
                line-height: 12pt;
                padding: 0px;
                text-indent: 18px;
            }
          </style>
          <script src="//code.jquery.com/jquery-1.10.2.js"></script>
          <script src="//code.jquery.com/ui/1.11.4/jquery-ui.js"></script>
          <script>
          $(function() {
            $( "#accordion" ).accordion({
                active: false,
                collapsible: true,
                heightStyle: "content"
            });
          });
          </script>
        </head>
        <body>
        <h2>%(title)s</h2>
        <div id="accordion">
        <table><tr>
            <td width=%(width)d>Residual</td>
            <td width=%(width)d>Precedent</td>
            <td width=%(width)d>Consequent</td>
            <td width=%(width)d>Predicted <font color=blue>prob</font> , <font color=red>lift</font></td>
            <td width=%(width)d>Actual <font color=blue>prob</font> , <font color=red>lift</font></td>
        </tr></table>
        <div>Expand any line on this table to show reasons for predicted link</div>
    '''%{"width":summarywidth,"title":title,"totwidth":5.6*summarywidth}

def html_why_footer():
    return "</div></body></html>"

def html_why(w1,w2,resid):
    output = ''
    summary_data,link_data = why_inner(w1,w2)
    summary_data['width']=summarywidth
    summary_data['resid']=resid
    output += "<table><tr>"\
              "<td width=%(width)d>%(resid).2f</td>"\
              "<td width=%(width)d>%(w2)s</td>"\
              "<td width=%(width)d>-&gt;&nbsp;&nbsp;&nbsp;%(w1)s</td>"\
              "<td width=%(width)d><font color=blue>%(predp).2f</font> , <font color=red>%(predl).2f</font><td>"\
              "<td width=%(width)d><font color=blue>%(actualp).2f</font> , <font color=red>%(actuall).2f</font><td>"\
              "</tr></table>\n"\
              "<div><font size=small><table><tr>"\
              "<td width=%(width)d><u>Lift contribution</u></td>"\
              "<td width=%(width)d><u>Link 1</u> -<font color=blue>prob</font>-<font color=red>lift</font>-&gt;</td>"\
              "<td width=%(width)d><u>Linking word</u></td>"\
              "<td width=%(width)d><u>Link 2</u> -<font color=blue>prob</font>-<font color=red>lift</font>-&gt;</td>"\
              "<td><u>Downscaling</u></td>"\
              "</tr>"%summary_data
    for link in link_data:
        link['width']=linkwidth
        output += "<tr>"\
                  "<td width=%(width)d>%(liftcont).2f</td>"\
                  "<td width=%(width)d>-<font color=blue>%(pi2).2f</font>"\
                      "-<font color=red>%(li2).2f</font>-&gt;</td>"\
                  "<td width=%(width)d>%(i)s</td>"\
                  "<td width=%(width)d>-<font color=blue>%(p1i).2f</font>"\
                      "-<font color=red>%(l1i).2f</font>-&gt;</td>"\
                  "<td>(%(adjust).2f)</td>"\
                  "</tr>"%link
    output += "</table></font></div>\n"
    return output

def html_why_list(outliers,title):
    output = ""
    output += html_why_header(title)
    for resid,i,j in outliers:
        output += html_why(allwords[i],allwords[j],resid)
    output += html_why_footer()
    return output

def pred_stream(proportion,nonzero,transform = lambda x: x,predtransform = lambda x: x):
    '''Provides stream of predicted and actual lifts on pairs of words with index i and j,
        for input to regressions, histograms, etc.

        Argments
        proportion - proportion of dataset sampled
        nonzero - if true, only returns predictions for nonzero lifts, else returns only zero lifts
        transform - optional function to transform output of lift
        predtransform - optional function to transform output of prediction

        Returns
        Generator for sequence of tuples i,j,lift,predicted_lift
        '''
    
    proportion = float(proportion)
    fnumstories = float(numstories)

    def process(i,j,corr):
            inv_pa = fnumstories/wordcounts_from_index[i]
            return i,j,\
                transform(corr*inv_pa),\
                predtransform(predict_a_given_b(i,j)*inv_pa)

    def printprog(index,end,lastprog):
        prog = float(index)/end*100
        if prog>lastprog:
            print "progress: %.1f%%\r"%prog,
            return prog+0.1

    if proportion==1:
        def next_index(index):
            return index + 1
    else:
        def next_index(index):
            return index + int(random.expovariate(proportion))
                
    random.seed(int(options.seed))
    lastprog = 0
    if nonzero:
        # we assume nonzeros are sparse so find these first
        nzi,nzj = col_matrix.nonzero()
        nnz = len(nzi)
        index = 0
        while index<nnz:
            i = nzi[index]
            j = nzj[index]
            lastprog = printprog(index,nnz,lastprog)
            if i!=j:
                yield process(i,j,col_matrix[i,j])
            index = next_index(index) 
    else:
        # we assume matrix has lots of zeros so just discard nonzeros
        index = 0
        end = n*n
        while index<end:
            i = index/n
            j = index%n
            lastprog = printprog(index,end,lastprog)
            if col_matrix[i,j]==0 and i!=j:
                yield process(i,j,0)
            index = next_index(index) 
                
    print

def get_quantiles(data):
    if not data:
        return []
    data.sort()
    quantiles = []
    for quantile in range(num_quantiles+1):
        index = int(float(len(data)-1)/num_quantiles*quantile)
        quantiles += [data[index]]
    return quantiles

if options.histograms0 or options.histograms1 or options.histograms2:
    if options.histograms0:
        print "computing prediction histogram for zero measurements"
        zpreds = []
        for _,_,_,pred in pred_stream(options.proportion,False):
            zpreds += [pred]
        zero_pred_quantiles = get_quantiles(zpreds)

    if options.histograms1 or options.histograms2:
        print "computing prediction histograms (pred & lift) for nonzero measurements"
        preds = []
        lifts = []
        for _,_,lift,pred in pred_stream(options.proportion,True):
            preds += [pred]
            lifts += [lift]
        nonzero_pred_quantiles = get_quantiles(preds)
        nonzero_lift_quantiles = get_quantiles(lifts)

        if options.histograms2:
            print "computing bi-histogram"
            pred_bin_minima = nonzero_pred_quantiles[:-1]
            lift_bin_minima = nonzero_lift_quantiles[:-1]
            largest = []
            smallest = []
            count = 0
            bins = numpy.zeros((num_quantiles,num_quantiles),int)
            for i,j,lift,pred in pred_stream(options.proportion,True):
                liftbin = bisect_right(lift_bin_minima,lift)-1
                predbin = bisect_right(pred_bin_minima,pred)-1
                bins[liftbin,predbin]+=1
                if options.outliers_by_quantile:
                    resid = float(liftbin-predbin)/num_quantiles
                    if count<numoutliers:
                        heapq.heappush(largest,(resid,i,j))
                        heapq.heappush(smallest,(-resid,i,j))
                    else:
                        heapq.heappushpop(largest,(resid,i,j))
                        heapq.heappushpop(smallest,(-resid,i,j))
                    count += 1
                    
            # correlation bins end up on vertical axis of csv file, pred on horizontal
            numpy.savetxt(options.outputbase+"_bihist.csv",bins,delimiter=",")

            if options.outliers_by_quantile:
                print "outputting outliers by quantile"
                largest.sort()
                largest.reverse()
                smallest.sort()
                smallest.reverse()
                outfile = open(options.outputbase+"_outliers_by_quantile_low.html","w")
                outfile.write(html_why_list(smallest,"Quantile outliers - underperforming"))
                outfile.close()
                outfile = open(options.outputbase+"_outliers_by_quantile_high.html","w")
                outfile.write(html_why_list(largest,"Quantile outliers - overperforming"))
                outfile.close()

    print "writing histograms"
    outfile = open(options.outputbase+"_quantiles.csv","w")
    if options.histograms0:
        outfile.write("pred quantiles for zero lift\n")
        outfile.write("%s\n"%",".join(map(str,zero_pred_quantiles)))
        outfile.write("\n")
    if options.histograms1:
        outfile.write("pred quantiles for nonzero lift\n")
        outfile.write("%s\n"%",".join(map(str,nonzero_pred_quantiles)))
        outfile.write("\n")
        outfile.write("nonzero lift quantiles\n")
        outfile.write("%s\n"%",".join(map(str,nonzero_lift_quantiles)))
    outfile.close()

if options.benchmark:
    print "testing time to iterate through matrix"
    import time
    t1=time.clock()
    for _ in pred_stream(options.proportion,True):
        pass
    t2 = time.clock()
    print t2-t1

if options.r0:
    print "counting nonzeros"
    if options.sparse:
        num_nonzeros = col_matrix.nnz
    else:
        num_nonzeros = sum((1 for x in col_matrix.flat if x>0))
    num_zeros = n*n-num_nonzeros
    print num_zeros,"zeros and",num_nonzeros,"non zeros"

    print "producing ROC curve"
    preds = []
    zpreds = []
    for _,_,_,pred in pred_stream(options.proportion,False):
        zpreds += [pred]
    for _,_,_,pred in pred_stream(options.proportion,True):
        preds += [pred]
            
    zpreds.sort()
    preds.sort()

    def sens_spec(cutoff):
        spec = float(bisect_right(zpreds,cutoff))/len(zpreds)
        sens = 1-float(bisect_right(preds,cutoff))/len(preds)
        return sens,spec

    def p_nonzero(cutoff):
        num_zeros_above = len(zpreds)-bisect_right(zpreds,cutoff)
        num_nonzeros_above = len(preds)-bisect_right(preds,cutoff)
        return float(num_nonzeros_above)*num_nonzeros/num_zeros/(num_zeros_above+num_nonzeros_above)

    allpreds = zpreds + preds
    ss=[]
    outfile=open(options.outputbase+"_ROC.csv","w")
    for cutoff in get_quantiles(allpreds,100):
        sens,spec = sens_spec(cutoff)
        ss += [(sens+spec,cutoff,sens,spec)]
        outfile.write("%f,%f,%f\n"%(cutoff,sens,spec))
    outfile.close()
    ss.sort()
    combined,binary_cutoff,sens,spec = ss[-1]
    del allpreds,ss
    print "best combined %.3f sens: %.3f, spec: %.3f, threshold %f"%(combined,sens,spec,binary_cutoff)
    print "P(lift>0|pred>threshold) =",p_nonzero(binary_cutoff)
    print "computing resids for corr=0"
    largest = []
    smallest = []
    for count,(i,j,_,pred) in enumerate(pred_stream(options.proportion,False)):
        resid = -pred
        if count<numoutliers:
            heapq.heappush(largest,(resid,i,j))
            heapq.heappush(smallest,(-resid,i,j))
        else:
            heapq.heappushpop(largest,(resid,i,j))
            heapq.heappushpop(smallest,(-resid,i,j))

    print "outputting"
    largest.sort()
    largest.reverse()
    smallest.sort()
    smallest.reverse()
    outfile = open(options.outputbase+"_resids_zero_low.html","w")
    outfile.write(html_why_list(smallest,"Non occuring outliers"))
    outfile.close()

if options.r1:
    print "regressing nonzero lifts"
    print "summing"
    count = 0
    sum_corr = 0
    sum_pred = 0
    for _,_,corr,pred in pred_stream(options.proportion,True,numpy.log):
        count += 1
        sum_corr += corr
        sum_pred += pred
    mean_corr = sum_corr/count
    mean_pred = sum_pred/count

    print "computing stds"
    var_corr = 0
    var_pred = 0
    covar = 0
    for _,_,corr,pred in pred_stream(options.proportion,True,numpy.log):
        var_corr += pow(corr-mean_corr,2)
        var_pred += pow(pred-mean_pred,2)
        covar += (corr-mean_corr)*(pred-mean_pred)
    var_corr /= n
    var_pred /= n
    covar /= n

    rho = covar/pow(var_corr*var_pred,0.5)
    slope = covar/var_pred 
    print "rho",rho
    print "slope",slope

    print "computing resids for corr=0, no log"
    largest = []
    smallest = []
    for count,(i,j,corr,pred) in enumerate(pred_stream(options.proportion,True,numpy.log)):
        resid = (corr-mean_corr)-(pred-mean_pred)*slope
        if count<numoutliers:
            heapq.heappush(largest,(resid,i,j))
            heapq.heappush(smallest,(-resid,i,j))
        else:
            heapq.heappushpop(largest,(resid,i,j))
            heapq.heappushpop(smallest,(-resid,i,j))

    print "outputting"
    largest.sort()
    largest.reverse()
    smallest.sort()
    smallest.reverse()
    outfile = open(options.outputbase+"_resids_nonzero_low.html","w")
    outfile.write(html_why_list(smallest,"Links lowest below prediction (regressed)"))
    outfile.close()
    outfile = open(options.outputbase+"_resids_nonzero_high.html","w")
    outfile.write(html_why_list(largest,"Links highest above prediction (regressed)"))
    outfile.close()
  
if options.why:
    import IPython
    IPython.embed()
        
