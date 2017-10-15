# -*- coding: utf-8 -*-
from math import log, exp
import pylab
import sys
#sys.setdefaultencoding('utf-8')

class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0

pos = MyDict()
neg = MyDict()
features = set()
totals = [0,0]
words=[]
text = open('HSWN_WN.txt', 'r').read()
text=text.decode('utf-8')

sense = text.split('\n')

#print sense
pos_dict = {}
neg_dict = {}

for line in sense:
	if line != '[Decode error - output not utf-8]':
		# temp = line.split()
		#try:
		if len(line.split())>=5:
			allwords = line.split()[4]
			allwords = allwords.split(',')
			for word in allwords:
					pos_dict[word] = float(line.split()[2])
					neg_dict[word] = float(line.split()[3])
			

def split():
	pos_lines = open('combined+.txt','r').read().decode('utf-8').split('\n')
	neg_lines = open('comined-.txt','r').read().decode('utf-8').split('\n')
	train_pos_lines = []
	test_pos_lines = []
	train_neg_lines = []
	test_neg_lines = []

	for indx,line in enumerate(pos_lines):
		if indx%5 == 0:
			test_pos_lines.append(line)
		else:
			train_pos_lines.append(line)
	for indx,line in enumerate(neg_lines):
		if indx%5 == 0:
			test_neg_lines.append(line)
		else:
			train_neg_lines.append(line)
	f = open('pos_train.txt','w')
	for line in train_pos_lines:
		f.write(line.encode('utf-8'))
		f.write('\n')
	f.close()
	f = open('neg_train.txt','w')
	for line in train_neg_lines:
		f.write(line.encode('utf-8'))
		f.write('\n')
	f.close()
	f = open('pos_test.txt','w')
	for line in test_pos_lines:
		f.write(line.encode('utf-8'))
		f.write('\n')
	f.close()
	f = open('neg_test.txt','w')
	for line in test_neg_lines:
		f.write(line.encode('utf-8'))
		f.write('\n')
	f.close()


def prune_features():
    """
    Remove features that appear only once.
    """
    global pos, neg
    for k in pos.keys():
        if pos[k] <= 1 and neg[k] <= 1:
            del pos[k]

    for k in neg.keys():
        if neg[k] <= 1 and pos[k] <= 1:
            del neg[k]


def train():
    global pos, neg, totals
    
    text = open('pos_train.txt','r').read()
    text = text.decode('utf-8')
    lines = text.split('\n')

    for line in lines:
    	words = line.split()
    	for word in words:
    		if word in pos:
    			pos[word]+=1
    		else:
    			pos[word] = 1

    text = open('neg_train.txt','r').read()
    text = text.decode('utf-8')
    lines = text.split('\n')

    for line in lines:
    	words = line.split()
    	for word in words:
    		if word in neg:
    			neg[word]+=1
    		else:
    			neg[word] = 1

    prune_features()

    totals[0] = sum(pos.values())
    totals[1] = sum(neg.values())
    
    countdata = (pos, neg, totals)
    print totals
    #cPickle.dump(countdata, open(CDATA_FILE, 'w'))
    return countdata

def classify(text):
    words = set(word for word in text.split() if word in features)
    if (len(words) == 0): return True
    # Probability that word occurs in pos documents
    for word in words:
    	if word not in pos_dict:
    		pos_dict[word] = 0
    	if word not in neg_dict:
    		neg_dict[word] = 0

    pos_prob = sum(log(1.0*((1.0/(2-pos_dict[word]))*pos[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log(1.0*((1.0/(2-neg_dict[word]))*neg[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob


def classify2(text):
    """
    For classification from pretrained data
    """
    words = set(word for word in text.split() if word in pos or word in neg)
    if (len(words) == 0): return True
    # Probability that word occurs in pos documents
    pos_prob = sum(log(1.0* (pos[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log(1.0* (neg[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob

def MI(word):
    """
    Compute the weighted mutual information of a term.
    """
    T = totals[0] + totals[1]
    W = pos[word] + neg[word]
    I = 0
    if W==0:
        return 0
    if neg[word] > 0:
        # doesn't occur in -ve
        I += 1.0* (totals[1] - neg[word]) / T * log (1.0* (totals[1] - neg[word]) * T / (T - W) / totals[1])
        # occurs in -ve
        I += 1.0 * neg[word] / T * log (1.0 * neg[word] * T / W / totals[1])
    if pos[word] > 0:
        # doesn't occur in +ve
        I += 1.0* (totals[0] - pos[word]) / T * log (1.0* (totals[0] - pos[word]) * T / (T - W) / totals[0])
        # occurs in +ve
        I += 1.0* pos[word] / T * log (1.0* pos[word] * T / W / totals[0])
    return I


def test():
    global pos, neg, totals, features
    split()
    train()
    words = list(set(pos.keys() + neg.keys()))
    print "Total no of features:", len(words)
    words.sort(key=lambda w: -MI(w))
    num_features, accuracy = [], []
    bestk = 0
    limit = 50
    #path = "./aclImdb/test/"
    step = 50
    start = 500
    best_accuracy = 0.0
    for w in words[:start]:
        features.add(w)
    for k in xrange(start, 3000, step):
        for w in words[k:k+step]:
            features.add(w)
        correct = 0
        size = 0

        lines = open('pos_test.txt','r').read().decode('utf-8').split('\n')
        for line in lines:
            correct += classify(line) == True
            size += 1

        lines = open('neg_test.txt','r').read().decode('utf-8').split('\n')
        for line in lines:
            correct += classify(line) == False
            size += 1
        #print 1.0*correct/size
        num_features.append(k+step)
        accuracy.append(1.0* correct / size)
        if (1.0*correct / size) > best_accuracy:
            bestk = k
            best_accuracy = 1.0*correct / size
        #print k+step, 1.0* correct / size
   
    features = set(words[:bestk])
    
    pylab.plot(num_features, accuracy)
    pylab.show()

    print 'Best k:', bestk
    print 'Best accuracy: ', best_accuracy

test()
