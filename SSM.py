from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
import time
from collections import Counter
import math
from operator import itemgetter

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import ast
import json
import numpy as np
import csv
import multiprocessing as mp
from pyemd import emd_with_flow, emd

#youtubedictn = {}
distances = []
docs = []

def narrative_filter():
	with open("Youtube_with_narritive.csv",'w') as wf:
		Youtubewriter = csv.writer(wf)
		with open("Youtube_text_ex.csv","r") as rf:
			Youtubereader = csv.reader(rf)
			for input in Youtubereader:
				#input = line[0].split("")
				if input[1]!="":
					Youtubewriter.writerow([input[0],input[1],input[2]])
				#print(line)
	
def preprocess():
	wv = KeyedVectors.load_word2vec_format(datapath("GoogleNews-vectors-negative300.bin.gz"), binary=True)
	sw = set(stopwords.words('english'))
	youtubedict = {}
	youtubetext = []

	with open("Youtube_with_narritive.csv",'r') as rf:
		Youtubereader = csv.reader(rf)
		for input in Youtubereader:
			tokens = nltk.word_tokenize(input[2])
			words = [w.lower() for w in tokens if w.lower() not in sw and w.lower() in wv.vocab and w.lower().isalnum()]
			for word in words:
				if word not in youtubedict.keys():
					length = len(youtubedict)
					youtubedict[word] = length
					#youtubedictn[length] = word
			numbers = [youtubedict[word] for word in words]
			youtubetext.append([input[0],numbers])

	with open("Youtube_words.csv",'w') as wf1:
		Youtubewriter = csv.writer(wf1)
		for key, value in youtubedict.items():
			Youtubewriter.writerow([key,value])

	with open("Youtube_doc.csv",'w') as wf2:
		Youtubewriter = csv.writer(wf2)
		for item in youtubetext:
			Youtubewriter.writerow(item)

#preprocess()
def compute_words_distances():
	wv = KeyedVectors.load_word2vec_format(datapath("GoogleNews-vectors-negative300.bin.gz"), binary=True)
	words = []
	distances = []
	with open("Youtube_words.csv",'r') as rf:
		Youtubereader = csv.reader(rf)
		for input in Youtubereader:
			words.append(input[0])

	for w1 in words:
		v1 = wv[w1]
		distance = []
		for w2 in words:
			v2 = wv[w2]
			distance.append(np.sqrt(np.sum((v1 - v2)**2)))
		distances.append(distance)

	with open("Youtube_words_distance.csv",'w') as wf:
		Youtubewriter = csv.writer(wf)
		for dis in distances:
			Youtubewriter.writerow(dis)	

def SSM_setup(document1, document2, rate):
	len_pre_oov1 = len(document1)
	len_pre_oov2 = len(document2)
	n = float(len_pre_oov1)*rate
	if n>float(len_pre_oov2):
		n = float(len_pre_oov2)

	diff1 = len_pre_oov1 - len(document1)
	diff2 = len_pre_oov2 - len(document2)
	if diff1 > 0 or diff2 > 0:
	    logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

	if not document1 or not document2:
	    logger.info(
	        "At least one of the documents had no words that were in the vocabulary. "
	        "Aborting (returning inf)."
	    )
	    return float('inf')

	dictionary = Dictionary(documents=[document1, document2])
	vocab_len = len(dictionary)

	if vocab_len == 1:
	    # Both documents are composed by a single unique token
	    return 0.0

	scale1 = float(len_pre_oov1)/n 
	scale2 = float(len_pre_oov2)/n 

        # Sets for faster look-up.
	docset1 = set(document1)
	docset2 = set(document2)

	distance_matrix = np.zeros((vocab_len+2, vocab_len+2), dtype=float)
	for i, t1 in dictionary.items():
		if t1 not in docset1:
			continue

		for j, t2 in dictionary.items():
			if t2 not in docset2 or distance_matrix[i, j] != 0.0:
				continue

                # Compute Euclidean distance between word vectors.
			#print(t1,t2)
			distance_matrix[i, j] = distance_matrix[j, i] = distances[int(t1)][int(t2)]
	distance_matrix[vocab_len, vocab_len+1] = distance_matrix[vocab_len+1, vocab_len] = float(10.0) 

	if np.sum(distance_matrix) == 0.0:
	    # `emd` gets stuck if the distance matrix contains only zeros.
	    logger.info('The distance matrix is all zeros. Aborting (returning inf).')
	    return float('inf')

	def nbow(document):
	    d = np.zeros(vocab_len, dtype=float)
	    nbow = dictionary.doc2bow(document)  # Word frequencies.
	    doc_len = len(document)
	    for idx, freq in nbow:
	        d[idx] = freq / float(doc_len)  # Normalized word frequencies.
	    return d

    # Compute nBOW representation of documents.
	d1 = nbow(document1)
	d2 = nbow(document2)

	d1 = [a*scale1 for a in d1]
	d2 = [a*scale2 for a in d2]
	d1.append((scale2-1.0))
	d1.append(0)
	d2.append(0)
	d2.append((scale1-1.0))

	d1 = np.array(d1)
	d2 = np.array(d2)

        # Compute WMD.
	return [d1, d2, distance_matrix]

def ComputeSSM(input):
	d1 = input[0]
	d2 = input[1]
	distance_matrix = input[2]

	return emd(d1, d2, distance_matrix)


def SSM(document1, document2, rate):
	len_pre_oov1 = len(document1)
	len_pre_oov2 = len(document2)
	n = float(len_pre_oov1)*rate
	if n>float(len_pre_oov2):
		n = float(len_pre_oov2)

	diff1 = len_pre_oov1 - len(document1)
	diff2 = len_pre_oov2 - len(document2)
	if diff1 > 0 or diff2 > 0:
	    logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

	if not document1 or not document2:
	    logger.info(
	        "At least one of the documents had no words that were in the vocabulary. "
	        "Aborting (returning inf)."
	    )
	    return float('inf')

	dictionary = Dictionary(documents=[document1, document2])
	vocab_len = len(dictionary)

	if vocab_len == 1:
	    # Both documents are composed by a single unique token
	    return 0.0

	scale1 = float(len_pre_oov1)/n 
	scale2 = float(len_pre_oov2)/n 

        # Sets for faster look-up.
	docset1 = set(document1)
	docset2 = set(document2)

	distance_matrix = np.zeros((vocab_len+2, vocab_len+2), dtype=float)
	for i, t1 in dictionary.items():
		if t1 not in docset1:
			continue

		for j, t2 in dictionary.items():
			if t2 not in docset2 or distance_matrix[i, j] != 0.0:
				continue

                # Compute Euclidean distance between word vectors.
			#print(t1,t2)
			distance_matrix[i, j] = distance_matrix[j, i] = distances[int(t1)][int(t2)]
	distance_matrix[vocab_len, vocab_len+1] = distance_matrix[vocab_len+1, vocab_len] = float(10.0) 

	if np.sum(distance_matrix) == 0.0:
	    # `emd` gets stuck if the distance matrix contains only zeros.
	    logger.info('The distance matrix is all zeros. Aborting (returning inf).')
	    return float('inf')

	def nbow(document):
	    d = np.zeros(vocab_len, dtype=float)
	    nbow = dictionary.doc2bow(document)  # Word frequencies.
	    doc_len = len(document)
	    for idx, freq in nbow:
	        d[idx] = freq / float(doc_len)  # Normalized word frequencies.
	    return d

    # Compute nBOW representation of documents.
	d1 = nbow(document1)
	d2 = nbow(document2)

	d1 = [a*scale1 for a in d1]
	d2 = [a*scale2 for a in d2]
	d1.append((scale2-1.0))
	d1.append(0)
	d2.append(0)
	d2.append((scale1-1.0))

	d1 = np.array(d1)
	d2 = np.array(d2)

        # Compute WMD.
	return emd(d1, d2, distance_matrix)

def parallel_helper(i,te,setup):
	doc_len = len(docs)
	similarity = np.zeros(doc_len, dtype=float)
	for j, tr in enumerate(docs):
		if i == j:
			continue
		similarity[j] = ComputeSSM(setup[j])
	return [te[0], similarity]
	

#compute_words_distances()
def run():
		#Youtubewriter.writerow([te[0], similarity])

	#docs[0] = id, docs[1] = text
	with open("Youtube_words_distance.csv",'r') as rf1:
		Youtubereader1 = csv.reader(rf1)
		for dis in Youtubereader1:
			distances.append(dis)

	with open("Youtube_doc.csv", 'r') as rf2:
		Youtubereader2 = csv.reader(rf2)
		for input in Youtubereader2:
			if input[1] == "[]":
				continue
			docs.append(input)
	doc_len = len(docs)
	#with open("Youtube_nonempty.csv",'w') as wf1:
	#	Youtubewriter = csv.writer(wf1)
	#	for doc in docs:
	#		Youtubewriter.writerow(doc)

	for input in docs:
		input[1] = ast.literal_eval(input[1])
		input[1] = [str(i) for i in input[1]]


	#similarity = np.zeros((doc_len,doc_len), dtype=float)
	#pool = mp.Pool(mp.cpu_count()-2)

	#def collect_results(res):
	#	similarity[res[0]][res[1]] = res[2]

	with open("Youtube_similarity.csv",'w') as wf:
		Youtubewriter = csv.writer(wf)
		for i, te in enumerate(docs):
			if i < 1706:
				continue
			print(i)
			#setup = []
			#for j, tr in enumerate(docs):
			#	setup.append(SSM_setup(te[1],tr[1],0.6))
			similarity = np.zeros(doc_len, dtype=float)
			#Youtubewriter.writerow(pool.apply(parallel_helper, args=(i,te,setup)))
			#Youtubewriter.writerow(res)
			for j, tr in enumerate(docs):
				if i == j:
					continue
				#pool.apply_async(SSM, args=(i,j,te[1],tr[1],0.6), callback=collect_results)
				similarity[j] = SSM(te[1],tr[1],0.6)
			Youtubewriter.writerow(similarity)
				#similarity[i][j] = SSM(te[1],tr[1],0.6)

	#with open("Youtube_similarity.csv",'w') as wf:
	#	Youtubewriter = csv.writer(wf)
	#	for sim in similarity:
	#		Youtubewriter.writerow(sim)	

#run()
def sortnoutput():
	k = 500
	#ids = []
	names = []
	with open("Youtube_nonempty.csv",'r') as rf:
		Youtubereader = csv.reader(rf)
		for ins in Youtubereader:
			names.append(ins[0])

	wf = open("knn_results_6217_10108.csv",'w')
	Youtubewriter = csv.writer(wf)

	with open("Youtube_similarity_6217_10108.csv",'r') as rf:
		Youtubereader = csv.reader(rf)
		base = 6217
		count = 0
		for ins in Youtubereader:
			#print(ins)
			if not ins:
				continue
			print(ins[0])
			scores = []
			for i, sim in enumerate(ins):
				scores.append([i,float(sim)])
			sorted_scores = sorted(scores, key = itemgetter(1))
			knn = sorted_scores[:k]
			idd = names[base+count]
			row = []
			for ii in knn:
				row.append(names[ii[0]])
			Youtubewriter.writerow([idd,row])
			count += 1

	wf.close()

sortnoutput()





