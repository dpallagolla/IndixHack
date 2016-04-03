import pickle
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
def build_data_frame(path):
	rows = []
	index = []

	data_dict = pickle.load(open(path,'r'))
	for i,record in data_dict.iteritems():
		rows.append({'text': record["title"].strip(), 'class': record["brand_id"].strip()})
		index.append(i)
		
	data_frame = DataFrame(rows, index=index)
	return data_frame

import pickle
from sklearn.externals import joblib
from pandas import DataFrame
import numpy

def build_test_data_frame(path):
        rows = []
        index = []

        data_list = pickle.load(open(path,'r'))
        for i,record in enumerate(data_list):
                rows.append({'text': record["title"].strip()})
                index.append(i)

        data_frame = DataFrame(rows, index=index)
        return data_frame

test_data_frame = build_test_data_frame('test_list.pkl')
    
# def append_data_frame(path,data_frame,classification,list_or_dict):
#     if list_or_dict == 'dict':
#         data_frame2 = build_data_frame(path,classification,'dict',len(data_frame.index))
#     else:
#         i = len(data_frame.index)
#         rows = []
#         index = []
#         data_list = pickle.load(open(path,'r'))
#         for email in data_list:
#             rows.append({'text': email, 'class': classification})
#             index.append(i)
#             i = i + 1
        
#         data_frame2 = DataFrame(rows, index=index)
        
#     return pd.concat([data_frame,data_frame2])

data = build_data_frame('train_4.5x_large_sample.pkl')
data = data.reindex(numpy.random.permutation(data.index))

# fbk_data_frame = data.loc[data['class'] == 'fbk']
# # train_fbk_indices = [i for i in xrange(0,int(0.8*len(fbk_data_frame.index)))]
# # train_fbk_data_frame = fbk_data_frame.iloc[train_fbk_indices]
# # test_fbk_indices = [i for i in xrange(int(0.8*len(fbk_data_frame.index)),len(fbk_data_frame.index))]
# # test_fbk_data_frame = fbk_data_frame.iloc[test_fbk_indices]
# # print len(fbk_data_frame.index), len(train_fbk_data_frame.index), len(test_fbk_data_frame.index)

# non_fbk_data_frame = data.loc[data['class'] == 'non-fbk']
# # train_non_fbk_indices = [i for i in xrange(0,int(0.8*len(non_fbk_data_frame.index)))]
# # train_non_fbk_data_frame = non_fbk_data_frame.iloc[train_non_fbk_indices]
# # test_non_fbk_indices = [i for i in xrange(int(0.8*len(non_fbk_data_frame.index)),len(non_fbk_data_frame.index))]
# # test_non_fbk_data_frame = non_fbk_data_frame.iloc[test_non_fbk_indices]
# # print len(non_fbk_data_frame.index),len(train_non_fbk_data_frame.index), len(test_non_fbk_data_frame.index)


# train_data_frame = pd.concat([fbk_data_frame,non_fbk_data_frame])
# # train_data_frame = pd.concat([train_fbk_data_frame,train_non_fbk_data_frame])
# train_data_frame = train_data_frame.reindex(numpy.random.permutation(train_data_frame.index))
train_data_frame = data
# test_data_frame = pd.concat([test_fbk_data_frame,test_non_fbk_data_frame])
# test_data_frame = test_data_frame.reindex(numpy.random.permutation(test_data_frame.index))

# print len(train_data_frame.index), len(test_data_frame)


############################# SGD Classifier #####################################
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2),stop_words ='english')),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',         OneVsRestClassifier(LinearSVC(random_state=0)))
])
# neg - non fbk classifier
# Training ---
# ('Total emails classified:', 8745)
# ('Score:', 0.91425768698293586)
# Confusion matrix:
# [[2276  266]
#  [ 493 5710]]
# Hold Out ---
# ('Total emails held back:', 2187)
# ('Score:', 0.90327330189413457)
# Confusion matrix:
# [[ 545   91]
#  [ 122 1429]]

# from sklearn.neural_network import MLPClassifier

# pipeline = Pipeline([
#     ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2))),
#     ('tfidf_transformer',  TfidfTransformer()),
#     ('classifier',         MLPClassifier())
# ])


from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

k_fold = KFold(n=len(train_data_frame), n_folds=6)
scores = []
i = 0
#confusion = numpy.array([[0, 0], [0, 0]])
confusion = numpy.zeros(shape=(1062,1062))
for train_indices, test_indices in k_fold:
	train_text = train_data_frame.iloc[train_indices]['text'].values
	train_y = train_data_frame.iloc[train_indices]['class'].values

	test_text = train_data_frame.iloc[test_indices]['text'].values
	test_y = train_data_frame.iloc[test_indices]['class'].values

	pipeline.fit(train_text, train_y)
#	predictions = pipeline.predict(test_text)

#	print 'fold ' + str(i)
#	i += 1
#	print confusion_matrix(test_y, predictions)
#	print f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')
	
	test_indices = [x for x in xrange(0,619240)]

	test_text = test_data_frame.iloc[test_indices]['text'].values
	
	predictions_test = pipeline.predict(test_text)

	with open("submissions/submission_8.txt","w") as op:
        	for pred in predictions_test:
                	op.write(str(pred)+"\n")
	exit()
#	confusion += confusion_matrix(test_y, predictions)
#	score = f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')
#	scores.append(score)
#	joblib.dump(pipeline, 'models/classifier_mlp_model_fold_' + str(i) + '.pkl', compress=9)

print 'Training ---'
print('Total products classified:', len(train_data_frame))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)



#from sklearn.externals import joblib
#joblib.dump(pipeline, 'models/classifier_sgd_model.pkl', compress=9)


# # Then to load it back in 
# print 'Hold Out ---'
# trained_pipeline = joblib.load('pos_fbk_neg_fbk_classifier_sgd_model.pkl')

# test_text = test_data_frame['text'].values
# test_y = test_data_frame['class'].values

# predictions = trained_pipeline.predict(test_text)

# confusion = confusion_matrix(test_y, predictions)
# score = f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')

# print('Total emails held back:', len(test_data_frame))
# print('Score:', score)
# print('Confusion matrix:')
# print(confusion)

