import pickle
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pickle
from sklearn.externals import joblib
from pandas import DataFrame
import numpy

def build_train_data_frame(path):
	rows = []
	index = []

	data_dict = pickle.load(open(path,'r'))
	for i,record in data_dict.iteritems():
		rows.append({'title': record["title"].strip(), 'class': record["brand_id"].strip()})
		index.append(i)
		
	data_frame = DataFrame(rows, index=index)
	return data_frame

def build_test_data_frame(path):
        rows = []
        index = []

        data_list = pickle.load(open(path,'r'))
        for i,record in enumerate(data_list):
                rows.append({'title': record["title"].strip()})
                index.append(i)

        data_frame = DataFrame(rows, index=index)
        return data_frame


class ItemSelector(BaseEstimator, TransformerMixin):
   
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
    
class TextCategoryExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, records):
        features = np.recarray(shape=(len(records),),
                               dtype=[('title', object), ('cat_id', object)])

		#data_list = pickle.load(open(path,'r'))
		for i,record in enumerate(records):
			features['title'][i] = record["title"].strip()
			features['cat'][i] = record["cat_id"].strip()

        return features

############################# SGD Classifier #####################################
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

pipeline = Pipeline([
    # Extract the title and category
    ('title_cat', TextCategoryExtractor()),

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ('title', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2),stop_words ='english')),
                ('tfidf_transformer',  TfidfTransformer()),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('cat_id', Pipeline([
                ('selector', ItemSelector(key='cat_id')),
                ('vect', DictVectorizer())
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'title': 0.8,
            'cat_id': 1.0
        },
    )),

    # Use a SVC classifier on the combined features
   ('classifier', OneVsRestClassifier(LinearSVC(random_state=0)))
])

train_data_list = pickle.load(open("train_xx_large_sample_list.pkl"))

train_data_frame = build_train_data_frame('train_xx_large_sample.pkl')
train_data_frame = train_data_frame.reindex(numpy.random.permutation(data.index))

pipeline.fit(train_data_list, train.target)



from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

k_fold = KFold(n=len(train_data_frame), n_folds=6)
scores = []
i = 0
#confusion = numpy.array([[0, 0], [0, 0]])
confusion = numpy.zeros(shape=(1062,1062))
for train_indices, test_indices in k_fold:
	train_data = train_data_list.iloc[train_indices]['text'].values
	train_y = train_data_frame.iloc[train_indices]['class'].values

	test_text = train_data_frame.iloc[test_indices]['text'].values
	test_y = train_data_frame.iloc[test_indices]['class'].values
	# pipeline.fit(train.data, train.target)
	pipeline.fit(train_text, train_y)
#	predictions = pipeline.predict(test_text)

test_data_frame = build_test_data_frame('test_list.pkl')

# #	print 'fold ' + str(i)
# #	i += 1
# #	print confusion_matrix(test_y, predictions)
# #	print f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')
	
# 	test_indices = [x for x in xrange(0,619240)]

# 	test_text = test_data_frame.iloc[test_indices]['text'].values
	
# 	predictions_test = pipeline.predict(test_text)

# 	with open("submissions/submission_4.txt","w") as op:
#         	for pred in predictions_test:
#                 	op.write(str(pred)+"\n")
# 	exit()
# #	confusion += confusion_matrix(test_y, predictions)
# #	score = f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')
# #	scores.append(score)
# #	joblib.dump(pipeline, 'models/classifier_mlp_model_fold_' + str(i) + '.pkl', compress=9)

# print 'Training ---'
# print('Total products classified:', len(train_data_frame))
# print('Score:', sum(scores)/len(scores))
# print('Confusion matrix:')
# print(confusion)



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

