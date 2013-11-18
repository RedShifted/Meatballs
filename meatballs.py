# meatballs by Sam Bonin
# Thanks to Tim Dettmers for the head start:
# http://www.kaggle.com/c/crowdflower-weather-twitter/forums/t/6046/how-to-get-started-in-python-sklearn

print ''

import pandas
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="pandas", lineno=570)
# regarding ^ see link below for details
# http://stackoverflow.com/questions/19554900/scikitlearn-breaks-pandas-installation

paths = ['C:\Users\sdbon_000\Desktop\Python Programs\meatballs\mtrain.csv',
'C:\Users\sdbon_000\Desktop\Python Programs\meatballs\mtest.csv']

traindata = pandas.read_csv(paths[0])
testdata = pandas.read_csv(paths[1])
print traindata
print ''

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
tfidf.fit(traindata['tweet'])
X = tfidf.transform(traindata['tweet'])
test = tfidf.transform(testdata['tweet'])
y = numpy.array(traindata.ix[:,4:])

print "X.shape = ",X.shape
print "y.shape =",y.shape
print ''