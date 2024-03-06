import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
dataset=pd.read_csv('../input/websiteclassification/website_classification.csv')
dataset.shape
dataset.head()

# Create a new column 'category_id' with encoded categories 
df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates()
# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)
# New dataframe
df.head()
#Fig- Appended category id to the various categories.
category_id_df
#Fig-Result
WordCloud
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize=(40,25))
subset = df[df['Category']=='Travel']
text = subset.cleaned_website_text.values
cloud1=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,1)
plt.axis('off')
plt.title("Travel",fontsize=40)
plt.imshow(cloud1)
subset = df[df['Category']=='Social Networking and Messaging']
text = subset.cleaned_website_text.values
cloud2=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,2)
plt.axis('off')
plt.title("Social Networking and Messaging",fontsize=40)
plt.imshow(cloud2)
subset = df[df['Category']=='News']
text = subset.cleaned_website_text.values
cloud3=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,3)
plt.axis('off')
plt.title("News",fontsize=40)
plt.imshow(cloud3)
subset = df[df['Category']=='Streaming Services']
text = subset.cleaned_website_text.values
cloud4=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,4)
plt.axis('off')
plt.title("Streaming Services",fontsize=40)
plt.imshow(cloud4)
subset = df[df['Category']=='Sports']
text = subset.cleaned_website_text.values
cloud5=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,5)
plt.axis('off')
plt.title('Sports',fontsize=40)
plt.imshow(cloud5)
subset = df[df['Category']=='Photography']
text = subset.cleaned_website_text.values
cloud6=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,6)
plt.axis('off')
plt.title("Photography",fontsize=40)
plt.imshow(cloud6)
subset = df[df['Category']=='Law and Government']
text = subset.cleaned_website_text.values
cloud7=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,7)
plt.axis('off')
plt.title("Law and Government",fontsize=40)
plt.imshow(cloud7)
subset = df[df['Category']=='Health and Fitness']
text = subset.cleaned_website_text.values
cloud8=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,8)
plt.axis('off')
plt.title("Health and Fitness",fontsize=40)
plt.imshow(cloud8)
subset = df[df['Category']=='Games']
text = subset.cleaned_website_text.values
cloud9=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,9)
plt.axis('off')
plt.title("Games",fontsize=40)
plt.imshow(cloud9)
subset = df[df['Category']=='E-Commerce']
text = subset.cleaned_website_text.values
cloud10=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,10)
plt.axis('off')
plt.title("E-Commerce",fontsize=40)
plt.imshow(cloud10)
subset = df[df['Category']=='Forums']
text = subset.cleaned_website_text.values
cloud11=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,11)
plt.axis('off')
plt.title("Forums",fontsize=40)
plt.imshow(cloud11)
subset = df[df['Category']=='Food']
text = subset.cleaned_website_text.values
cloud12=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,12)
plt.axis('off')
plt.title("Food",fontsize=40)
plt.imshow(cloud12)
subset = df[df['Category']=='Education']
text = subset.cleaned_website_text.values
cloud13=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,13)
plt.axis('off')
plt.title("Education",fontsize=40)
plt.imshow(cloud13)
subset =df[df['Category']=='Computers and Technology']
text = subset.cleaned_website_text.values
cloud14=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,14)
plt.axis('off')
plt.title("Computers and Technology",fontsize=40)
plt.imshow(cloud14)
subset = df[df['Category']=='Business/Corporate']
text = subset.cleaned_website_text.values
cloud15=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,15)
plt.axis('off')
plt.title("Business/Corporate",fontsize=40)
plt.imshow(cloud15)
subset = df[df['Category']=='Adult']
text = subset.cleaned_website_text.values
cloud16=WordCloud(stopwords=STOPWORDS,background_color='pink',colormap="Dark2",collocations=False,width=2500,height=1800).generate(" ".join(text))
plt.subplot(4,4,16)
plt.axis('off')
plt.title("Adult",fontsize=40)
plt.imshow(cloud16)
plt.show()
#FigResultant 
WordCloud
#Finding the most 3 corelated terms with each category.
# Finding the three most correlated terms with each of the categories
N = 3
for Category, category_id in sorted(category_to_id.items()):
 features_chi2 = chi2(features, labels == category_id)
 indices = np.argsort(features_chi2[0])
 feature_names = np.array(tfidf.get_feature_names())[indices]
 unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
 bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
 print("\n==> %s:" %(Category))
 print(" * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
 print(" * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))
##pliting the data into train and test sets The original data was divided into features (X) and 
#target (y), which were then splitted into train (75%) and test (25%) sets. Thus, the algorithms 
#would be trained on one set of data and tested out on a completely different set of data (not 
#seen before by the algorithm).
X = df['cleaned_website_text'] # Collection of text
y = df['Category'] # Target or the labels we want to predict
X_train, X_test, y_train, y_test = train_test_split(X, y, 
 test_size=0.25,
 random_state = 0)
y_train.value_counts()
y_test.value_counts()
#Fig- showing the data generated from the dataset.
#Plotting the confusion matrix
conf_mat = confusion_matrix(y_test, predicted,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="OrRd", fmt='d',xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16)
#Fig- Confusion matrix (Actual vs Predicted)
from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import pandas as pd
import os
class ScrapTool:
 def visit_url(self, website_url):
 #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) 
#AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
    content = requests.get(website_url,timeout=60).content
 
 #lxml is apparently faster than other settings.
    soup = BeautifulSoup(content, "lxml")
    result = {
    "website_url": website_url,
    "website_name": self.get_website_name(website_url),
    "website_text": self.get_html_title_tag(soup)+self.get_html_meta_tags(soup)+self.get_html_heading_tags(soup)+self.get_text_content(soup)}
 
 #Convert to Series object and return
    return pd.Series(result)
 
 def get_website_name(self,website_url):
    
    return "".join(urlparse(website_url).netloc.split(".")[-2])
 
 def get_html_title_tag(self,soup):
    return '. '.join(soup.title.contents)
 
 def get_html_meta_tags(self,soup):
 
    tags = soup.find_all(lambda tag: (tag.name=="meta") & (tag.has_attr('name') & (tag.has_attr('content'))))
    content = [str(tag["content"]) for tag in tags if tag["name"] in ['keywords','description']]
    return ' '.join(content)
 
 def get_html_heading_tags(self,soup):
 
    tags = soup.find_all(["h1","h2","h3","h4","h5","h6"])
    content = [" ".join(tag.stripped_strings) for tag in tags]
    return ' '.join(content)
 
 def get_text_content(self,soup):
 
    tags_to_ignore = ['style', 'script', 'head', 'title', 'meta', '[document]',"h1","h2","h3","h4","h5","h6","noscript"]
    tags = soup.find_all(text=True)
    result = []
    for tag in tags:
        stripped_tag = tag.strip()
        if tag.parent.name not in tags_to_ignore and isinstance(tag, bs4.element.Comment)==False and not stripped_tag.isnumeric() and len(stripped_tag)>0:
            result.append(stripped_tag)
    return ' '.join(result)
import spacy as sp
from collections import Counter
sp.prefer_gpu()
import en_core_web_sm
#anconda prompt ko run as adminstrator and copy paste this:python -m spacy 
#download en
nlp = en_core_web_sm.load()
import re
def clean_text(doc):
 '''
 Clean the document. Remove pronouns, stopwords, lemmatize the words and 
lowercase them
 '''
 doc = nlp(doc)
 tokens = []
 exclusion_list = ["nan"]
 for token in doc:
    if token.is_stop or token.is_punct or token.text.isnumeric() or (token.text.isalnum()==False) or token.text in exclusion_list :
        continue
 token = str(token.lemma_.lower().strip())
 tokens.append(token)
 return " ".join(tokens) 
#Here we can change any website link and it predicts the category
website='https://leetcode.com/problemset/all/'
scrapTool = ScrapTool()
try:
 web=dict(scrapTool.visit_url(website))
 text=(clean_text(web['website_text']))
 t=fitted_vectorizer.transform([text])
 print(id_to_category[m1.predict(t)[0]])
 data=pd.DataFrame(m1.predict_proba(t)*100,columns=df['Category'].unique())
 data=data.T
 data.columns=['Probability']
 data.index.name='Category'
 a=data.sort_values(['Probability'],ascending=False)
 a['Probability']=a['Probability'].apply(lambda x:round(x,2))
except:
 print("Connection Timedout!")