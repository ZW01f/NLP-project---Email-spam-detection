## imports 


```python
import pandas as pd
import nltk 
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tabulate import tabulate 
import warnings as wr
wr.filterwarnings("ignore")
#---------#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
#--------#
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
from matplotlib.colors import ListedColormap
#from nltk import FreqDist
```


```python
from nltk import FreqDist
```

## downloads


```python
# punkt = nltk.data.load('./tokenizers/punkt/english.pickle')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.data.load(r'spam.csv',format='text')
```


```python
def Cleaning(copy_data):
    mapping={'ham':0,'spam':1}
    copy_data['v1']=copy_data['v1'].replace(mapping)

    copy_data=copy_data.fillna(' ')
    copy_data['concat']=copy_data['v2']+copy_data['Unnamed: 2']+copy_data['Unnamed: 3']+copy_data['Unnamed: 4']

    copy_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4','v2'],axis=1,inplace=True)
    copy_data.rename(columns={'v1':'classifier','concat':'texts'},inplace=True) 

    copy_data['texts'] = copy_data['texts'].str.lower()

    return copy_data
```

# **Task 2**


```python
def lemmaOrStem(tokens,stemm=False,stemmer='',lemma=False):

    if (stemm):
        #---------porterstremmer------#  
        if(stemmer=='Porter'):
            porter = PorterStemmer()
            porterStemming = [porter.stem(j) for j in tokens]
            return porterStemming

        #---------landstremmer------#    
        elif(stemmer=='lancaster'):    
            lancaster=LancasterStemmer()
            lancasterStemming = [lancaster.stem(j) for j in tokens]
            return lancasterStemming
            
    #---------------SnowballStemmer--------#

        elif(stemmer=='snow') :
                sm = SnowballStemmer("english")
                snowballStemming = [sm.stem(j) for j in tokens]
                return snowballStemming
   
    if(lemma):        
    #----------Lemmatization---------#
        lemm = nltk.WordNetLemmatizer()
        #ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
        lemm_1 = [lemm.lemmatize(j,pos='v') for j in tokens]
        return lemm_1
```


```python
stop_list= list(stopwords.words("english"))
```


```python
def preprocessing(x):
    for i in range(0,x.shape[0]):
        # print(i)
        tokens = word_tokenize(x['texts'][i])
        # #----------------------------------#
        sentences=nltk.sent_tokenize(x['texts'][i])
        # # print(sentences)
        x['sentences'][i] =len(sentences)
        #-----------------------------------#
        lemm_stem = lemmaOrStem(tokens,stemm=False,stemmer='',lemma=True)
        x['pureText'][i] = " ".join(lemm_stem)
        #----------------------------------#
        # print(lemm_stem)
        tokensOfpureTextWithoutstop=[token for token in lemm_stem if token not in stop_list]
        # print(tokensOfpureTextWithoutstop)
        x['pureText'][i] = " ".join(tokensOfpureTextWithoutstop)
        
        #----------------------------------#

        textWithoutPunctuations= re.findall(r'\w+',x['pureText'][i])
        # textWithoutPunctuations = re.sub(r'([-[_\],!?():{}&$#@%*+;"\t\n\b])', r'', x['pureText'][i])
        x['pureText'][i] = " ".join(textWithoutPunctuations)
        # print(textWithoutPunctuations)
        # if (i==5):
            # break
    return x
```

## Data Gram


```python
def extractGram(data,num):
  for i in range(0,len(data)):
    data = " ".join(data)
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [" ".join(grams) for grams in n_grams]  

```

## Features extraction


```python
def features_extraction(extraction_alg,x):
    if extraction_alg == 'binary_encoding':
        vector=CountVectorizer(binary=True)
        # to extract vocabs and put them in dictionary 
        vector.fit(x['pureText'])
        return pd.DataFrame(vector.transform(x['pureText']).toarray(),columns=sorted(vector.vocabulary_.keys()))
    elif extraction_alg == 'counting':
        vector=CountVectorizer(binary=False)
        vector.fit(x['pureText'])
        return pd.DataFrame(vector.transform(x['pureText']).toarray(),columns=sorted(vector.vocabulary_.keys()))
    elif extraction_alg == 'tf_idf':
        vec = TfidfVectorizer()
        vec.fit(x['pureText'])
        return pd.DataFrame(vec.transform(x['pureText']).toarray(), columns=sorted(vec.vocabulary_.keys()))
```

# *Run* 


```python
data=pd.read_csv(r'spam.csv',encoding="ISO-8859-1")
data_copy=data.copy()
clean_copy = Cleaning(data_copy)
clean_copy['sentences'] = 0
clean_copy['pureText'] = clean_copy['texts']
```

### Preprocessing 


```python
data_after = preprocessing(clean_copy)
data_after
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>classifier</th>
      <th>texts</th>
      <th>sentences</th>
      <th>pureText</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>go until jurong point, crazy.. available only ...</td>
      <td>1</td>
      <td>go jurong point crazy available bugis n great ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>ok lar... joking wif u oni...</td>
      <td>1</td>
      <td>ok lar joke wif u oni</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>free entry in 2 a wkly comp to win fa cup fina...</td>
      <td>1</td>
      <td>free entry 2 wkly comp win fa cup final tkts 2...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>u dun say so early hor... u c already then say...</td>
      <td>1</td>
      <td>u dun say early hor u c already say</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>nah i don't think he goes to usf, he lives aro...</td>
      <td>1</td>
      <td>nah n t think go usf live around though</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>1</td>
      <td>this is the 2nd time we have tried 2 contact u...</td>
      <td>4</td>
      <td>2nd time try 2 contact u u win å 750 pound pri...</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>0</td>
      <td>will ì_ b going to esplanade fr home?</td>
      <td>1</td>
      <td>ì_ b go esplanade fr home</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>0</td>
      <td>pity, * was in mood for that. so...any other s...</td>
      <td>2</td>
      <td>pity mood suggestions</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>0</td>
      <td>the guy did some bitching but i acted like i'd...</td>
      <td>1</td>
      <td>guy bitch act like d interest buy something el...</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>0</td>
      <td>rofl. its true to its name</td>
      <td>2</td>
      <td>rofl true name</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 4 columns</p>
</div>



### stemming  (optional)


```python
x = data_after['pureText'][0]
tokens = word_tokenize(x)
tokens[0:5]
```




    ['go', 'jurong', 'point', 'crazy', 'available']




```python
lemmaOrStem(tokens,stemm=True,stemmer='Porter',lemma= False)[0:5]
```




    ['go', 'jurong', 'point', 'crazi', 'avail']




```python
lemmaOrStem(tokens,stemm=True,stemmer='lancaster',lemma=False)[0:5]
```




    ['go', 'jurong', 'point', 'crazy', 'avail']




```python
lemmaOrStem(tokens,stemm=True,stemmer='snow',lemma=False)[0:5]
```




    ['go', 'jurong', 'point', 'crazi', 'avail']



### n-gram


```python
extractGram(data_after['pureText'],2)[0:5]
```




    ['go jurong',
     'jurong point',
     'point crazy',
     'crazy available',
     'available bugis']




```python
extractGram(data_after['pureText'],3)[0:5]
```




    ['go jurong point',
     'jurong point crazy',
     'point crazy available',
     'crazy available bugis',
     'available bugis n']



# Features extraction

## Binary encoding 


```python
Encoding = features_extraction('binary_encoding',data_after)
```

##  counting [optional]


```python
features_extraction('binary_encoding',data_after[0:9][5:15])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>09061701461</th>
      <th>12</th>
      <th>50</th>
      <th>900</th>
      <th>aid</th>
      <th>back</th>
      <th>brother</th>
      <th>call</th>
      <th>callers</th>
      <th>callertune</th>
      <th>...</th>
      <th>still</th>
      <th>tb</th>
      <th>treat</th>
      <th>valid</th>
      <th>value</th>
      <th>vettam</th>
      <th>week</th>
      <th>winner</th>
      <th>word</th>
      <th>xxx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 53 columns</p>
</div>



### tf_idf [optional]


```python
features_extraction('tf_idf',data_after)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00</th>
      <th>000</th>
      <th>000pes</th>
      <th>008704050406</th>
      <th>0089</th>
      <th>0121</th>
      <th>01223585236</th>
      <th>01223585334</th>
      <th>0125698789</th>
      <th>02</th>
      <th>...</th>
      <th>ó_</th>
      <th>û_</th>
      <th>û_thanks</th>
      <th>ûªm</th>
      <th>ûªt</th>
      <th>ûªve</th>
      <th>ûï</th>
      <th>ûïharry</th>
      <th>ûò</th>
      <th>ûówell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 7787 columns</p>
</div>



# ML algorithms:) 


```python
#print(sum(Encoding.iloc[0,:]))
#print ((data_after['pureText'][0].split()))
#print(len(data_after['pureText'][0].split())) # note encoding removes single char
```


```python
X_train,X_test,y_train,y_test = train_test_split(Encoding,data_after.classifier, test_size=0.15, random_state=42)
```


```python
classifiers=[SVC(),LogisticRegression(),MultinomialNB()]
train,test=[],[]
for cls in classifiers:
    cls.fit(X_train,y_train)
    train.append(cls.score(X_train,y_train))
    test.append(cls.score(X_test,y_test))

```


```python
data={'train':train ,
      'test':test,
      }

Results = pd.DataFrame(data, index =["SVC","LogisticRegression","MultinomialNB"])
```


```python
precision =[]
recall =[]
f1_score = []
trainset_accuracy = []
testset_accuracy = []

for i in classifiers:
    pred_train = i.predict(X_train)
    pred_test = i.predict(X_test)
    prec = metrics.precision_score(y_test, pred_test)
    recal = metrics.recall_score(y_test, pred_test)
    f1_s = metrics.f1_score(y_test, pred_test)
    train_accuracy = i.score(X_train,y_train)
    test_accuracy = i.score(X_test,y_test)
  
    #Appending scores
    precision.append(prec)
    recall.append(recal)
    f1_score.append(f1_s)
    trainset_accuracy.append(train_accuracy)
    testset_accuracy.append(test_accuracy)
```


```python
data = {'Precision':precision,
'Recall':recall,
'F1score':f1_score,
'Accuracy on Testset':testset_accuracy,
'Accuracy on Trainset':trainset_accuracy}
# Creates pandas DataFrame.
Results = pd.DataFrame(data, index =["SVC","LogisticRegression","MultinomialNB"])
cmap2 = ListedColormap(["#E2CCFF","#E598D8"])
Results.style.background_gradient(cmap=cmap2)
```




<style  type="text/css" >
#T_d2e05_row0_col0,#T_d2e05_row0_col2,#T_d2e05_row0_col3,#T_d2e05_row0_col4,#T_d2e05_row1_col0,#T_d2e05_row1_col2,#T_d2e05_row1_col3,#T_d2e05_row1_col4,#T_d2e05_row2_col1{
            background-color:  #e598d8;
            color:  #000000;
        }#T_d2e05_row0_col1,#T_d2e05_row1_col1,#T_d2e05_row2_col0,#T_d2e05_row2_col2,#T_d2e05_row2_col3,#T_d2e05_row2_col4{
            background-color:  #e2ccff;
            color:  #000000;
        }</style><table id="T_d2e05_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Precision</th>        <th class="col_heading level0 col1" >Recall</th>        <th class="col_heading level0 col2" >F1score</th>        <th class="col_heading level0 col3" >Accuracy on Testset</th>        <th class="col_heading level0 col4" >Accuracy on Trainset</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_d2e05_level0_row0" class="row_heading level0 row0" >SVC</th>
                        <td id="T_d2e05_row0_col0" class="data row0 col0" >1.000000</td>
                        <td id="T_d2e05_row0_col1" class="data row0 col1" >0.861111</td>
                        <td id="T_d2e05_row0_col2" class="data row0 col2" >0.925373</td>
                        <td id="T_d2e05_row0_col3" class="data row0 col3" >0.982057</td>
                        <td id="T_d2e05_row0_col4" class="data row0 col4" >0.996622</td>
            </tr>
            <tr>
                        <th id="T_d2e05_level0_row1" class="row_heading level0 row1" >LogisticRegression</th>
                        <td id="T_d2e05_row1_col0" class="data row1 col0" >1.000000</td>
                        <td id="T_d2e05_row1_col1" class="data row1 col1" >0.870370</td>
                        <td id="T_d2e05_row1_col2" class="data row1 col2" >0.930693</td>
                        <td id="T_d2e05_row1_col3" class="data row1 col3" >0.983254</td>
                        <td id="T_d2e05_row1_col4" class="data row1 col4" >0.996410</td>
            </tr>
            <tr>
                        <th id="T_d2e05_level0_row2" class="row_heading level0 row2" >MultinomialNB</th>
                        <td id="T_d2e05_row2_col0" class="data row2 col0" >0.870690</td>
                        <td id="T_d2e05_row2_col1" class="data row2 col1" >0.935185</td>
                        <td id="T_d2e05_row2_col2" class="data row2 col2" >0.901786</td>
                        <td id="T_d2e05_row2_col3" class="data row2 col3" >0.973684</td>
                        <td id="T_d2e05_row2_col4" class="data row2 col4" >0.992821</td>
            </tr>
    </tbody></table>




```python
import matplotlib.pyplot as plt


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

for cls, ax in zip(classifiers, axes.flatten()):
    plot_confusion_matrix(cls, 
                          X_test, 
                          y_test, 
                          ax=ax   
                          )
    ax.title.set_text(type(cls).__name__)
plt.tight_layout()  
plt.show()
```


    
![png](output_40_0.png)
    



```python

```
