import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import re, string, unicodedata
from string import punctuation
from termcolor import colored
from collections import Counter


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


import keras
import tensorflow as tf
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer

import nltk
from nltk.corpus import stopwords
from textblob import Word
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet


from warnings import filterwarnings
filterwarnings('ignore')

from sklearn import set_config
set_config(print_changed_only = False)

print(colored("\nLIBRARIES WERE SUCCESFULLY IMPORTED...", "green"))

true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

print(colored("\nDATASETS WERE SUCCESFULLY LOADED...", "green"))

true_news["news_class"], fake_news["news_class"] = 1, 0

news = pd.concat([true_news, fake_news])

print(colored("\nDATASETS WERE SUCCESFULLY MERGED...", "green"))

#get basic information about dataset

news.info(memory_usage = True, verbose = True)

#check whether there are duplicated values

news.duplicated().sum()

# drop duplicated values from the dataset

news.drop_duplicates(inplace = True)

print(colored("\nDUPLICATED VALUES WERE SUCCESFULLY DROPPED...", "green"))

#look class frequencies of 'news_class' variable

grouped_n = news.groupby("news_class").count()
grouped_n["title"]

#check whether there are 'nan' values

news.isnull().sum()

#look class frequencies of 'subject' variable

news["subject"].value_counts()

#the number of the texts

news["title"].count()



plt.figure(figsize = [8, 7], clear = True, facecolor = 'white')

sns.barplot(x = news["news_class"].value_counts().index,
            y = news["news_class"].value_counts(),
            saturation = 1).set(title = "Class frequencies of the dataset (true - 1, fake - 0)");


plt.figure(figsize = [15, 9], clear = True, facecolor = 'white')
sns.barplot(x = news["subject"].value_counts().index,
            y = news["subject"].value_counts(),
            saturation = 1).set(title = "Class frequencies of the dataset (true - 1, fake - 0)");


fig = px.pie(data_frame = news, names = "news_class", hole = 0.4, title = "counts in news_class",
             width = 1000, height = 500, color_discrete_sequence = px.colors.sequential.Sunset_r)

fig.update_traces(textposition = "inside", textinfo = "percent+label",
                  marker = dict(line = dict(width = 1.2, color = "#000000")))

fig.update_layout(title_x = 0.5, title_font = dict(size = 30), uniformtext_minsize = 25)

fig.show()


fig = px.pie(news, names = "subject", title = "counts in news_class", hole = 0.5,
            width = 1000, height = 500, color_discrete_sequence = px.colors.sequential.Sunset_r)

fig.update_traces(textposition = "inside", textinfo = "percent+label",
                  marker = dict(line = dict(width = 1.2, color = "#000000")))

fig.update_layout(title_x = 0.5, title_font = dict(size = 30), uniformtext_minsize = 25)

fig.show()




pd.crosstab(news["news_class"], news["subject"],
            normalize = True).plot(kind = "bar", 
                                   backend = "matplotlib",
                                   legend = True, table = True, stacked = True);



#Wordcloud for true news

text = " ".join(i for i in true_news.text)

wc = WordCloud(background_color = "white", width = 1200, height = 600,
               contour_width = 0, contour_color = "red", max_words = 1000,
               scale = 1, collocations = False, repeat = True, min_font_size = 1)

wc.generate(text)

plt.figure(figsize = [15, 7])
plt.imshow(wc)
plt.axis("off")
plt.show


#Wordcloud for fake news

text = " ".join(i for i in fake_news.text)

wc = WordCloud(background_color = "white", width = 1200, height = 600,
               contour_width = 0, contour_color = "red", max_words = 1000,
               scale = 1, collocations = False, repeat = True, min_font_size = 1)

wc.generate(text)

plt.figure(figsize = [15, 7])
plt.imshow(wc)
plt.axis("off")
plt.show


news["text"] = news["text"] + " " + news["title"]
news.drop(["title", "date", "subject"], axis = 1, inplace = True)

print(colored("\n'TITLE','DATE' AND 'SUBJECT' COLUMNS WERE SUCCESFULLY DROPPED...", "green"))



#convert uppercase letters to lowercase letters

news["text"] = news["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

print(colored("\nCONVERTED SUCCESFULLY...", "green"))


#delete punctuation marks

news["text"] = news["text"].str.replace('[^\w\s]','')

print(colored("\nDELETED PUNCTUATION MARKS SUCCESFULLY...", "green"))


#delete numbers

news["text"] = news["text"].str.replace('\d','')

print(colored("\n NUMBERS DELETED SUCCESFULLY...", "green"))



#delete stopwords

stop_words = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stop_words.update(punctuation)

news["text"] = news["text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

print(colored("\nSTOPWORDS DELETED SUCCESFULLY...", "green"))




#lemmatization. That is, we get the roots of the words

news["text"] = news["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

print(colored("\nLEMMATIZED SUCCESFULLY...", "green"))




#remove URLs

news["text"] = news["text"].apply(lambda x: " ".join(re.sub(r'http\S+', '', x) for x in x.split()))

print(colored("\nURLs WERE SUCCESFULLY REMOVED...", "green"))




#look at the latest condition of the dataset

news.head(n = 10).style.background_gradient(cmap = "summer")



#get every words from dataset and append them to 'corpus' list
corpus = []
for i in news.text:
    for j in i.split():
        corpus.append(j.strip())

#count the words
counter = Counter(corpus)
common_words = counter.most_common(15)
dict(common_words)




#Wordcloud for whole dataset

text = " ".join(i for i in news.text)

wc = WordCloud(background_color = "white", width = 1200, height = 600,
               contour_width = 0, contour_color = "red", max_words = 1000,
               scale = 1, collocations = False, repeat = True, min_font_size = 1)

wc.generate(text)

plt.figure(figsize = [15, 7])
plt.imshow(wc)
plt.axis("off")
plt.show



#average word length in true news

fig,ax = plt.subplots(figsize = (15, 8))
text_words = news[news["news_class"] == 1]["text"].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(text_words.map(lambda x: np.mean(x)), color = "#12741F", ax = ax).set_title("T R U E   N E W S");



#average word length in fake news

fig, ax = plt.subplots(figsize = (15, 8))
text_words = news[news["news_class"] == 0]["text"].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(text_words.map(lambda x: np.mean(x)), color = "#AC0C1D", ax = ax).set_title("F A K E   N E W S");



#divide the dataset into test and train sets

x = news["text"]
y = news["news_class"]

train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    test_size = 0.20,
                                                    shuffle = True,
                                                    random_state = 11)

print(colored("\nDIVIDED SUCCESFULLY...", "green"))


print(train_x.shape, test_x.shape)



tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(train_x)

tokenized_train = tokenizer.texts_to_sequences(train_x)
tokenized_test = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(tokenized_train, maxlen = 300)
test_x = sequence.pad_sequences(tokenized_test, maxlen = 300)



GLOVE_EMBEDDING = "/content/drive/MyDrive/pioneer/glove.twitter.27B.100d.txt"



def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype = "float32")
embeddings_index = dict(get_coefs(*g.rstrip().rsplit(" ")) for g in open(GLOVE_EMBEDDING))

#_________________________________________________________________________________________#

#_________________________________________________________________________________________#

embeddings = np.stack(embeddings_index.values())
embedding_mean, embedding_std = embeddings.mean(), embeddings.std()
embedding_size = embeddings.shape[1]

word_index = tokenizer.word_index
nb_words = min(10000, len(word_index))

embedding_matrix = embedding_matrix = np.random.normal(embedding_mean, embedding_std, (nb_words, embedding_size))
for word, i in word_index.items():
    if i >= 10000:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


lr_reduce = ReduceLROnPlateau(monitor = "val_accuracy", patience = 2, factor = 0.5, min_lr = 0.00001)



model = Sequential()

model.add(Embedding(10000,
                    output_dim = 100,
                    weights = [embedding_matrix],
                    input_length = 300,
                    trainable = False))

model.add(LSTM(units = 128,
               return_sequences = True,
               recurrent_dropout = 0.3,
               dropout = 0.3))

model.add(LSTM(units = 64,
               recurrent_dropout = 0.15,
               dropout = 0.15))

model.add(Dense(units = 32,
                activation = "relu"))

model.add(Dense(1,
                activation = "sigmoid"))

model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.01),
              loss = "binary_crossentropy",
              metrics = ["accuracy"])


model.summary()

history = model.fit(train_x,
                    train_y,
                    batch_size = 128,
                    validation_data = (test_x, test_y),
                    epochs = 10,
                    callbacks = [lr_reduce])


epochs = [i for i in range(10)]

fig, ax = plt.subplots(1, 2)
train_acc = history.history["accuracy"]
train_loss = history.history["loss"]
val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]
fig.set_size_inches(20, 10)

ax[0].plot(epochs, train_acc, "go-", label = "Train accuracy")
ax[0].plot(epochs, val_acc, "ro-", label = "Test accuracy")
ax[0].set_title("Train and test accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, "go-", label = "Train loss")
ax[1].plot(epochs, val_loss, "ro-", label = "Test loss")
ax[1].set_title("Train and test loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()


prediction = model.predict(test_x)
classes_pred = np.argmax(prediction, axis = 1)

print(classification_report(test_y, classes_pred))

conf_mat = confusion_matrix(test_y, classes_pred)
print(conf_mat)

sns.heatmap(conf_mat, square = True, annot = True, robust = True)
plt.show()
