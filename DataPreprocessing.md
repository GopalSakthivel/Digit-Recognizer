
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpg

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools

from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau 

train = pd.read_csv("E:/Gopal DS/Python/Codes work/Digit/train.csv")
test = pd.read_csv("E:/Gopal DS/Python/Codes work/Digit/test.csv")

y_train=train["label"]

x_train=train.drop(labels=["label"],axis=1)

del train

sns.set(style="white",context="notebook",palette="deep")
sns.countplot(y_train)

y_train.value_counts()

x_train.isnull().any().describe()
test.isnull().any().describe()

x_train=x_train/255
test=test/255

x_train=x_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)

y_train=to_categorical(y_train)

X_train,X_val,Y_train,Y_val=train_test_split(x_train,y_train,test_size=0.1)

plt.imshow(X_train[0][:,:,0])
