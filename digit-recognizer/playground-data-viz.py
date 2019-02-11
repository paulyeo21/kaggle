import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
X_train = (train.iloc[:,1:].values).astype("float32") # all pixel values
y_train = train.iloc[:,0].values.astype("int32") # only labels i.e targets digits

# convert train datset to (num_images, img_rows, img_cols) format 
X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap("gray"))
    plt.title(y_train[i]);
    plt.show()
