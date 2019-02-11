import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')

# Plot train Data
# count = train['label'].value_counts().sort_values()
# count.plot.bar()
# plt.show()

labels = np.array(train['label'])

train = np.array(train.drop('label', axis=1))

train_data, test_data, train_label, test_label = train_test_split(train, labels, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(train_data, train_label)

test_accuracy = rf.score(test_data, test_label)
print(test_accuracy)

test = np.array(pd.read_csv('test.csv'))

result = rf.predict(test)

with open('rf_submission1.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['ImageId', 'Label'])
    for i in range(len(result)):
        filewriter.writerow([i + 1, result[i]])
