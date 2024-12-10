from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.2, shuffle=True, stratify=labels)

# Initialize SVM model
model = SVC(kernel='rbf', C=1, gamma='scale')

# Train the model
model.fit(x_train, y_train)

# Test the model
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_test, y_predict)
print("Accuracy Score: ", score)

# Save the model
with open('svm_model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)
