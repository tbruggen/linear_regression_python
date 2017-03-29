import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframeChallenge = pd.read_csv('challenge_dataset.txt')
x_challenge = dataframeChallenge[['X']]
y_challenge = dataframeChallenge[['Y']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_challenge, y_challenge)
predicts = body_reg.predict(x_challenge)

#visualize results
plt.scatter(x_challenge, y_challenge)
plt.plot(x_challenge, body_reg.predict(x_challenge))
plt.show()


#print errors
errors = abs(predicts - y_challenge)
print("errors:")
for num in errors.Y:
        print ("\t", num)