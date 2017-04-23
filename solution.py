import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt



#challenge

#dataframe = pd.read_fwf('linear_regression_demo\\challenge_dataset.txt')
df = pd.read_csv('linear_regression_demo\\challenge_dataset.txt', 
                 header=None, usecols=[0, 1], names = ["Independent", "Dependent"])


x_values = df[["Independent"]]
y_values = df[["Dependent"]]

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values)

y_predicted = body_reg.predict(x_values)

for i in range(len(y_predicted)):
    s = "error for index "
    s += str(i)
    s += " =  "
    s += str(y_predicted - y_values)
    print(s)


plt.plot(x_values, y_predicted)
plt.show()


#
#
#dataframe = pd.read_fwf('linear_regression_demo\\brain_body.txt')
#x_values = dataframe[['Brain']]
#y_values = dataframe[['Body']]
#
#body_reg = linear_model.LinearRegression()
#body_reg.fit(x_values, y_values)
#plt.scatter(x_values, y_values)
#plt.plot(x_values, body_reg.predict(x_values))
#plt.show()
