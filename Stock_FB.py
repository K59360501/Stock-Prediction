import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#ดึง Dataset เฉพาะ 5 แถวแรก
url = "https://gitlab.com/59362345/softengineering/blob/master/FB.csv"
df = pd.read_csv("FB.csv")
#print(df.head())

#Show dataset เฉพาะคอลัมน์ Adj Close
df = df[['Adj Close']]  
print(df.head())


#สร้างตัวแปรทำนายวันข้างหน้า
predictDay = 30 #จำนวนวันที่ต้องการ predict
#สร้างคอลัมน์ Prediction แล้วทำการ shift ข้อมูลไป 30 แถว ข้อมูล30 แถวสุดท้าย ทำให้เป็น null
df['Prediction'] = df[['Adj Close']].shift(-predictDay)
print(df.tail())

X = np.array(df.drop(['Prediction'],1))
X = X[:-predictDay]
print("แสดงข้อมูลอาเรย์ 30แถวในแนวตั้ง : ", X)

y = np.array(df['Prediction'])
y = y[:-predictDay]
print("แสดงข้อมูลอาเรย์ 30แถวในแนวนอน : ", y)

#Train 80% และ Test 20%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#เปรียบเทียบความเป็นไปได้ที่ใกล้เคียงค่าคาดคะเนที่สุดระหว่าง SVR กับ LN โดยใช้การ Train80% Test20%
#เป็นคะแนนสัมประสิทธิ์ในการตัดสินใจว่าจะเลือกเชื่อแบบไหน
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)
svm_possible = svr_rbf.score(x_test, y_test)
print("svm possible: ", svm_possible)

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_possible = lr.score(x_test, y_test)
print("lr possible: ", lr_possible)

#สร้าง x_predict ดรอปคอลัมน์ prediction ที่เราสร้างออก กรอกข้อมูลที่เราทำนายได้เก็บไว้ในตัวแปร x_predict แล้วแปลงเป็น Array ทั้งหมด 30 แถว
x_predict = np.array(df.drop(['Prediction'],1))[-predictDay:]
print(x_predict)

#Print model LR ที่เรา predict มา 30 วัน
#Print model SVR ที่เรา predict มา 30 วัน
lr_prediction = lr.predict(x_predict)
print("Predict โดย Linear Regression : ",lr_prediction)
svm_prediction = svr_rbf.predict(x_predict)
print("Predict โดย Support Vector Regression : ",svm_prediction)


