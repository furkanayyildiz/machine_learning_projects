############ STEP 1 ################
from tkinter import *
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv('Mall_Customers.csv')

############ STEP 2 ################

print("Number of Rows", data.shape[0])
print("Number of Columns", data.shape[1])

############ STEP 3 ################
data.info()

print(data.isnull().sum())


############ STEP 4 ################
A = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(A.head())


km = KMeans()
km.fit(A)

km = KMeans(n_clusters=5)
print(km.fit_predict(A))

############ STEP 5 ################
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(A)
    wcss.append(km.inertia_)

wcss


plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

############ STEP 6 ################

A = data[['Annual Income (k$)', 'Spending Score (1-100)']]

km = KMeans(n_clusters=5, random_state=42)
ym = km.fit_predict(A)


plt.scatter(A.iloc[ym == 0, 0], A.iloc[ym == 0, 1],
            s=100, c='orange', label="C1")
plt.scatter(A.iloc[ym == 1, 0], A.iloc[ym == 1, 1],
            s=100, c='purple', label="C2")
plt.scatter(A.iloc[ym == 2, 0], A.iloc[ym == 2, 1],
            s=100, c='brown', label="C3")
plt.scatter(A.iloc[ym == 3, 0], A.iloc[ym == 3, 1],
            s=100, c='blue', label="C4")
plt.scatter(A.iloc[ym == 4, 0], A.iloc[ym == 4, 1],
            s=100, c='black', label="C5")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1], s=100, c="yellow")
plt.title("Customer Data")
plt.xlabel("Customer Earning")
plt.ylabel("Customer Expenditure")
plt.legend()
plt.show()

print(km.predict([[80, 80]]))

############ STEP 7 ################
joblib.dump(km, "customer_income_expenditure")

pattern = joblib.load("customer_income_expenditure")


############ STEP 8 ################
def show_entry_fields():
    k1 = int(m1.get())
    k2 = int(m2.get())

    pattern = joblib.load('customer_income_expenditure')
    result = pattern.predict([[k1, k2]])

    match result[0]:
        case  0:
            Label(main, text="mid earning and mid expenditure").grid(row=31)
        case  1:
            Label(main, text="high earning but less expenditure").grid(row=31)
        case  2:
            Label(main, text="less earning and less expenditure").grid(row=31)
        case  3:
            Label(main, text="less earning but high expenditure").grid(row=31)
        case  4:
            Label(main, text="high earning and high expenditure").grid(row=31)


main = Tk()
main.title("Customer income-expenditure Analysis")


label = Label(main, text="Customer income-expenditure Analysis", bg="blue", fg="yellow"). \
    grid(row=0, columnspan=2)

Label(main, text="Income").grid(row=1)
Label(main, text="Expenditure").grid(row=2)

m1 = Entry(main)
m2 = Entry(main)

m1.grid(row=1, column=1)
m2.grid(row=2, column=1)

Button(main, text='Analyze', command=show_entry_fields).grid()
mainloop()
