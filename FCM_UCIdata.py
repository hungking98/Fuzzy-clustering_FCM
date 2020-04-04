import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from unicodedata import name
from fcmeans import FCM
from sklearn.datasets import make_blobs
import copy
df0 = pd.read_csv('./seeds_dataset.txt', sep ="\t", header =None, names = ["A", "P","C", "LK","WK","AC","L","cluster"])
A0 = df0['cluster']
del df0['cluster']

X = df0.to_numpy()

#số lượng phần tử dữ liệu
n = len(df0)
#tạo một mảng để gán nhãn dữ liệu
A = np.zeros(n)
#tham số m, số cụm c
m = float(input('Nhap tham so m:'))
c =3



#tạo ma trận C:
np.random.seed(500)
C = np.random.random(size=(c,7))


#tạo ma trận phụ thuộc U thỏa mãn các cột tổng bằng 1
U = np.random.random(size=(n,c))
for i in range(n):
    
    sum_of_Urow = sum(U[i])
    U[i] = U[i]/sum_of_Urow


#Update tam cum 
def Update_C(C):
    for j in range(c):
        tuso = 0.0
        mauso = 0.0
        for i in range(n):
            tuso += ((U[i,j])**m)*X[i]
            mauso += (U[i,j])**m
        C[j] = tuso/mauso

#Update Uik
def update_U_ik(i,k):  
   Uik = 0.0
   for j in range(c):
      Uik +=((np.linalg.norm(X[i]-C[k]))/np.linalg.norm(X[i]-C[j]))**(2/(m-1))
   Uik = 1/Uik
   return Uik

#Update ma tran U
def update_Umatrix(U):
   for i in range(n):
      for k in range(c):
         U[i,k] = update_U_ik(i,k)
   return U

#Tinh toan gia tri ham muc tieu J:
def J_uc():
   J= 0.0
   for i in range(n):
      for k in range(c):
         J += ((U[i,k])**m)*(np.linalg.norm(X[i]-C[k]))**2
   return J

def check_end_loop(a,b):
   if(np.linalg.norm(a-b)<0.01):
      return True
   else:
      return False

#Gán nhãn
def select_Cluster(A,U):
   
   for i in range(n):
      for j in range(c):
         if max(U[i]) == U[i,j]:
            A[i] =j+1
#test bằng FCM chuẩn
# fit the fuzzy-c-means
fcm = FCM(n_clusters=3)
fcm.fit(X)
fcm_centers = fcm.centers 
fcm_labels  = fcm.u.argmax(axis=1) +1
print('Tâm cụm chuẩn:', fcm_centers)

#hàm FCM_mean theo các hàm con đã viết
def Clustering_fcmean(C,U,df):
    count =1
    print('Trạng thái ban đầu:')
    print('Ma trận tâm C:\n',C)
    print('Ma trận hàm thuộc U:\n',U)
    print('---------------------')
    while True:
    
    
        print('Lần duyệt thứ:',count)
        U1  = copy.deepcopy(U)
        Update_C(C)
        U2  = copy.deepcopy(update_Umatrix(U))
        print('Ma trận tâm C:\n',C)
        print('Ma trận hàm thuộc U:\n',U)
        print('Độ lệch U so với lần trước:', np.linalg.norm(U2-U1))
        print('----------------------------\n')
        if  check_end_loop(U2,U1):
            break
        else :
            count+=1
    df['Ui0'] = U[:,0]
    df['Ui1'] = U[:,1]
    df['Ui2'] = U[:,2]
    select_Cluster(A,U)
    df['Cluster Sandard'] = A0
    df['Cluster Test'] = A
    df['Cluster FCM'] = fcm_labels
Clustering_fcmean(C,U,df0)
print(df0)
df0.to_csv (r'C:\Users\BacHung\Desktop\GR1\FuzzyCMeans\ketqua.csv', index = True, header=True)
print('Tâm cụm chuẩn:\n', fcm_centers)
print('Tâm cụm theo tính toán:\n',C)