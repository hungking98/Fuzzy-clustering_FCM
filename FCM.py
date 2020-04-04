import pandas as pd    # thư viện để tạo và truy vấn dữ liệu dạng bảng
import numpy as np     # thư viện liên quan đến xử lí mảng, rất mạnh với xử lí ma trận
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs   #thư viện tạo dữ liệu học máy
import copy              
from fcmeans import FCM


#creat data 2 chieu:
centers = [(-6, -5), (1, 0), (7, 8)]    #cái này là tâm chuẩn theo phân phối cụm nhá
X, y = make_blobs(n_samples=300, n_features=2, cluster_std=1.0,            #sinh 300 điểm dữ liệu phân bố đều vào 3 tâm
                  centers=centers, shuffle=False, random_state=42)
C1 = np.array(centers)    #mảng này lưu trữ tập tâm cụm chuẩn được sinh ra cùng với điểm dữ liệu
n  = len(X) #so lượng row của bảng data  = số điểm dữ liệu

 # Một mảng lưu giá trị phân nhóm sau khi tiến hành thuật toán A
A = np.zeros(n)
#creat matrix C,#creat số tâm cụm và trọng số mũ m

c= 3
m = float (input('Nhập tham số m:'))
C = np.random.random(size=(c,2))

#creat U_matrix: ma trận phụ thuộc
U = np.random.random(size=(n,c))
for i in range(n):
    sum_of_Urow  =0.0
    for j in range(c):
        sum_of_Urow +=U[i,j]
    U[i] = U[i]/sum_of_Urow



#Update tâm cụm

def update_C(C):
   for j in range(c):
     tuso = 0.0
     mauso= 0.0
     for i in range(n):
        tuso += ((U[i,j])**m)*X[i]
        mauso += (U[i,j])**m
     C[j] = tuso/mauso


#update Uik: độ phụ thuộc của vecto X[i] vào cụm thứ k
def update_U_ik(i,k):  
   Uik = 0.0
   for j in range(c):
      Uik +=((np.linalg.norm(X[i]-C[k]))/np.linalg.norm(X[i]-C[j]))**(2/(m-1))
   Uik = 1/Uik
   return Uik

#update ma tran U
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

def select_Cluster(A,U):
   
   for i in range(n):
      for j in range(c):
         if max(U[i]) == U[i,j]:
            A[i] =j
 

def Clustering_fcmean(C,U):
    count =1
    print('Trạng thái ban đầu:')
    print('Ma trận tâm C:\n',C)
    print('Ma trận hàm thuộc U:\n',U)
    print('---------------------')
    while True:
    
    
        print('Lần duyệt thứ:',count)
        U1  = copy.deepcopy(U)
        update_C(C)
        U2  = copy.deepcopy(update_Umatrix(U))
        print('Ma trận tâm C:\n',C)
        print('Ma trận hàm thuộc U:\n',U)
        print('Độ lệch U so với lần trước:', np.linalg.norm(U2-U1))
        print('----------------------------\n')
        if  check_end_loop(U2,U1):
            break
        else :
            count+=1
    select_Cluster(A,U) 

#Thử dùng FCM thư viện người ta viết    
fcm = FCM(n_clusters=3)
fcm.fit(X)
fcm_centers = fcm.centers

#Dùng hàng tôi viết
Clustering_fcmean(C,U)

#Biểu diễn ra hình
fig = plt.figure(figsize=(5, 5))
c_color =['r','g','y']
for i in range(n):
   if A[i] == 0:
      plt.scatter(X[i,0], X[i,1], color='c')
   elif A[i] == 1:
       plt.scatter(X[i,0], X[i,1], color='m')
   elif A[i] == 2:
       plt.scatter(X[i,0], X[i,1], color='g')
for i in range(c):
    plt.scatter(C1[i,0],C1[i,1],c = 'r')
    plt.scatter(C[i,0],C[i,1],c = 'k')

print('----------------------------------------')

print('Tâm cụm theo tính toán bằng FCM thư viện:\n',fcm_centers)
print('Tâm cụm theo thuật toán:\n', C)
plt.show()



