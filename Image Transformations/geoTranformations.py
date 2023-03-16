#Κακιώνης Βασίλειος ΑΜ2981
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def changeToNewCoordinates(r,c):
    A = np.zeros([3,r*c])
    index=0
    for i in range(r):
        for j in range(c):
            A[0,index] = j-c/2
            A[1,index] = i-r/2
            A[2,index]=1
            index+=1
    return A

def changeToOldCoordinates(G,r,c):
    B = np.zeros([3,r*c])
    index=0
    for i in range(r*c):
        B[0,index] = G[0,index] + c/2
        B[1,index] = G[1,index] + r/2
        B[2,index] = 1
        index+=1
    return B
    
def calculateNearNeighbor(Tr,grid,oldImage,r,c):
    pos=0
    image = np.zeros([r,c])
    for i in range(r):
        for j in range(c):
            minDistance = 50000
            for k in range(r*c):  
                #Υπολγισμός της ελάχιστης ευκλείδιας απόστασης
                #για τον υπολογισμό του κοντινότερου γείτονα
                distanceX = (Tr[0][k]-i)**2
                distanceY = (Tr[1][k]-j)**2
                EuclideanDistance = (distanceX+distanceY)**(1/2)
                if EuclideanDistance < minDistance:
                    minDistance = EuclideanDistance
                    pos = k
            #Υπολογισμός της καινούργιας εικόνας που περιλαμβάνει 
            #και τις φωτεινότητες απο των κοντινότερων γειτόνων
            x = grid[0,pos]
            y = grid[1,pos]
            image[i][j]=oldImage[int(x)][int(y)]
    return image

#Άνοιγμα της εικόνας, η αποθήκευση της,
#ο υπολογισμός γραμμών και στηλών της 
#και ο η εμφάνιση τους
filename = sys.argv[1]
firstImage = np.array(Image.open(filename))
rows,columns = firstImage.shape[0],firstImage.shape[1]
print("rows = ",rows," columns = ",columns)
print("") 
plt.imshow(firstImage,cmap="gray")
plt.show()

#Αποθήκευση των μεταβλητών για τον μετασχηματισμό 
a1 = float(sys.argv[3]) 
a2 = float(sys.argv[4]) 
a3 = float(sys.argv[5]) 
a4 = float(sys.argv[6]) 
a5 = float(sys.argv[7]) 
a6 = float(sys.argv[8]) 

#Δημιουργία αφινικού πίνακα
T = np.array([[a1,a2,a3],[a4,a5,a6],[0,0,1]])
print("Taffine = ",T)
print("")

#Δημιουργία του καινούργιου πλέγματος συντεταγμένων [-50,50]
grid1 = np.zeros([3,rows*columns])
grid1 = changeToNewCoordinates(rows,columns)
print("First grid = ")
print(grid1)
print("")

#Μετασχηματισμός        
Transfomation = T@grid1
print("Transfomation = ")
print(Transfomation)
print("")

#Αλλαγή συντεταγμένων του κανούργιου πλέγματος στο παλιό [0,101]
newGrid = np.zeros([3,rows*columns])
newGrid = changeToOldCoordinates(grid1,rows,columns)
print("New grid = ")
print(newGrid)
print("")

#Αλλαγή συντεταγμένων του μετασχηματισμό στο παλιό πλέγμα [0,101]
newTransformation = np.zeros([3,rows*columns])
newTransformation = changeToOldCoordinates(Transfomation,rows,columns)
print("New Transformation = ")
print(newTransformation)
print("")

#Παρεμβολή του κοντινότερου γείτονα και δημιουργία της μετασχηματισμένης εικόνας
newImage = np.zeros([rows,columns])
newImage = calculateNearNeighbor(newTransformation,newGrid,firstImage,rows,columns)
plt.imshow(newImage,cmap="gray")
plt.show()
Image.fromarray(newImage.astype(np.uint8)).save(sys.argv[2])              