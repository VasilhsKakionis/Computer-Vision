#Κακιώνης Βασίλειος ΑΜ2981
# Μετά τα πειράματα με τις 2 εικόνες παρατήρησα ότι δεν υπάρχει 
# κατάλληλο κατώφλι αφού αν αυτό είναι πολύ χαμηλό τότε 
# η εικόνα αλλοιώνεται επειδή έχει πάρα πολλά άσπρα εικονοστοιχεία
# ενώ αν ειναι πολύ υψηλό τότε η εικόνα αλλοιώνεται επειδή
# έχει πάρα πολλά μαύρα εικονοστοιχεία. Οπότε μπορούμε να αποδώσουμε  
# μια σχετικά κατάλληλη τιμή του κατωφλίου μετά απο πειράματα
# η οποία αλλάζει από εικόνα σε εικόνα.
import sys
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image


def if_RGB_Change(im,r,c):
    im = double (im)
    change = np.zeros([r,c])
    if(len(im.shape) == 3):
        for i in range(r):
            for j in range(c):
                change[i][j] = (im[i][j][0]+im[i][j][1]+im[i][j][2])/3
        print("Grey image")
        plt.imshow(change,cmap="gray")
        plt.show()
        return change 
    else:
        return im
    
def thresholdImage(imageArray,r,c,t):
    array1= np.zeros([r,c])
    for i in range(r):
        for j in range(c):
            if imageArray[i][j] > t:
                array1[i][j] = 255
            else:
                array1[i][j]= 0
    return array1


    
image = sys.argv[1]
saveImage = sys.argv[2]
threshold = int(sys.argv[3])

#Άνοιγμα εικόνας, αποθήκευση και προβολή της
#καθώς και υπολογισμό των γραμμών και των
#στηλών της εικόνας
f = np.array(Image.open(image))
rows,columns = f.shape[0],f.shape[1]
print("(rows,colums) = (",f.shape[0],',',f.shape[1],')')
print("image = ",f)
plt.imshow(f, cmap="gray")
plt.show()
    
#Έλεγχος αν μια εικόνα είναι RGB ή Greyscale και αν είναι RGB μετατροπή σε Greyscale
#με τον μέσο όρο των τριών καναλιών της RGB εικόνας  
newImage = np.zeros([rows,columns])
newImage = if_RGB_Change(f,rows,columns)

#Υπολογισμός της καταφλιωμένης εικόνα, καθώς και η εμφάνιση και αποθήκευσή της
thresholdedImage = np.zeros([rows,columns])
thresholdedImage = thresholdImage(newImage,rows,columns,threshold)
print("threshold = ",threshold)    
plt.imshow(thresholdedImage,cmap="gray")
plt.show()
Image.fromarray(thresholdedImage.astype(np.uint8)).save(saveImage)
