#Μετά τα πειράματα παρατήρησα ότι για μικρές τιμές παραθύρου η εικόνα
#αλλοιώνεται κατά πολύ αλλά η επίδοση του αλγορίθμου είναι καλή ενώ
#για μεγάλο μέγεθος παραθύρου η εικόνα δεν αλλοιώεται αλλά η επίδοση
# του αλγορίθμου πέφτει. Για να επιτευχθεί μια σχετική βελτίωση του
#αλγορίθμου θα μπορούσα να μην κοιτάξω όλες τις τιμές των κατωφλίων 
#αλλά σε αυτή την περίπτωση η εικόνα αλλοιώνεται, όμως όχι αισθητά.  
import sys
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image

#Ελέγχω αν μια εικόνα είναι έγχρωμη και αν είναι
#την μετατρέπει σε ασπρόμαυρη με τον υπολγισμό του
#μέσου όρου των τριών καναλιών RGB. Αν δεν είναι
#έγχρωμη επιστρέφεται απλά η ασπρόμαυρη εικόνα
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

#Κάνω την κατωφλίωση της αρχικής εικόνας συγκρίνοντας
#κάθε κελί της με το αντίστοιχο κελί του πίνακα των κατωφλίων
#που προέκυψε απο τους υποπίνακες και την μέθοδο Otsu
def thresholdImage(imageArray,r,c,t):
    array1= np.zeros([r,c])
    for i in range(r):
        for j in range(c):
            if imageArray[i][j] > t[i][j]:
                array1[i][j] = 255
            else:
                array1[i][j]= 0
    return array1

#Για κάθε πίξελ της αρχικής εικόνας υπολογίζω τον υποπίνακα
#όπου το window size είναι άρτιο. Ελέγχω αν τα στοιχεία
#του υποπίνακα βρίσκονται εντός της εικόνας και κρατάω τις τιμές
#του υποπίνακα σε ένα μονοδιάστατο πίνακα. Κρατάω μόνο τα στοιχεία του 
#υποπίνακα που είναι εντός του πλέγματος της αρχικής εικόνας. Με τον 
#τελευταίο πίνακα εκτελώ την μέθοδο Otsu για να βρω το καλύτερο κατώφλι 
#και να το αποθηκεύσω στον πίνακα κατωφλίων.    
def forOddSize(w,r,c):
    n = int(w/2)
    threshold = np.zeros([r,c])
    for i in range(r):
        for j in range(c):
            OtsuArray = np.array([])
            #Χρήση κατάλληλων ορίων για τον υποπίνακα
            for l in range(-n,n+1):
                for k in range(-n,n+1):
                    #Έλγχος για πίξελ του υποπίνακα εκτός ορίων
                    if((i+l)>=0 and (j+k)>=0 and (i+l)<r and (j+k)<c ):
                        g = np.array([newImage[i+l][j+k]])
                        OtsuArray = np.append(OtsuArray,g)
            bestThreshold=0
            bestPrice=0
            for k in range(1,256):
                pixels1 = OtsuArray[OtsuArray<k]
                pixels2 = OtsuArray[OtsuArray>=k]
                #Έλεγχος αν τα πίξελ δεν βρίσκονται σε ένα από τα δύο διαστήματα
                if not((len(pixels1)==0 and len(pixels2)==0) or (len(pixels1)==0 and len(pixels2)!=0) or (len(pixels1)!=0 and len(pixels2)==0)):
                    mu1 = np.mean(pixels1)
                    mu2 = np.mean(pixels2)
                    mu_synoliko = np.mean(OtsuArray.flatten())
                    pi1 = len(pixels1) / (len(pixels1) + len(pixels2))
                    pi2 = len(pixels2) / (len(pixels1) + len(pixels2))
                    newBestPrice = pi1 * (mu1 - mu_synoliko)**2 + pi2 * (mu2 - mu_synoliko)**2
                    if (newBestPrice>bestPrice):
                        bestThreshold = k
                        bestPrice = newBestPrice
            threshold[i][j] = bestThreshold
    return threshold
    
#Για κάθε πίξελ της αρχικής εικόνας υπολογίζω τον υποπίνακα
#όπου το window size είναι περιττό. Ελέγχω αν τα στοιχεία
#του υποπίνακα βρίσκονται εντός της εικόνας και κρατάω τις τιμές
#του υποπίνακα σε ένα μονοδιάστατο πίνακα. Κρατάω μόνο τα στοιχεία του 
#υποπίνακα που είναι εντός του πλέγματος της αρχικής εικόνας. Με τον 
#τελευταίο πίνακα εκτελώ την μέθοδο Otsu για να βρω το καλύτερο κατώφλι 
#και να το αποθηκεύσω στον πίνακα κατωφλίων.  
def forEvenSize(w,r,c):
    n = int(w/2)
    threshold = np.zeros([r,c])
    for i in range(r):
        for j in range(c):
            OtsuArray = np.array([])
            #Χρήση κατάλληλων ορίων για τον υποπίνακα
            for l in range(-n,n+1):
                for k in range(-n,n+1):
                    #Έλγχος για πίξελ του υποπίνακα εκτός ορίων
                    if((i+l)>=0 and (j+k)>=0 and (i+l)<r and (j+k)<c ):
                        g = np.array([newImage[i+l][j+k]])
                        OtsuArray = np.append(OtsuArray,g)
            bestThreshold=0
            bestPrice=0
            for k in range(1,256):
                pixels1 = OtsuArray[OtsuArray<k]
                pixels2 = OtsuArray[OtsuArray>=k]
                #Έλεγχος αν τα πίξελ δεν βρίσκονται σε ένα από τα δύο διαστήματα
                if not((len(pixels1)==0 and len(pixels2)==0) or (len(pixels1)==0 and len(pixels2)!=0) or (len(pixels1)!=0 and len(pixels2)==0)):
                    mu1 = np.mean(pixels1)
                    mu2 = np.mean(pixels2)
                    mu_synoliko = np.mean(OtsuArray.flatten())
                    pi1 = len(pixels1) / (len(pixels1) + len(pixels2))
                    pi2 = len(pixels2) / (len(pixels1) + len(pixels2))
                    newBestPrice = pi1 * (mu1 - mu_synoliko)**2 + pi2 * (mu2 - mu_synoliko)**2
                    if (newBestPrice>bestPrice):
                        bestThreshold = k
                        bestPrice = newBestPrice
            threshold[i][j] = bestThreshold
    return threshold
                
#Άνοιγμα, αποθήκευση και εμφάνιση της αρχικής εικόνας                       
image = sys.argv[1]
f = np.array(Image.open(image))
rows,columns = f.shape[0],f.shape[1]
print("(rows,colums) = (",f.shape[0],',',f.shape[1],')')
print("image = ",f)
plt.imshow(f, cmap="gray")
plt.show()

#Έλεγχος αν η εικόνα είναι έγχρωμη  
newImage = np.zeros([rows,columns])
newImage = if_RGB_Change(f,rows,columns)

#Έλεγχος αν το window size είναι περιττό ή άρτιο
#για να καλέσω την κατάλληλη συνάρτηση που δημιουργεί
#σωστά τον υποπίνακα για περιττό ή άρτιο μέγεθος αντίστοιχα
window_size = int(sys.argv[3])
thresholds = np.zeros([rows,columns])
if(window_size%2==0):
    thresholds=forEvenSize(window_size,rows,columns)
else:
    thresholds=forOddSize(window_size,rows,columns)       

#Κατωφλίωση της αρχικής εικόνας          
thresholdedImage = np.zeros([rows,columns])
thresholdedImage = thresholdImage(newImage,rows,columns,thresholds)

#Εμφάνιση και αποθήκευση της τελικής εικόνας
saveImage = sys.argv[2]
plt.imshow(thresholdedImage,cmap="gray")
plt.show()
Image.fromarray(thresholdedImage.astype(np.uint8)).save(saveImage)

