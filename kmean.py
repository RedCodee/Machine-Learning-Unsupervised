import matplotlib . pyplot as plt
from scipy . io import arff
import time
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn.metrics import silhouette_score


# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
path = '/home/kharoubi/tp_un/Machine-Learning-Unsupervised/clustering-benchmark-master/src/main/resources/datasets/artificial/'
databrut = arff.loadarff ( open ( path + "D31.arff" , 'r') ) #xtarget
datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
f0 = [x[0] for x in datanp] # tous les elements de la premiere colonne
f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne
plt . scatter ( f0 , f1 , s = 8 )
plt . title ( " Donnees initiales " )
plt . show ()
print ( " Appel KMeans pour une valeur fixee de k " )


# **********Graph après clustering Kmeans
k = 31
tps1 = time.time ()
model = cluster.KMeans (n_clusters = k, init ='k-means++', n_init=10)
model . fit ( datanp )
tps2 = time . time ()
labels = model . labels_
iteration = model . n_iter_
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Donnees apres clustering Kmeans " )
plt . show ()
print ( " nb clusters = " ,k , " , nb iter = " , iteration , " , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ),"ms")



# *********Graph du coefficient de silouhette permettant de trouver le nombre de cluster le plus adapté
# Initialiser une liste pour stocker les scores de silhouette pour chaque k
silhouette_scores = []

# Boucle sur différentes valeurs de k de 2 à 20
for k in range(2, 6):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10)
    labels = model.fit_predict(datanp)
    silhouette_avg = silhouette_score(datanp, labels)
    silhouette_scores.append(silhouette_avg)
  #  print(f"Pour k={k}, le coefficient de silhouette moyen est : {silhouette_avg}")

# Tracer le graphique des scores de silhouette en fonction de k
plt.plot(range(2, 6), silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Coefficient de silhouette moyen')
plt.title('Évolution du coefficient de silhouette en fonction de k')
plt.show()


