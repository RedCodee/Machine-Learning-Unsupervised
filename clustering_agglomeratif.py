import scipy . cluster . hierarchy as shc
import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
import time
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score

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
databrut = arff.loadarff ( open ( path + "target.arff" , 'r') )
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

# Donnees dans datanp
print ("Dendrogramme 'single' donnees initiales")
linked_mat = shc . linkage ( datanp , 'single')
plt . figure ( figsize = ( 12 , 12 ) )
shc . dendrogram ( linked_mat ,
orientation = 'top' ,
distance_sort = 'descending' ,
show_leaf_counts = False )
plt . show ()

#SOIT METHODE 1 OU 2 on choisit en fonction du set distance_thre ou set nb cluster


# METHODE 1
#A EXPLIQUER CA CHANGE AVEC LA distance_threshold PPPQQQQQ
# set distance_threshold ( 0 ensures we compute the full tree )
#tps1 = time . time ()
#model = cluster . AgglomerativeClustering ( distance_threshold = 1 ,linkage = 'single' , n_clusters = None )
#model = model . fit ( datanp )
#tps2 = time . time ()

# METHODE 2
k = 6
tps1 = time . time ()
model = cluster . AgglomerativeClustering ( linkage = 'single' , n_clusters = k )
model = model . fit ( datanp )
tps2 = time . time ()



labels = model . labels_
k = model . n_clusters_
leaves = model . n_leaves_
# Affichage clustering
plt . scatter ( f0 , f1 , c = labels , s = 8 )
plt . title ( " Resultat du clustering " )
plt . show ()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ")