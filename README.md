# Projet-FMJ-cuda

Calcul des ensembles de Mandelbrot et Julia avec cuda

### Description :

Ce programme va générer des images issue d'un calcul de **Mandelbrot** ou **Julia**, puis créer une vidéo a partir de ces images.
* Les images sont stockés dans la dossier Resultat.
* Les vidéos sont stockés dans le dossier courant

Pour Mandelbrot, un zoom sera effectué sur la partie inférieur droite
Pour Julia, un changement de la constante c est effectué.

Les constantes sont **width**,**height**,**nb_iteration** sont definis au debut du main

### Fichier config :

* config_mandelbrot.txt : zoom sur la partie inferieure droite
* config_julia.txt : contient des jolies valeurs de la constante c

### Pré-requis : 
* SFML 
* cuda 11
	
### Compilation :

`nvcc -arch=compute_50 -code=sm_50 main.cu -o fractales -lsfml-graphics --expt-relaxed-constexpr`

* -arch=compute_50
* -code=sm_50 
*pour la compatibilité des versions netre nvidia et cuda avec ma machine, je dois utiliser ces options*

* -lsfml-graphics : utilisation des images de SFML
* --expt-relaxed-constexpr : utilisation des opérations sur les nombres complexes

### Avant la première compilation :

* export PATH=/usr/local/cuda-11.1/bin/:$PATH
* export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LB_LIBRARY_PATH





