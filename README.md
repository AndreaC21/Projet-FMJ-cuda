# Projet-FMJ-cuda

Calcul des ensembles de Mandelbrot et Julia avec cuda

Pré-requis : 
* SFML 
* cuda 11
	
Compilation :

nvcc -arch=compute_50 -code=sm_50 main.cu -o fractales -lsfml-graphics --expt-relaxed-constexpr

-lsfml-graphics : utilisation des images de SFML
--expt-relaxed-constexpr : utilisation des opérations sur les nombres complexes

Avant la première compilation :

* export PATH=/usr/local/cuda-11.1/bin/:$PATH
* export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LB_LIBRARY_PATH




