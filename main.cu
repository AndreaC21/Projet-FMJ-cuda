#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <SFML/Graphics.hpp>
#include <vector>

using namespace std;

#include "Ensemble.cu"

__global__ void lance_calcul(Mandelbrot m, sf::Uint8 *p,int w, int h,int *b)
{
 m.calcul(p,w,h,b);
}
__global__ void lance_calcul(Julia j, sf::Uint8 *p,int w, int h,int *b)
{
 j.calcul(p,w,h,b);
}

void savePicture(Ensemble *e,int w, int h, sf::Uint8* pixels, int*b)
{
  int index_b = 0;
  for (int i = 0; i < w*h*4 ; i+=4)
  {
    pixels[i] = 0;
    pixels[i+1] = 0;
    pixels[i+2] = b[index_b];
    pixels[i+3] = 255;
    ++index_b;
  }
  e->saveImage(w,h,pixels);
}
enum fractales { mandelbrot=0, julia=1};

int main(void)
{
  const int width = 1024;
  const int height = 1024;
  const int iteration_max = 1000;
  //const fractales fractale_choisi = 0;

  vector<Mandelbrot> list_mand = {Mandelbrot(iteration_max)};

  // vers bas droite
  list_mand.push_back(Mandelbrot(-1.2f,-0.7f,iteration_max,2,1));
  list_mand.push_back(Mandelbrot(-0.5f,0.2f,iteration_max,1,2));

  //vers haut droite
  list_mand.push_back(Mandelbrot(-0.5f,-0.7f,iteration_max,1,3));
  list_mand.push_back(Mandelbrot(-0.5f,-1.2f,iteration_max,1,4));

  vector<Julia> list_Julia = {Julia(iteration_max)};

  vector<complex<double>> list_complexC;
  list_complexC.push_back(complex<double>(0.5f,0.01f));
  list_complexC.push_back(complex<double>(0.3f,0.01f));
  list_complexC.push_back(complex<double>(-0.75f,0.01f));
  list_complexC.push_back(complex<double>(-0.5f,0.64f));
  list_complexC.push_back(complex<double>(-0.818f,0.165f));
  list_complexC.push_back(complex<double>(-0.51f,0.56f));

  for (int i = 0 ; i< list_complexC.size() ; ++i)
  {
    list_Julia.push_back(Julia(list_complexC[i],iteration_max,i));
  }

  sf::Uint8 *pixels,*d_pixels;
  int *b,*d_b;

  pixels = new sf::Uint8[width*height*4];
  b = new int[width * height];

  cudaMalloc(&d_pixels,sizeof(sf::Uint8) * width * height);
  cudaMalloc(&d_b,sizeof(int) * width * height);

  dim3 bloc(16, 16);
  dim3 grille(width / bloc.x, height / bloc.y);

  // JULIA //
  for ( int i = 0 ; i < list_Julia.size() ; ++i)
  {
    lance_calcul<<<grille,bloc>>>(list_Julia[i],d_pixels,width,height,d_b);
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    //cudaMemcpy(pixels,d_pixels,sizeof(sf::Uint8) * width * height,cudaMemcpyDeviceToHost);
    cudaMemcpy(b,d_b,sizeof(int) * width * height,cudaMemcpyDeviceToHost);

    savePicture(&list_Julia[i],width,height,pixels,b);
  }
  // creation video
  system("ffmpeg -y -r 1 -i Resultat/Julia%d.png julia.mp4");

  // MANDELBROT //
  for ( int i = 0 ; i < list_mand.size() ; ++i)
  {
    lance_calcul<<<grille,bloc>>>(list_mand[i],d_pixels,width,height,d_b);
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    cudaMemcpy(b,d_b,sizeof(int) * width * height,cudaMemcpyDeviceToHost);
    savePicture(&list_mand[i],width,height,pixels,b);
  }

  // creation video
  system("ffmpeg -y -r 1 -i Resultat/Mandelbrot%d.png mandelbrot.mp4");

  cudaDeviceReset();

  delete [] b;
  delete [] pixels;

  cudaFree(d_pixels);
  cudaFree(d_b);

  cout << "\n\nDeux vidéos ont été générés dans le dossier courant:\n-mandelbrot.mp4 avec un zoom\n-julia.mp4 avec différentes valeurs de c\n\n" << endl;
  return 0;
}
