#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <SFML/Graphics.hpp>

using namespace std;

#include "Ensemble.cu"

// CHANGMENT //

__global__ void lance_calcul(Mandelbrot m, sf::Uint8 *p,int w, int h,int *b)
{
 m.calcul(p,w,h,b);
}
__global__ void lance_calcul(Julia j, sf::Uint8 *p,int w, int h,int *b)
{
 j.calcul(p,w,h,b);
}

__global__ void calc(Ensemble *e, sf::Uint8 *p,int w, int h,int *b)
{
  //printf("%d\n",e->init_z_real(0,0,0));
  //printf("coucou");
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
  int idx = w*row + col;

  //complex<double> c( ((double)col / w) * 3 - 2.1f,((double)row / w) * 3 - 1.2f);
  //complex<double> c(0.285,0.01);
  complex<double> c = e->init_c(col,row,w);

  //complex<double> z(0,0);
  //complex<double> z( ((double)col / w) * 3 - 1.1f,((double)row / w) * 3 + 1.2f);
  complex<double> z = e->init_z(col,row,w);

  double z_real = z.real();
  double z_imag = z.imag();

  float i = 0;

  // On est oblige de decomposer les calcul pour les complex car les operator +,*,= de complex<double> sont inaccessible avec __device_
  while ( (z_real*z_real+ z_imag*z_imag) < 4.0f && i < e->getIterationMax()) // |z| < 2 et i < nb_iteration_max |||| zi² + z² < 4
  {
    double tempo = z_real;
    z_real = z_real*z_real - z_imag*z_imag + c.real();
    z_imag =  2*z_imag *tempo +c.imag() ;
    ++i;
  }

if ( i != e->getIterationMax())
{
  int color = i * 5;
  if (color >= 256) color = 0;
  /*p[idx] = 0;
  p[idx+1] = 0;
  p[idx+2] = i;
  p[idx+3] = 255;*/
  b[idx] = color;//i * 255 / e->getIterationMax();
}
else
{/*
  p[idx] = 0;
  p[idx+1] = 0;
  p[idx+2] = 0;
  p[idx+3] = 255;*/
  b[idx] = 0;
}

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

int main(void)
{
  const int width = 512;
  const int height = 512;
  const int iteration_max = 1000;

  Ensemble *e,*d_e;
  Mandelbrot m_p,d_m;
  sf::Uint8 *pixels,*d_pixels;
  int *b,*d_b;

  Mandelbrot m(iteration_max);
  Julia j(iteration_max);

  pixels = new sf::Uint8[width*height*4];
  b = new int[width * height];

  cudaMalloc(&d_e,sizeof(Ensemble&));
  cudaMalloc(&d_pixels,sizeof(sf::Uint8) * width * height);
  cudaMalloc(&d_b,sizeof(int) * width * height);

  dim3 bloc(16, 16);
  dim3 grille(width / bloc.x, height / bloc.y);
  //lance_calcul<<<grille,bloc>>>(m,d_pixels,width,height,d_b);

  // JULIA //
  lance_calcul<<<grille,bloc>>>(j,d_pixels,width,height,d_b);
  printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaDeviceSynchronize();
  //cudaMemcpy(pixels,d_pixels,sizeof(sf::Uint8) * width * height,cudaMemcpyDeviceToHost);
  cudaMemcpy(b,d_b,sizeof(int) * width * height,cudaMemcpyDeviceToHost);
  e = &j;
  savePicture(e,width,height,pixels,b);

  // MANDELBROT //
  lance_calcul<<<grille,bloc>>>(m,d_pixels,width,height,d_b);
  printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaDeviceSynchronize();
  cudaMemcpy(b,d_b,sizeof(int) * width * height,cudaMemcpyDeviceToHost);
  e = &m;
  savePicture(e,width,height,pixels,b);


  cudaDeviceReset();

  delete [] b;
  delete [] pixels;

  cudaFree(d_e);
  cudaFree(d_pixels);

  return 0;
}
