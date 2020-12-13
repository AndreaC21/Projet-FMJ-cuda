#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <SFML/Graphics.hpp>
//#include <cuda/std/*>
using namespace std;

class Ensemble
{

protected:
  float x,y;
  complex<double> c,z;
  int rang; // nombre iteration maximum
  sf::Image image;
  string nomImage;

public:
  Ensemble(){}
  Ensemble(float x,float y,int r)
  {
    this->x = x;
    this->y = y;
    this->rang = r;
  }
  Ensemble( Ensemble& e)
  {
    this->x = e.x;
    this->y = e.y;
    this->rang = e.rang;
    this->image = e.image;
    this->nomImage = e.nomImage;
  }
  ~Ensemble()
  {

  }
  __device__ virtual complex<double> init_c(int col, int row, int w){printf("here3");return complex<double>(0,0);}
  __device__ virtual complex<double> init_z(int col, int row, int w){return complex<double>(0,0);}

  __device__ virtual void calcul(sf::Uint8 *p,int w, int h,int *b) = 0;

  __device__ int getIterationMax() const
  {
    return this->rang;
  }
  __device__ sf::Image getImage() const
  {
    return this->image;
  }
  __device__ string getNom() const
  {
    return this->nomImage+".png";
  }

  __host__ void saveImage(int w,int h, sf::Uint8 *pixels)
  {
    this->image.create(w,h,pixels);
    this->nomImage.append("-size"+to_string(w)+"x"+to_string(w));
    this->image.saveToFile(this->nomImage+".png");
  }

};

class Mandelbrot : public Ensemble
{
public:
  // Mandelbrot est toujours compris entre -2.1 et 0.6 sur l'axe des abscisse et entre -1.2 et 1.2 sur l'axe des ordonnées.

  Mandelbrot(){}
  Mandelbrot(int iteration_max):Ensemble(-2.1f,-1.2f,iteration_max)
  {
    this->nomImage = "Mandelbrot";
    this->nomImage.append("-it_"+to_string(iteration_max));
  }

  __device__ virtual complex<double> init_c(int col, int row, int w) override
  {
    //c = x + y
    printf("here");
    return complex<double>( ((double)col / w) * 3 +this->x,((double)row / w) * 3 +this->y);
  }
  __device__ virtual complex<double> init_z(int col, int row, int w)
  {
    printf("z");
    // z0 = 0
    return complex<double>(0,0);
  }


  __device__ void calcul(sf::Uint8 *p,int w, int h,int *b) override
  {
    //printf("h");
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
    int idx = w*row + col;

    complex<double> c( ((double)col / w) * 3 - 2.1f,((double)row / w) * 3 - 1.2f);

    complex<double> z(0,0);
    double z_real = z.real();
    double z_imag = z.imag();

    float i = 0;

    // On est oblige de decomposer les calcul pour les complex car les operator +,*,= de complex<double> sont inaccessible avec __device_
    while ( (z_real*z_real+ z_imag*z_imag) < 4.0f && i < this->getIterationMax()) // |z| < 2 et i < nb_iteration_max |||| zi² + z² < 4
    {
      double tempo = z_real;
      z_real = z_real*z_real - z_imag*z_imag + c.real();
      z_imag =  2*z_imag *tempo +c.imag() ;
      ++i;
    }

  if ( i != this->getIterationMax())
  {
    int color = i * 5;
    if (color >= 256) color = 0;
    b[idx] = color;//i * 255 / e->getIterationMax();
  }
  else
  {
    b[idx] = 0;
  }

  }
};

class Julia : public Ensemble
{
public:

  Julia(){}
  Julia(int iteration_max):Ensemble(-1.1f,-1.1f,iteration_max)
  {
    this->nomImage = "Julia";
    this->nomImage.append("-it_"+to_string(iteration_max));
  }

  __device__ virtual complex<double> init_c(int col, int row, int w) override
  {
    //c = constante
    return complex<double>(0.285,0.01);
  }
  __device__ virtual complex<double> init_z(int col, int row, int w)
  {
    //z = x + y
      return complex<double>( ((double)col / w) * 3 +this->x,((double)row / w) * 3 +this->y);
  }

  __device__ void calcul(sf::Uint8 *p,int w, int h,int *b) override
  {
    //printf("h");
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
    int idx = w*row + col;

    complex<double> c(0.285,0.01);

    complex<double> z( ((double)col / w) * 3 +this->x,((double)row / w) * 3 +this->y);
    double z_real = z.real();
    double z_imag = z.imag();

    float i = 0;

    // On est oblige de decomposer les calcul pour les complex car les operator +,*,= de complex<double> sont inaccessible avec __device_
    while ( (z_real*z_real+ z_imag*z_imag) < 4.0f && i < this->getIterationMax()) // |z| < 2 et i < nb_iteration_max |||| zi² + z² < 4
    {
      double tempo = z_real;
      z_real = z_real*z_real - z_imag*z_imag + c.real();
      z_imag =  2*z_imag *tempo +c.imag() ;
      ++i;
    }

  if ( i != this->getIterationMax())
  {
    int color = i * 5;
    if (color >= 256) color = 0;
    b[idx] = color;//i * 255 / e->getIterationMax();
  }
  else
  {
    b[idx] = 0;
  }

  }
};
