#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <SFML/Graphics.hpp>

using namespace std;

class Ensemble
{

protected:
  float x,y,zoom;
  int rang,id_image; // nombre iteration maximum
  sf::Image image;
  string nomImage;

public:
  Ensemble(){}
  Ensemble(float x,float y,int r, float zoom, int id_image)
  {
    this->x = x;
    this->y = y;
    this->rang = r;
    this->zoom = zoom;
    this->id_image = id_image;
  }
  Ensemble( Ensemble& e)
  {
    this->x = e.x;
    this->y = e.y;
    this->zoom = e.zoom;
    this->rang = e.rang;
    this->image = e.image;
    this->nomImage = e.nomImage;
    this->id_image = e.id_image;
  }
  ~Ensemble()
  {

  }
  __device__ virtual void calcul(sf::Uint8 *p,int w, int h,int *b) = 0;

  __device__ int getIterationMax() const  { return this->rang;}

  __device__  __host__ float getZoom() const  {  return this->zoom; }

  __host__ void saveImage(int w,int h, sf::Uint8 *pixels)
  {
    this->image.create(w,h,pixels);
    //this->nomImage.append("-size"+to_string(w)+"x"+to_string(w));
    this->image.saveToFile("Resultat/"+this->nomImage+".png");
  }

};

class Mandelbrot : public Ensemble
{
public:
  Mandelbrot() : Ensemble(-2.1f,-1.2f,1000,3,0)
  {
    this->nomImage = "Mandelbrot"+to_string(0);
    //this->nomImage.append("-it_"+to_string(1000));
    //this->nomImage.append("-zoom_"+to_string(3));
  }
  Mandelbrot(int iteration_max):Ensemble(-2.1f,-1.2f,iteration_max,3,0)
  {
    this->nomImage = "Mandelbrot"+to_string(0);
  }
  Mandelbrot(float x, float y,int iteration_max,float zoom,int id_image):Ensemble(x,y,iteration_max,zoom,id_image)
  {
    this->nomImage = "Mandelbrot"+to_string(id_image);
  }
  Mandelbrot(const Mandelbrot& e)
  {
    this->x = e.x;
    this->y = e.y;
    this->zoom = e.zoom;
    this->rang = e.rang;
    this->image = e.image;
    this->nomImage = e.nomImage;
  }

  __device__ void calcul(sf::Uint8 *p,int w, int h,int *b) override
  {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = w*row + col;

    complex<double> c( ((double)col / w) * this->zoom + this->x,((double)row / h) * this->zoom + this->y);

    complex<double> z(0,0);
    double z_real = z.real();
    double z_imag = z.imag();

    float i = 0;

    // On est oblige de decomposer les calcul pour les complex car les operator +,*,= de complex<double> sont inaccessible sur le GPU
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
  private :
    complex<double> c;
  public:

  Julia() : Ensemble(-1.5f,-1.5f,1000,3,0)
  {
    this->nomImage = "Julia"+to_string(0);
    c = complex<double>(0.285,0.01);
  //  this->nomImage.append("-it_"+to_string(1000));
    //this->nomImage.append("-zoom_"+to_string(3));
  }
  Julia(int iteration_max):Ensemble(-1.5f,-1.5f,iteration_max,3,0)
  {
    this->nomImage = "Julia"+to_string(0);
    c = complex<double>(0.285,0.01);
  }
  Julia(complex<double> c,int iteration_max,float zoom,int id_image) : Ensemble(-1.5f,-1.5f,iteration_max,3,id_image)
  {
    this->nomImage = "Julia"+to_string(id_image);
    this->c = c;
  }
  Julia(float cr, float ci,int iteration_max,float zoom,int id_image) : Ensemble(-1.5f,-1.5f,iteration_max,3,id_image)
  {
    this->nomImage = "Julia"+to_string(id_image);
    this->c = complex<double>(cr,ci);
  }

  Julia(const Julia& e)
  {
    this->x = e.x;
    this->y = e.y;
    this->zoom = e.zoom;
    this->rang = e.rang;
    this->image = e.image;
    this->nomImage = e.nomImage;
    this->id_image = e.id_image;
    this->c = e.c;
  }
  __device__ void calcul(sf::Uint8 *p,int w, int h,int *b) override
  {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = w*row + col;

    complex<double> c(this->c);

    complex<double> z( ((double)col / w) * this->zoom +this->x,((double)row / w) * this->zoom +this->y);
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
