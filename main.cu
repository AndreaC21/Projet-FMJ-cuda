#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <SFML/Graphics.hpp>
#include <vector>
#include <regex>

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

vector<string> split(const string& input, const string& regex) {
    std::regex re(regex);
    std::sregex_token_iterator
        first{input.begin(), input.end(), re, -1},
        last;
    return {first, last};
}

fstream f;

template <typename T>
void lectureFichier(int& w, int&h, vector<T>& v, int it_max)
{
  vector<T> result;
  string line;
  string delimiter ="=";

  getline (f,line);
  w = stoi(split(line,delimiter)[1]);
  getline (f,line);
  h = stoi(split(line,delimiter)[1]);

  int id_frame,i;
  float x,y,zoom;
  zoom = 3;
  i = 0;
  while ( getline (f,line) )
    {
      string s(line);
      if( !s.empty() )
      {
        if (i==0) { id_frame = stoi(split(s,delimiter)[1]); ++i;}
        else if (i==1) { x = stof(split(s,delimiter)[1]); ++i; }
        else if (i==2) { y = stof(split(s,delimiter)[1]); ++i; }
      }
      else
      {
        i = 0;
        v.push_back(T(x,y,it_max,zoom,id_frame));
        zoom-=0.2f;
      }
    }
    if ( f.eof() )
    {
      v.push_back(T(x,y,it_max,zoom,id_frame));
    }
}
int main(void)
{
  //const int width = 1024;
  //const int height = 1024;
  const int iteration_max = 1000;
  const string file_m ="config_mandelbrot.txt";
  const string file_j ="config_julia.txt";
  int width = 0;
  int height = 0;

  // lecture fichier config de Mandelbrot
  f.open(file_m);
	if ( !f.good()) throw runtime_error("impossible ouvrir fichier config");
  vector<Mandelbrot> list_mandelbrot{};
  lectureFichier<Mandelbrot>(width,height,list_mandelbrot,iteration_max);
  f.close();

  // lectur du fichier config de Julia
  f.open(file_j);
  if ( !f.good()) throw runtime_error("impossible ouvrir fichier config");
  vector<Julia> list_julia;
  lectureFichier<Julia>(width,height,list_julia,iteration_max);
  f.close();

  sf::Uint8 *pixels,*d_pixels;
  int *b,*d_b;

  pixels = new sf::Uint8[width*height*4];
  b = new int[width * height];

  cudaMalloc(&d_pixels,sizeof(sf::Uint8) * width * height);
  cudaMalloc(&d_b,sizeof(int) * width * height);

  dim3 bloc(16, 16);
  dim3 grille(width / bloc.x, height / bloc.y);


  // JULIA //
  for ( int i = 0 ; i < list_julia.size() ; ++i)
  {
    lance_calcul<<<grille,bloc>>>(list_julia[i],d_pixels,width,height,d_b);
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    //cudaMemcpy(pixels,d_pixels,sizeof(sf::Uint8) * width * height,cudaMemcpyDeviceToHost);
    cudaMemcpy(b,d_b,sizeof(int) * width * height,cudaMemcpyDeviceToHost);

    savePicture(&list_julia[i],width,height,pixels,b);
  }
  // creation video
  system("ffmpeg -y -r 1 -i Resultat/Julia%d.png julia.mp4");

  // MANDELBROT //
  for ( int i = 0 ; i < list_mandelbrot.size() ; ++i)
  {
    lance_calcul<<<grille,bloc>>>(list_mandelbrot[i],d_pixels,width,height,d_b);
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    cudaMemcpy(b,d_b,sizeof(int) * width * height,cudaMemcpyDeviceToHost);
    savePicture(&list_mandelbrot[i],width,height,pixels,b);
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
