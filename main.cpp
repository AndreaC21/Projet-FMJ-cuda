#include <iostream>
#include <complex>
#include <cmath>
#include <SFML/Graphics.hpp>

using namespace std;

class Ensemble
{

protected:
  float x1,x2,y1,y2;
  complex<double> c,z;
  int rang; // nombre iteration maximum
  int zoom;
  float image_x,image_y;
  sf::Image image;
  string nomImage;

public:
  Ensemble(float x1, float x2, float y1, float y2, int r, int z)
  {
    this->x1 = x1;
    this->x2 = x2;
    this->y1 = y1;
    this->x2 = x2;
    this->rang = r;
    this->zoom = z;
    this->image_x = (x2 - x1) * zoom;
    this->image_y = (y2 - y1) * zoom;
    this->image.create(image_x, image_y, sf::Color::Black); //On crée une image vide toute noire
  }
  ~Ensemble()
  {

  }
  virtual complex<double> init_c(int x, int y){return complex<double>(0,0);}
  virtual complex<double> init_z(int x, int y){return complex<double>(0,0);}
  virtual void drawPixel(int i, int x, int y){}

  int getIterationMax() const
  {
    return this->rang;
  }
  sf::Image getImage() const
  {
    return this->image;
  }
  string getNom() const
  {
    return this->nomImage+".png";
  }

  void equation()
  {
    for ( int x = 0; x < this->image_x; ++x)
    {
      for ( int y = 0; y < this->image_y; ++y)
      {
        complex<double> c = this->init_c(x,y);
        complex<double> z = this->init_z(x,y);
        float i = 0;

        while (sqrt(pow(z.real(),2)+ pow(z.imag(),2)) < 2 && i < this->rang) // |z| < 2 et i < nb_iteration_max
        {
          z = pow(z,2) + c;
          ++i;
        }

        this->drawPixel(i,x,y);
      }
    }
    this->image.saveToFile(this->nomImage+".png");
  }

};

class Mandelbrot : public Ensemble
{
public:
  // Mandelbrot est toujours compris entre -2.1 et 0.6 sur l'axe des abscisse et entre -1.2 et 1.2 sur l'axe des ordonnées.

  // x1 = -0.75 y2 = 0
  Mandelbrot(int iteration_max,int zoom):Ensemble(-2.1f,0.6f,-1.2f,1.2f,iteration_max,zoom)
  {
    this->nomImage = "Mandelbrot";
    this->nomImage.append("-it_"+to_string(iteration_max));
    this->nomImage.append("-zoom_"+to_string(zoom));
  }

  virtual complex<double> init_c(int x, int y)
  {
    //c = x + y
    return complex<double>(x/static_cast<double>(this->zoom)+this->x1,y/static_cast<double>(this->zoom)+this->y1);
  }
  virtual complex<double> init_z(int x, int y)
  {
    // z0 = 0
    return complex<double>(0,0);
  }

  virtual void drawPixel(int i,int x, int y)
  {
    if (i!=this->rang)
    {
      sf::Color color(0, 0, 0);
      color.b = i*255/this->rang;
      this->image.setPixel(x,y,color);
    }
  }
};
class Julia : public Ensemble
{
public:
  Julia(int iteration_max,int zoom):Ensemble(-1,1,-1.2f,1.2f,iteration_max,zoom)
  {
    this->nomImage = "Julia";
  }

  virtual complex<double> init_c(int x, int y)
  {
    // constante c
    return complex<double>(0.285,0.01);
  }
  virtual complex<double> init_z(int x, int y)
  {
    // z = x + y
    return complex<double>(x/static_cast<double>(this->zoom)+this->x1,y/static_cast<double>(this->zoom)+this->y1);
  }
  virtual void drawPixel(int i,int x, int y)
  {
    if (i==this->rang)
    {
      sf::Color color(0, 0, 0);
      color.b = i*255/this->rang;
      this->image.setPixel(x,y,color);
    }
    else
    {
        //mandelbrot.image.setPixel(x, y, sf::Color::White);
    }
  }
};

int main(void)
{

  Mandelbrot m(150,300);
  m.equation();
  Julia j(50,100);
  j.equation();
  sf::Image icon;
  icon.loadFromFile("icone.png");


  sf::RenderWindow window(sf::VideoMode(900, 700), "Projet FMJ");
  window.setFramerateLimit(60);
  window.setIcon(256,256,icon.getPixelsPtr());
  sf::View view(sf::Vector2f(m.getImage().getSize().x/2,m.getImage().getSize().y/2), sf::Vector2f(900/2,700/2));
  window.setView(view);

  sf::Sprite sprite_afficher;
  sf::Texture texture_afficher;
  texture_afficher.loadFromFile(m.getNom());
  sprite_afficher.setTexture(texture_afficher);

  while (window.isOpen())
  {
      sf::Event event;

      //sf::Time currentTime = f_clock.getElapsedTime();
      while (window.pollEvent(event))
      {
        if (event.type == sf::Event::Closed)
            {
                window.close();
            }
      else if(event.type==sf::Event::KeyPressed) //Déplacement
            {
              switch (event.key.code)
              {
                  // zoom avant
                  case sf::Keyboard::Numpad8:
                    view.zoom(0.7);
                    break;
                  //zoom arriere
                  case sf::Keyboard::Numpad2:
                    view.zoom(1.3);
                    break;
                  case sf::Keyboard::Left:
                    view.move(-10,0);
                    break;
                  case sf::Keyboard::Right:
                    view.move(10,0);
                    break;
                  case sf::Keyboard::Up:
                    view.move(0,10);
                    break;
                  case sf::Keyboard::Down:
                    view.move(0,-10);
                    break;
                  }
          }
      }
      window.clear();
      window.setView(view);
      window.draw(sprite_afficher); //On l'affiche
      window.display();
  }
  return 0;

}
