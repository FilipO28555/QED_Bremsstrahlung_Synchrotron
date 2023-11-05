#define OLC_PGE_APPLICATION
#define _USE_MATH_DEFINES
#include "olcPixelGameEngine.h"
#include "LittleHelp_no_olc.h"

//#include <thread> 
//#include <atomic>
//std::atomic<bool> ThreadContinous = true;
//std::thread thread;

const int SIZEX = 1200;
const int SIZEY = 500;
const int MAXSIZE = SIZEX > SIZEY ? SIZEX : SIZEY;

using olc::vd2d;
using std::vector;


class vd3d {
public:
	double x;
	double y;
	double z;

	vd3d() {
		x = 0;
		y = 0;
		z = 0;
	}
	vd3d(double X, double Y, double Z) {
		x = X;
		y = Y;
		z = Z;
	}
	vd3d  operator -  (const vd3d& rhs) const { return vd3d(this->x - rhs.x, this->y - rhs.y, this->z - rhs.z); }
	vd3d  operator +  (const vd3d& rhs) const { return vd3d(this->x + rhs.x, this->y + rhs.y, this->z + rhs.z); }
	vd3d& operator -= (const vd3d& rhs) { this->x -= rhs.x; this->y -= rhs.y; this->z -= rhs.z; return *this; }
	vd3d& operator += (const vd3d& rhs) { this->x += rhs.x; this->y += rhs.y; this->z += rhs.z; return *this; }
	vd3d& operator *= (const double& rhs) { this->x *= rhs; this->y *= rhs; this->z *= rhs; return *this; }
	vd3d  operator *  (const double& rhs) const { return vd3d(this->x * rhs, this->y * rhs, this->z * rhs); }
	vd3d  operator /  (const double& rhs) const { return vd3d(this->x / rhs, this->y / rhs, this->z / rhs); }

	const std::string str() const { return std::string("(") + std::to_string(this->x) + "," + std::to_string(this->y) + "," + std::to_string(this->z) + ")"; }
	friend std::ostream& operator << (std::ostream& os, const vd3d& rhs) { os << rhs.str(); return os; }

	double  dot(const vd3d& rhs) const { return this->x * rhs.x + this->y * rhs.y + this->z * rhs.z; }
	vd3d cross(const vd3d& rhs) {
		return vd3d(this->y * rhs.z - this->z * rhs.y, this->z * rhs.x - this->x * rhs.z, this->x * rhs.y - this->y * rhs.x);
	}

	double mag2() {
		return x * x + y * y + z * z;
	}

	double mag() {
		return sqrt(mag2());
	}

	vd3d norm() {
		double magnitude = this->mag();
		return vd3d(this->x / magnitude, this->y / magnitude, this->z / magnitude);
	}

	vd3d rotate(vd3d axes, double angle) {
		axes = axes.norm();
		double out_x = x * (cos(angle) + axes.x * axes.x * (1 - cos(angle))) +
			y * (axes.x * axes.y * (1 - cos(angle)) - axes.z * sin(angle)) +
			z * (axes.x * axes.z * (1 - cos(angle)) + axes.y * sin(angle));
		double out_y = y * (cos(angle) + axes.y * axes.y * (1 - cos(angle))) +
			x * (axes.x * axes.y * (1 - cos(angle)) + axes.z * sin(angle)) +
			z * (axes.y * axes.z * (1 - cos(angle)) - axes.x * sin(angle));
		double out_z = z * (cos(angle) + axes.z * axes.z * (1 - cos(angle))) +
			y * (axes.z * axes.y * (1 - cos(angle)) + axes.x * sin(angle)) +
			x * (axes.x * axes.z * (1 - cos(angle)) - axes.y * sin(angle));
		return vd3d(out_x, out_y, out_z);
	}
};

double c = 200; //[px/s]
double E_mag = 1;
double stddevTheta = 0.7; //standard deviation of angle distribution [rad]
double velToFreq = 50; // 1px/s -> 100Hz. electron moveing at 1px/s will most likely emit photon with freq 100Hz
double stddev = 1; //standard deviation of freq distribution
double h = 0.01; //Planck constant [J*s] = [kg*px^2/s]
double m_e = 1; //electron mass [kg]

vd3d E(vd3d pos) {
	return vd3d(E_mag, 0, 0);
}

double scatter_prop(vd3d E, vd3d p) {
	return E.dot(p*-1);
}

//propability distribution of theta [0,pi]. the distribution is rotationally symettric
double propTheta(double theta) {
	return exp(-(theta * theta / stddevTheta)) / sqrt(M_PI); //normalized to 1
}
const double maxPropTheta = 0.57; //max propability of emitting photon

//propability density function of frequency
double propFreq(double freq,double vel) {
	double x = (freq - vel * velToFreq);// (x-mean)
	return exp(-(x*x/stddev)) / sqrt(M_PI); //normalized to 1
}
const double maxPropfreq = 0.57; //max propability of emitting photon

//propability density function of frequency for a synchrotron radiation?
double propFreqB(double freq, double vel, double B, double theta) {
	double meanFreq = velToFreq / (1 - (vel * vel) / (c * c)) * B * sin(theta);

	double x = (freq - meanFreq);// (x-mean)
	return exp(-(x * x / stddev)) / sqrt(M_PI); //normalized to 1
}
const double maxPropfreqB = 0.57; //max propability of emitting photon

struct photon;
vector<photon> photons;
struct particle;
vector<particle> particles;

//square detector
struct detector {
	int res;
	vector<vd3d> pos;
	vector<double> energy;
	double cellSize;
	
	//create a cetector with resolution x resolution cells
	detector(vd3d begin, vd3d span, int resolution) {
		res = resolution;
		cellSize = span.mag() / res;

		vd3d dirPerp = vd3d(0, 0, 1);
		vd3d dir = span.norm();
		for (int i = 0; i < res; i++) {
			for (int j = 0; j < res; j++) {
				pos.push_back(begin + dir * (i * cellSize) + dirPerp * ((j-res/2) * cellSize));
				energy.push_back(0);
			}
		}
	}



	bool operator()(photon y);
};

//on far right side of the screen
detector det(vd3d(SIZEX/2, 0, 0), vd3d(0, SIZEY, 0), 40);

struct photon
{
	vd3d pos;
	vd3d vel;
	double freq;

	photon(vd3d Pos, vd3d Vel, double Freq) {
		pos = Pos;
		vel = Vel.norm() * c; //normalize vel and set speed to c
		freq = Freq;
	}

	void update(double dt);


};



bool detector::operator()(photon y) {
	//check if photon hit detector
	for (int i = 0; i < this->pos.size(); i++) {
		if ((this->pos[i] - y.pos).mag() < cellSize/2) {
			energy[i] += y.freq * h;
			return true;
		}
	}
	return false;
}


void photon::update(double dt) {
	//update pos
	pos += vel * dt;
	//check if hit detector
	if (det(*this)) {
		//delete photon
		//photons.erase(photons.begin() + (&*this - &photons[0]));
		freq = 0;
	}
}


struct particle {
	vd3d pos;
	vd3d vel;
	double mass = m_e;
	double charge = -100;

	particle(vd3d Pos, vd3d Vel) {
		pos = Pos;
		vel = Vel;
	}

	void update(double dt) {
		
		//1) check if scatter
		double rand = MTrand(mt);
		//print(scatter_prop(E(pos), vel));
		if (rand < scatter_prop(E(pos), vel*charge)) {
			//scatter
			//1) choose new direction
			double theta = RandomFromDist(propTheta, -M_PI, M_PI, maxPropTheta);
			double phi = MTrand(mt) * M_PI;
			vd3d dir = vd3d(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
			dir = dir.rotate(vd3d(0,1,0), M_PI / 2); //rotate dir by 90 degrees around y
			//std::cout << dir << "\n";
			//dir = vel.norm()+dir; // hmmmm?????
			//dir = dir.norm();
			//1.2) choose new freq
			double magVel = (vel).mag();
			auto f = [magVel](double freq) {return propFreq(freq, magVel); };
			double mean = vel.mag() * velToFreq;
			double freq = RandomFromDist(f, mean - 3 * velToFreq, mean + 3 * velToFreq, maxPropfreq);
			//Create a photon
			photons.push_back(photon(pos, dir, freq));
			//calculate momentum of a photon 
			vd3d p = dir * h * freq/c; //p = h/l, c = l*f -> l = c/f -> p = h/(c/f) = h*f/c = E/c
			//2) update vel -> p = mv -> v = p/m, p_e = p_e - p_photon, v_e = (v_e*m_e - p_photon) / m_e
			vel = (vel*mass - p) / mass;
		}


		//Calculate the Lorentz force
		//vd3d F = E(pos) * charge;
		//vel += F / mass * dt;
		//4) update pos
		pos += vel * dt;

	}

};


class Example : public olc::PixelGameEngine
{
public:
	Example()
	{
		sAppName = "";

	}
public:
	void PlotFun(const std::function <double(double)>& f, double xMin, double xMax, double yMin, double yMax, olc::Pixel colour) {
		//Draw Axes
		double y0 = map(0, yMin, yMax, SIZEY, 0);
		double x0 = map(0, xMin, xMax, 0, SIZEX);
		DrawLine(x0, 0, x0, SIZEY, olc::WHITE);
		DrawLine(x0 + 1, 0, x0 + 1, SIZEY, olc::WHITE);
		DrawLine(0, y0, SIZEX, y0, olc::WHITE);
		DrawLine(0, y0 + 1, SIZEX, y0 + 1, olc::WHITE);
		//Draw Ticks
		for (int x = xMin; x < xMax; x += 1) {
			double y0 = map(0, yMin, yMax, SIZEY, 0);
			double x0 = map(x, xMin, xMax, 0, SIZEX);
			//if (x == 1 || )
			DrawString(x0 - 5, y0 + 15 * (1 - 2 * (int(x - xMin) % 2)), std::to_string(x), olc::WHITE, 1);
			DrawLine(x0, y0 - 5, x0, y0 + 5, olc::WHITE);
			DrawLine(x0 + 1, y0 - 5, x0 + 1, y0 + 5, olc::WHITE);
		}
		for (int y = yMin; y < yMax; y += 1) {
			double y0 = map(y, yMin, yMax, SIZEY, 0);
			double x0 = map(0, xMin, xMax, 0, SIZEX);
			DrawLine(x0 - 5, y0, x0 + 5, y0, olc::WHITE);
			DrawLine(x0 - 5, y0 + 1, x0 + 5, y0 + 1, olc::WHITE);
		}

		double dx = (xMax - xMin) / SIZEX;
		double px = f(xMin); //previous x
		double cx = 0; //courrent x
		for (int x = 1; x < SIZEX; x++) {
			cx = f(xMin + x * dx);
			DrawLine(x - 1, map(px, yMin, yMax, SIZEY, 0), x, map(cx, yMin, yMax, SIZEY, 0), colour);
			px = cx;
		}
	}
	void PlotAxes(int posx, int posy, int dx, int dy, double xMin, double xMax, double yMin, double yMax, int NoTickX, int NoTickY, olc::Pixel colour) {
		double tickX = (xMax - xMin) / NoTickX;
		double tickY = (yMax - yMin) / NoTickY;


		//Draw Axes
		double y0 = map(0, yMin, yMax, dy + posy, posy);
		double x0 = map(0, xMin, xMax, posx, posx + dx);
		DrawLine(x0, posy, x0, posy + dy, olc::WHITE);
		DrawLine(x0 + 1, posy, x0 + 1, posy + dy, olc::WHITE);
		DrawLine(posx, y0, posx + dx, y0, olc::WHITE);
		DrawLine(posx, y0 + 1, posx + dx, y0 + 1, olc::WHITE);
		//Draw Ticks
		for (double x = xMin; x < xMax; x += tickX) {
			double y0 = map(0, yMin, yMax, posy + dy, posy);
			double x0 = map(x, xMin, xMax, posx, posx + dx);
			DrawString(x0 - 5, y0 + 15 * (1 - 2 * (int(x - xMin) % 2)), std::to_string(x).substr(0, 6), olc::WHITE, 1);
			DrawLine(x0, y0 - 5, x0, y0 + 5, olc::WHITE);
			DrawLine(x0 + 1, y0 - 5, x0 + 1, y0 + 5, olc::WHITE);
		}
		for (double y = yMin; y < yMax; y += tickY) {
			double y0 = map(y, yMin, yMax, posy + dy, posy);
			double x0 = map(0, xMin, xMax, posx, posx + dx);
			DrawString(x0 + 15, y0, std::to_string(y).substr(0, 6), olc::WHITE, 1);
			DrawLine(x0 - 5, y0, x0 + 5, y0, olc::WHITE);
			DrawLine(x0 - 5, y0 + 1, x0 + 5, y0 + 1, olc::WHITE);
		}
	}
	void PlotGrid(double xMin, double xMax, double yMin, double yMax, double tickX, double tickY, olc::Pixel colour) {
		//Draw Axes
		double y0 = map(0, yMin, yMax, SIZEY, 0);
		double x0 = map(0, xMin, xMax, 0, SIZEX);
		DrawLine(x0, 0, x0, SIZEY, colour);
		DrawLine(x0 + 1, 0, x0 + 1, SIZEY, colour);
		DrawLine(0, y0, SIZEX, y0, colour);
		DrawLine(0, y0 + 1, SIZEX, y0 + 1, colour);
		//Draw Ticks
		for (double x = xMin; x < xMax; x += tickX) {
			double y0 = map(0, yMin, yMax, SIZEY, 0);
			double x0 = map(x, xMin, xMax, 0, SIZEX);
			if (xMin < -tickX)
				DrawString(x0 - 5, y0 + 15 * (1 - 2 * (int(x / tickX - xMin) % 2)), std::to_string(x).substr(0.6), colour, 1);
			else
				DrawString(x0 - 5, y0 - 15, std::to_string(x).substr(0.6), colour, 1);
			DrawLine(x0, SIZEY, x0, 0, colour);
			//DrawLine(x0 + 1, y0 - 5, x0 + 1, y0 + 5, olc::WHITE);
		}
		for (double y = yMin; y < yMax; y += tickY) {
			double y0 = map(y, yMin, yMax, SIZEY, 0);
			double x0 = map(0, xMin, xMax, 0, SIZEX);
			if (yMin < -tickY)
				DrawString(x0 + 15 * (1 - 2 * (int(y / tickY - yMin) % 2)), y0 + 5, std::to_string(y).substr(0, 6), colour, 1);
			else
				DrawString(x0 + 15, y0 + 5, std::to_string(y).substr(0, 6), colour, 1);
			DrawLine(0, y0, SIZEX, y0, colour);
			//DrawLine(x0 - 5, y0 + 1, x0 + 5, y0 + 1, olc::WHITE);
		}
	}
	void PlotVec(std::vector<double> vec, int x0, int x1, int y0, int y1, double xstart, double xend, double ymin, double ymax, olc::Pixel colour) {
		int N = vec.size();
		//double dx = (x1 - x0) / (N-1);
		PlotAxes(x0, y1, x1 - x0, y0 - y1, xstart, xend, ymin, ymax, 2, 5, olc::WHITE);
		int x = x0;
		int y = map(vec[0], ymin, ymax, y0, y1);
		int px = x;
		int py = y;
		for (int i = 1; i < N; i++) {
			x = map(i, 0, N - 1, x0, x1);
			y = map(vec[i], ymin, ymax, y0, y1);
			DrawLine(px, enclose(py, SIZEY), x, enclose(y, SIZEY), colour);
			px = x;
			py = y;
		}

	}
	void PlotFun(std::function <double(double)> f, int x0, int x1, int y0, int y1, double xstart, double xend, double ymin, double ymax, olc::Pixel colour) {
		std::vector<double> vec;
		int N = (x1 - x0);

		double dx = 1.0 * (xend - xstart) / (N + 1);
		for (int i = 0; i < N; i++)
			vec.push_back(f(xstart + i * dx));

		//PlotAxes(0, 0, SIZEX, SIZEY, xstart, xend, ymin, ymax, 10, 5, olc::WHITE);
		int x = x0;
		int y = map(vec[0], ymin, ymax, y0, y1);
		int px = x;
		int py = y;
		for (int i = 1; i < N; i++) {
			x = x0 + i;
			y = map(vec[i], ymin, ymax, y0, y1);
			DrawLine(px, py, x, y, colour);
			px = x;
			py = y;
		}

	}



	bool OnUserCreate() override
	{
		SetPixelMode(olc::Pixel::ALPHA);
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override {
		Clear(olc::BLACK);
		//Draw electrons
		for (auto& p : particles) {
			FillCircle(map(p.pos.x, 0, SIZEX, 0, SIZEX), map(p.pos.y, 0, SIZEY, 0, SIZEY), 2, olc::BLUE);
			//update
			p.update(fElapsedTime);
		}
		//Draw photons
		for (auto& p : photons) {
			vd3d pos = p.pos;
			p.update(fElapsedTime);
			//the more energy the photon has the more red it is
			olc::Pixel col = olc::Pixel((255/stddev) * p.freq / velToFreq, 255 - (255 / stddev) * p.freq / velToFreq, 0);
			DrawLine(map(pos.x, 0, SIZEX, 0, SIZEX), map(pos.y, 0, SIZEY, 0, SIZEY), map(p.pos.x, 0, SIZEX, 0, SIZEX), map(p.pos.y, 0, SIZEY, 0, SIZEY), col);
		}
		//Draw detector
		for (int i = 0; i < det.pos.size(); i++) {
			DrawRect(map(det.pos[i].x-det.cellSize/2, 0, SIZEX, 0, SIZEX), map(det.pos[i].y-det.cellSize/2, 0, SIZEY, 0, SIZEY), det.cellSize, det.cellSize, olc::WHITE);
		}
		//Draw 2d heatmap of energy. the plot coordinates: (SIZEX,SIZEX+SIZEY), (0,SIZEY)
		for (int i = 0; i < det.pos.size(); i++) {
			olc::Pixel col = olc::Pixel(255 * det.energy[i]/Max(det.energy), 0, 0);
			//print(255 * det.energy[i] / Max(det.energy));
			double dx = SIZEY * 1. / det.res;
			int x = i % det.res;
			int y = i / det.res;
			FillRect(SIZEX + x * dx, y * dx, dx, dx, col);
			//print(x);/*
			//print(y);
			//print(SIZEX + x * dx);
			//print(y * dx, "\n\n");*/
			

		}


		if (GetKey(olc::Key::R).bPressed) {
			int parNo = 1;
			//Clear vectors 
			particles.clear();
			photons.clear();
			//Clear detector
			for (int i = 0; i < det.pos.size(); i++) {
				det.energy[i] = 0;
			}
			//Create new particles
			for (int i = 0; i < parNo; i++) {
				particles.push_back(particle(vd3d(SIZEX / 7, SIZEY / 2 + (i - parNo / 2) * parNo / 2, 0), vd3d(50, 0, 0)));
			}
		}
		if (GetKey(olc::Key::N).bPressed) {
			int parNo = 10;
			//Create new particles
			for (int i = 0; i < parNo; i++) {
				particles.push_back(particle(vd3d(SIZEX / 7, SIZEY / 2 + (i - parNo / 2) * parNo / 2, 0), vd3d(50, 0, 0)));
			}
		}


		return true;
	}
};

int main(int argc, char* argv[])
{
	Example demo;
	if (demo.Construct(SIZEX+SIZEY, SIZEY, 1, 1, false))
		demo.Start();
	return 0;


}