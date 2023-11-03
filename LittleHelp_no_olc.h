#define M_PI       3.14159265358979323846   // pi

#include <iostream>
#include <iomanip>
#include <math.h>

#include <random>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <functional>
#include <numeric>

#include <set>
#include <map>
#include <array>
#include <vector>
#include <unordered_set>


// 2^-50
const double e2e50 = 8.8817841970012523233890533447266e-16;
// 2^-40
const double e2e40 = 9.094947017729282379150390625e-13;
// 2^-30
const double e2e30 = 0.000000000931322574615478515625;
// 2^-20
const double e2e20 = 0.00000095367431640625;
// 2^-15
const double e2e15 = 0.000030517578125;
// 2^-10
const double e2e10 = 0.0009765625;
// 2^-7
const double e2e7 = 0.0078125;
// 2^-5
const double e2e5 = 0.03125;
                        //Complex numbers
#include <complex>
// const double M_E = exp(1.);
//complex euler number
//std::complex<double> C_E = std::complex<double>(M_E,0);
//complex one
std::complex<double> I = std::complex<double>(1, 0);
//faster complex define
std::complex<double> FC(double re, double im) {
	return std::complex<double>(re, im);
}
//Complex real number
std::complex<double> C(double x) {
	return std::complex<double>(x, 0);
}
//complex imag number
std::complex<double> IC(double x) {
	return std::complex<double>(0, x);
}
//complex exponenitial
//std::complex<double> cExp(double x) {
//	return pow(C_E,std::complex<double>(0,x));
//}


#include<random>
std::mt19937 mt(time(0));
std::uniform_real_distribution<double> MTrand(0.0, 1.0);
std::uniform_real_distribution<double> MTrand101(-1.0, 1.0);

//Some of the functions
double Mag(std::vector<double> X);
void Mult(std::vector<double>& vec, double x);
double Min(std::vector<double> vec);
double Max(std::vector<double> vec);
std::vector<double> Add(std::vector<double> X1, std::vector<double> X2);

                                        //My normal functions
template <class T>
void print(T text, std::string end = "\n") {
    std::cout << std::to_string(text)<<end;
}
double map(double x, double in_min, double in_max, double out_min, double out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
// contains angle to (0, wrap)
double wrap(double angle, double wrap)
{
    return angle - wrap * floor(angle / wrap);
}
// modulo but with negative as well. (-wrap, +inf)
double Mod(int angle, int wrap)
{
    return (wrap + angle) % wrap;
}
// modulo with negative (-wrap, +inf) and with referance
void Modulo(int &angle, int wrap)
{
    angle = (wrap + angle) % wrap;
}
//sign function
int sign(int x) {
    return (x > 0) - (x < 0);
}

//Gratest Common Denominator
int GCD(int a, int b) {
    if (b == 0)
        return a;
    return GCD(b, a % b);
}

template <class T>
void SwitchIfNotOrdered(T &min, T &max) {
    if (min > max) {
        T temp = max;
        max = min;
        min = temp;
    }
}

//including min excluding max
bool isInside(double x,double min_x, double max_x) {
	if (x >= max_x || x < min_x)
		return false;
	return true;
}
//including min excluding max
bool isInside2d(double x, double y, double xmin, double dx, double ymin, double dy, bool excluding_max = true, bool include_min = true) {
    if (x >= xmin && x<xmin + dx && y>=ymin && y < ymin + dy)
        return true;
    return false;
}
bool isInside2dmax(double x, double y, double xmin, double xmax, double ymin, double ymax) {
    if (x > xmin && x<xmax && y>ymin && y < ymax)
        return true;
    return false;
}
bool isInside2dmaxInclude(double x, double y, double xmin, double xmax, double ymin, double ymax) {
    if (x >= xmin && x<=xmax && y>=ymin && y <= ymax)
        return true;
    return false;
}
bool isCircle(double R, double x, double y) {
	return (x * x + y * y <= R * R);
}
bool isParabola(double x,double y,double a) {
	return (y > a * x * x);
}
bool isBoundarySquare(int x, int y, int xmax,int ymax){
    return  x == xmax - 1 || y == ymax - 1;// || x == 0 || y == 0;
}
int enclose(int x, int max_x) {
	if (x >= max_x)
		return max_x-1;
	if (x < 0)
		return 0;
	return x;
}
double enclose(double x,double min_x, double max_x) {
    if (x > max_x)
        return max_x;
    if (x < min_x)
        return min_x;
    return x;
}
                                //Linear algebra
//Convention for matrixes (check for Gauss elimination):
//  A[y][x] =>    __                          __
//                | A[0][0] A[0][1] ... A[0][x] |
//                | A[1][0] A[1][1] ... A[1][x] |
//                | A[2][0] A[2][1] ... A[2][x] |
//                | .       .           .       |
//                | .       .           .       |
//                | .       .           .       |
//                |                             |
//                | A[y][0] A[y][1] ... A[y][x] |
//                --                          --
// 
//solveing tri-diagonal matrix
//d - diagonal, a-lower, b-upper, b-result vector: A*x=b. returns new wector x
std::vector<double> tri(std::vector<double> d, const std::vector<double>& a, const std::vector<double>& c, std::vector<double> b) {
    int n = d.size() - 1;
    if (n != a.size() || n != c.size() || n != b.size() - 1)
        throw "Wrong sizes!\n";
    for (int i = 1; i <= n; i++) {
        d[i] -= a[i - 1] / d[i - 1] * c[i - 1];
        b[i] -= a[i - 1] / d[i - 1] * b[i - 1];

    }

    std::vector<double> X(n + 1, (b[n] / d[n]));

    for (int i = n - 1; i >= 0; i--) {
        X[i] = (b[i] - c[i] * X[i + 1]) / d[i];
    }
    return X;

}

//Gauss Elimination for solveing A*x=B     
std::vector<double> GaussElimination(std::vector<std::vector<double>> A, std::vector<double> B) {
    int N = B.size();
    std::vector<double> x = std::vector<double>(N, 0);
    //Augment A
    int i = 0;
    for (auto& a : A) {
        a.push_back(B[i++]);
    }


    double temp;
    
    for (i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            if (abs(A[i][i]) < abs(A[j][i]))
            {
                for (int k = 0; k < N + 1; k++)
                {
                    // swapping mat[i][k] and mat[j][k] 
                    temp = A[i][k];
                    A[i][k] = A[j][k];
                    A[j][k] = temp;
                    
                }
            }
        }
    }
    

    //make above matrix upper triangular Matrix
    for (int j = 0; j < N - 1; j++)
    {
        for (i = j + 1; i < N; i++)
        {
            temp = A[i][j] / A[j][j];
            for (int k = 0; k < N + 1; k++)
                A[i][k] -= A[j][k] * temp;
        }
    }

    //find values of x,y,z using back substitution
    double s = 0;
    for (i = N - 1; i >= 0; i--)
    {
        s = 0;
        for (int j = i + 1; j < N; j++)
            s += A[i][j] * x[j];
        x[i] = (A[i][N] - s) / A[i][i];
    }

    return x;
}

//matrix multiplcation by a vector
std::vector<double> MatMul(std::vector<std::vector<double>> A, std::vector<double> B) {
    if (B.size() != A[0].size())
        throw "matrix and vector don't match\n";
    int x = A[0].size();
    int y = A.size();
    std::vector<double> result(y, 0);

    for (int i = 0; i < y; i++)
        for (int j = 0; j < x; j++)
            result[i] += A[i][j] * B[j];

    return result;
}

                                //Root finding
//Ridders method for finding root. (*add*)!! trying again
double Ridder(std::function<double(double)> f, double a, double b, double precision, int max_iter = 200, bool message = true, bool try_again = true){
    double x[3] = {a,(a+b)/2,b};
    double y[3] = {f(a),f((a+b)/2),f(b)};
    if(y[0]*y[2]>0){
        if(message)
            std::cout<<"Warning! f(a) and f(b) have the same sign! Returning middle point.\n"; 
        return x[1];
    }
    for(int i =0; i<3;i++){
        if(abs(y[i])<precision)
            return x[i];
    }
    bool sign;
    double X;
    double Y;
    unsigned int iter=0;
    
    while(abs(y[2])>precision && iter < max_iter){
        iter++;

        sign = y[0]>0;
        X = x[1] + (x[1]-x[0])*((-1+2*sign)*y[1]/sqrt(y[1]*y[1]-y[0]*y[2]));
        Y = f(X);
        if(Y*y[1]<0){
            x[0]=x[1];
            y[0]=y[1];
        }
        else if(Y*y[2]<0){
            x[0] = x[2];
            y[0] = y[2];
        }
        // else{
        //     x[0]=x[0];
        // }
        x[2] = X;
        y[2] = Y;

        x[1]=(x[0]+x[2])/2;
        y[1] = f(x[1]);     
        

    }
    //Best Y
    Y = ((abs(y[1]) < abs(y[2])) ? y[1] : y[2]);
    
     //std::cout<<"Iterations needed:"<<iter<<" second time? "<<(message?"false":"true")<<" Precision:"<<Y<<"\n"; 

     if(iter == max_iter && message)
         std::cout<<"Warning! Obtained precision is only: "<<Y<<"\n"; 




    return (abs(y[1])<abs(y[2]))?x[1]:x[2];
}
//Secant method for finding roots
double Secant(std::function<double(double)> f, double a, double b, double precision, int max_iter = 200, bool message = true, bool try_again = true){
    double x[2] = {a,b};
    double y[2] = {f(a),f(b)};
    if(y[0]*y[1]>0){
        if(message)
            std::cout<<"Warning! f(a) and f(b) have the same sign! Returning point a.\n"; 
        return x[0];
    }
    for(int i =0; i<2;i++){
        if(abs(y[i])<precision)
            return x[i];
    }
    unsigned int iter=0;
        bool i1 = (iter-1)%2;
        bool i2 = (iter)%2;
    while(abs(y[iter%2])>precision && iter < max_iter){
        iter++;  
        // std::cout<< iter<<" "<<x[iter%2]<<" "<<x[(iter+1)%2]<<"\n";
        // std::cout<<"    "<< y[iter%2]<<" "<<y[(iter+1)%2]<<"\n";
        
        x[i1] = (x[i1]*y[i2]-x[i2]*y[i1])/(y[i2]-y[i1]);
        // x[iter%2] = x[(iter+1)%2] - y[(iter+1)%2]*(x[(iter+1)%2]-x[(iter)%2])/(y[(iter+1)%2]-y[(iter)%2]);
        y[i1] = f(x[i1]);

        i1=!i1;
        i2=!i2;

    }
    //Best Y
    double Y=((abs(y[1])<abs(y[0]))?y[1]:y[0]);
    // std::cout<<iter<<"\n"; 
    if( iter==max_iter ){
        if(try_again){
            double new_A = a + (b-a)/200;
            double new_B = b - (b-a)/200;
            double X = Secant( f,  new_A,  new_B,  precision);
            double newY = abs(f(X));
            if(newY<Y){
                if(newY>precision && message)
                    std::cout<<"Warning! Obtained precision is only: "<<newY<<"\n"; 
                return X;
            }
        }
        if(message)
            std::cout<<"Warning! Obtained precision is only: "<<Y<<"\n"; 
        
    }



    return (abs(y[1])<abs(y[0]))?x[1]:x[0];
}


//simple bisection
double Bisection(std::function<double(double)> f, double a, double b, double precision){
    double X=a;
    double dx = (b-a)/2;
    bool dir = true;
    bool znak = (f(a)<0); //initial sign of a function
    double er = f(X);
    int iter =0;
    while(abs(er)>precision && iter < 1000){
        iter++;
        X += (-1+2*dir)*dx;
        dx/=2;
        er = (-1+2*znak)*f(X);
        if(er > 0){
            dir = !dir;
            znak = !znak;
        }   
    }
    // std::cout<<iter<<"\n"; 
    if(iter>999){
        std::cout<<"Warning! Obtained precision is only: "<<abs(er)<<"\n"; 
        }
    return X;
}
//Finds all roots in (a,b) separated by at least step using Ridders method. Clears and saves answear to vector results.
void FindAllRoots(std::function<double(double)> f, double a, double b, std::vector<double> &results, double step=0.015625, double precision= e2e40,bool clearVector = true){
    bool znak = (f(a)>0); //initial sign of a function
    if(clearVector)
        results.clear();
    for(double X=a;X<=b;X+=step){
        // std::cout<<X<<" "<<(-1+2*znak)<<" "<<f(X)<<"\n";
        if((-1+2*znak)*f(X)<0){
            results.push_back(Ridder(f,X-step,X,precision,5000));
            znak = !znak;
        }
    }
}

                                //Integration
double IntegrateQuadrature(std::function<double(double)> f, double a, double b, double N) {
	double h = (b - a) / (N-1);
	double sum = 0;
	for (int i=1; i < N; i++) {
		sum += h * ( f(a+(i-1)*h)+4*f(a+(i-1)*h+h/2)+f(a+(i)*h)  )/6;
	}
	return sum;
}
double IntegrateVec(std::vector<double> vec, double a, double b) {
    int N = vec.size();
    double h = (b - a) / (N - 1);
    double sum = 0;
    for (double v:vec) {
        sum += v;
    }
    return h * sum;
}

                                //Derivative
double df(std::function<double(double)> F,double x, double h) {
	return (F(x + h) - F(x - h)) / (2 * h);
    
}   
double df2(std::function<double(double)> F,double x, double h) {
	return (F(x + h) + F(x - h)-2*F(x)) / (h*h);
    
}                             

                                //Fixed point iteration
double FixedPoint(std::function<double(double)> f,double x0, int N){
	for (N; N > 0; N--) {
		x0 = f(x0);
	}
	return x0;

}

//Finds all maximum/minimum points of f in (a,b) separated by at least step.
void FindMinMax(std::function<double(double)> f,double a, double b, std::vector<double> &results, double step=0.015625, double precision = e2e10,bool clearVector = true){
    double h = precision;
    auto der = [f,h](double X){return (f(X + h) - f(X - h)) / (2 * h);};
    FindAllRoots(der,a,b,results,step,precision,clearVector);
}
//Finds all maximum points of f in (a,b) separated by at least step.
void FindMaxDepricated(std::function<double(double)> f,double a, double b, std::vector<double> &results, double step=0.015625, double precision = e2e10,bool clearVector = true){
    if(clearVector)
        results.clear();

    double h = precision;
    auto der = [f,h](double X){return (f(X + h) - f(X - h)) / (2 * h);};
    std::vector<double> minMax;
    FindAllRoots(der,a,b,minMax,step,precision,false);

    for(auto i : minMax){
        if(df(der,i,1e-6)<0){
            results.push_back(i);
        }
    }
}
//Finds one maximum in range (a,b) using golden ratio
double FindOneMax(std::function<double(double)> f, double a, double b) {
    double x1, x2, x11, x22, f1, f2;
    double xopt[2];
    double R = (sqrt(5) - 1) / 2;
    double d;

    x1 = a;
    x2 = b;

    d = R * (x2 - x1);

    x11 = x1 + d;
    f1 = f(x11);
    x22 = x2 - d;
    f2 = f(x22);
    xopt[0] = x11;
    xopt[1] = x22;

    int i = 0;
    int iter = 0;
    while (xopt[i]-xopt[1-i]) {
        iter++;
        d = R * d;
        //metoda
        if (f1 > f2) {
            x1 = x22;
            x22 = x11;
            //x11 = x1 + R * (x2 - x1); 
            x11 = x1 + d;
            f2 = f1;
            f1 = f(x11);
            xopt[i] = x11;
        }
        else {
            x2 = x11;
            x11 = x22;
            //x22 = x2 - R * (x2 - x1);
            x22 = x2 - d;
            f1 = f2;
            f2 = f(x22);
            xopt[i] = x22;
        }
        i = 1 - i;
    }
    std::cout<<iter<<"\n";
    return xopt[i];
}

//Finds one maximum in range (a,b) using parabolic approximation
double FindMaxMaxQuadratic(std::function<double(double)> f, double a, double b, double precision = e2e50) {
    double x1, x2, x11, x22, f1, f2;
    double xopt[2];
    double R = (sqrt(5) - 1) / 2;
    double d;

    x1 = a;
    x22 = b;

    d = R * (x22 - x1);

    x11 = x1 + d;    


    xopt[0] = x11;
    xopt[1] = x22;

    int i = 0;
    int iter = 0;
    while (abs(xopt[i] - xopt[1 - i])>precision) {
        iter++;
        double f0 = f(x1 );
        double f1 = f(x11);
        double f2 = f(x22);
        x2 = (f0 * (x11 * x11 - x22 * x22) + f1 * (x22 * x22 - x1 * x1) + f2 * (x1 * x1 - x11 * x11)) /
            (2 * f0 * (x11 - x22) + 2 * f1 * (x22 - x1) + 2 * f2 * (x1 - x11));
        xopt[i] = x2;
        if (x2 > x11) {
            x1 = x11;
            x11 = x2;
        }
        else {
            x22 = x11;
            x11 = x2;
        }
        i = 1 - i;
    }
    std::cout << iter << "\n";
    return xopt[i];
}
//Finds one maximum in range (a,b) using parabolic approximation and if not working than using golden ratio
double FindMaxMaxBrent(std::function<double(double)> f, double a, double b,int iterations =10,double precision = e2e30) {
    double x1, x2, x11, x22, f1, f2;
    double qx1, qx2, qx11, qx22, qf1, qf2;
    double xopt[2];
    double R = (sqrt(5) - 1) / 2;
    double d;

    x1 = a;
    x22 = b;

    d = R * (x22 - x1);

    x11 = x1 + d;


    xopt[0] = x11;
    xopt[1] = x22;

    int i = 0;
    int iter = 0;
    //while (iterations-- && abs(xopt[i] - xopt[1 - i])>precision) {
    while ( xopt[i] - xopt[1 - i] ) {
        iter++;
        double f0 = f(x1);
        double f1 = f(x11);
        double f2 = f(x22);
        x2 = (f0 * (x11 * x11 - x22 * x22) + f1 * (x22 * x22 - x1 * x1) + f2 * (x1 * x1 - x11 * x11)) /
            (2 * f0 * (x11 - x22) + 2 * f1 * (x22 - x1) + 2 * f2 * (x1 - x11));
        
        if (x2 > x22 || x2 < x1) {
            std::cout <<x1<<" - "<<x11<<" - "<<x22<<"   :   " << x2 << "\n";

            qx1 = x1;
            qx2 = x22;

            d = R * (qx2 - qx1);

            qx11 = qx1 + d;
            qf1 = f(qx11);
            qx22 = qx2 - d;
            qf2 = f(qx22);

            d = R * d;
            //metoda
            if (qf1 > qf2) {
                x1 = qx22;
                x11 = qx11;
                x22 = qx2;

                xopt[i] = qx11;
            }
            else {
                x22 = qx11;
                x11 = qx22;
                xopt[i] = x22;
            }
        }
        else if (x2 > x11) {
            xopt[i] = x2;
            x1 = x11;
            x11 = x2;
        }
        else {
            xopt[i] = x2;
            x22 = x11;
            x11 = x2;
        }
        i = 1 - i;
    }
    std::cout << iter << "\n";
    return xopt[i];
}


//Finds all maximum points of f in (a,b) separated by at least step.
void FindMax(std::function<double(double)> f, double a, double b, std::vector<double>& results, double step = 0.015625, double precision = e2e10, bool clearVector = true) {
    if (clearVector)
        results.clear();

    double h = precision;
    auto der = [f, h](double X) {return (f(X + h) - f(X - h)) / (2 * h); };
    std::vector<double> minMax;
    FindAllRoots(der, a, b, minMax, step, precision, false);

    for (auto i : minMax) {
        if (df(der, i, 1e-6) < 0) {
            results.push_back(i);
        }
    }
}
//Finds all minimums points of f in (a,b) separated by at least step.
void FindMin(std::function<double(double)> f,double a, double b, std::vector<double> &results, double step=0.015625, double precision = e2e10,bool clearVector = true){
    if(clearVector)
        results.clear();

    double h = precision;
    auto der = [f,h](double X){return (f(X + h) - f(X - h)) / (2 * h);};
    std::vector<double> minMax;
    FindAllRoots(der,a,b,minMax,step,precision,false);

    for(auto i : minMax){
        if(df(der,i,1e-3)>0){
            results.push_back(i);
        }
    }
}

//Finds position (x!) of the biggest maximum in value in range (a,b)
double FindMaxMax(std::function<double(double)> f,double a, double b, double step=e2e5, double precision = e2e30){
    double h = precision;
    auto der = [f,h](double X){return (f(X + h) - f(X - h)) / (2 * h);};
    std::vector<double> minMax;
    FindAllRoots(der,a,b,minMax,step,precision,false);
    if(minMax.size()>0){
        bool isMaxQ = false;
        double x = 0;
        double maxy = -INFINITY;
        for(auto i : minMax){
            if(df(der,i,1e-3)<0 && f(i)>maxy){
                isMaxQ=true;    
                x=i;
                maxy=f(i);
            }
        }
        if (isMaxQ && (f(x) > f(a) && f(x) > f(b)))
            return x ;
    }
    return (f(a) > f(b) ? a : b);

}
 //Finds position of the smallest minimum in value in range (a,b)
double FindMinMin(std::function<double(double)> f,double a, double b, double step=0.015625, double precision = e2e40){
    double h = precision;
    auto der = [f,h](double X){return (f(X + h) - f(X - h)) / (2 * h);};
    std::vector<double> minMax;
    FindAllRoots(der,a,b,minMax,step,precision,false);

    if(minMax.size()>0){
        bool isMaxQ = false;
        double x = 0;
        double maxy = INFINITY;
        for(auto i : minMax){
            if(df(der,i,1e-3)>0 &&f(i)<maxy){
                isMaxQ=true;    
                x=i;
                maxy=f(i);
            }
        }
        //std::cout << "foun minimum at: " << x << "\n";
        if (isMaxQ && (f(x) < f(a) && f(x) < f(b)))
            return x;
    }
    return (f(a) < f(b) ? a : b);
}
double FindMinMinValueFast(std::function<double(double)> f, double a, double b, double step = 0.015625) {
    int N = (b - a) / step;
    std::vector<double> Y(N+1,0);
    int i = 0;
    double x = a;
    while (x < b) {
        Y[i++] = f(x);
        x += step;
    }
    return Min(Y);
}
double FindMaxMaxValueFast(std::function<double(double)> f, double a, double b, double step = 0.015625) {
    int N = (b - a) / step;
    std::vector<double> Y(N+1,0);
    int i = 0;
    double x = a;
    while (x < b) {
        Y[i++] = f(x);
        x += step;
    }
    return Max(Y);
}
std::vector<double> FindMinMaxValueFast(std::function<double(double)> f, double a, double b, double step = 0.015625) {
    int N = (b - a) / step;
    std::vector<double> Y(N + 1, 0);
    int i = 0;
    double x = a;
    while (x < b) {
        Y[i++] = f(x);
        x += step;
    }
    return { Min(Y),Max(Y) };
}

//Find gradient of function of R^n->R returns R^n
std::vector<double> Gradient(std::function<double(std::vector<double>)> f, std::vector<double> x0,double h) {
    std::vector<double> grad;
    for (int i = 0; i < x0.size(); i++) {
        //function to diff
        auto f0 = [f, x0, i](double x) {
            std::vector<double> X = x0;
            X[i] = x;
            return f(X);
        };
        grad.push_back(df(f0, x0[i], h));
    }
    return grad;
}



//Find gradient of R^2->R function checking multiple surroundings returns alpha
double GradientMax(std::function<double(std::vector<double>)> f, std::vector<double> x0, double h, int directions = 8) {
    double alphaMax = 0;
    double maxGrad = f({ x0[0] + h,x0[1] });
    double da = 2 * M_PI / directions;
    for (double alpha = da; alpha < 2 * M_PI; alpha += da) {
        std::vector<double> xa = x0;
        xa[0] += h * cos(alpha);
        xa[1] += h * sin(alpha);

        double dfa = f(xa);
        //std::cout << xa[0] << " " << xa[1] << " " << dfa << " " << alpha << "\n";
        if (dfa > maxGrad) {
            alphaMax = alpha;
            maxGrad = dfa;
        }
    }
    return alphaMax;
}

//Find gradient of R^2->R function checking multiple surroundings returns alpha
double GradientMin(std::function<double(std::vector<double>)> f, std::vector<double> x0, double h, int directions = 8) {
    double alphaMax = MTrand(mt)*2*M_PI;
    double alphaEnd = 2 * M_PI + alphaMax;
    double minGrad = f({x0[0]+ h * cos(alphaMax),x0[1]+ h * sin(alphaMax) });
    double da = 2 * M_PI / directions;
    for (double alpha = alphaMax + da; alpha < alphaEnd; alpha += da) {
        std::vector<double> xa = x0;
        xa[0] += h * cos(alpha);
        xa[1] += h * sin(alpha);
        
        double dfa = f(xa);
        //std::cout << xa[0] << " " << xa[1] << " " << dfa << " " << alpha << "\n";
        if (dfa < minGrad) {
            alphaMax = alpha;
            minGrad = dfa;
        }
    }
    return alphaMax;
}


//NOT TESTED
//Find gradient of R^N->R function checking multiple surroundings returns normalized vector. if test_points = 0 -> 2^(dim+1)
std::vector<double> GradientMaxNd(std::function<double(std::vector<double>)> f, std::vector<double> x0, double h, int test_points = 0) {
    int dim = x0.size();
    if (test_points <= 0) test_points = pow(2, (dim + 1));


    double maxGrad = 0;
    double f0 = f(x0);
    std::vector<double> bestX;

    //first point to compare to
    std::vector<double> xa(dim);
    for (int i = 0; i < dim; i++) {
        xa.push_back(MTrand101(mt));
    }
    double mag = Mag(xa);
    Mult(xa, h/ mag); //multiply - normalize to h

    bestX = xa;
    maxGrad = f(Add(x0, xa)) - f0;



    while (--test_points) {
        //generate new vector
        for (int i = 0; i < dim; i++) {
            xa[i] = MTrand101(mt);
        }
        mag = Mag(xa);
        Mult(xa, h/ mag); //multiply - normalize to h

        double dfa = f(Add(x0, xa)) - f0; // check if better

        if (dfa > maxGrad) {
            bestX = xa;
            maxGrad = dfa;
        }
    }
    return bestX;
}

//Find gradient of R^N->R function checking multiple surroundings returns normalized vector. if test_points = 0 -> 2^(dim+1)
std::vector<double> GradientMinNd(std::function<double(std::vector<double>)> f, std::vector<double> x0, double h, int test_points = 0) {
    int dim = x0.size();
    if (test_points <= 0) test_points = pow(2, (dim + 1));


    double minGrad = 0;
    double f0 = f(x0);
    std::vector<double> bestX;

    //first point to compare to
    std::vector<double> xa(dim);
    for (int i = 0; i < dim; i++) {
        xa.push_back(MTrand101(mt));
    }
    double mag = Mag(xa);
    Mult(xa, h/ mag); //multiply - normalize to h
    
    bestX = xa;
    minGrad = f(Add(x0, xa)) - f0;
    

    while (--test_points) {
        //generate new vector
        for (int i = 0; i < dim; i++) {
            xa[i] = MTrand101(mt);
        }
        mag = Mag(xa);
        Mult(xa, h/mag); //multiply - normalize to h

        double dfa = f(Add(x0, xa)) - f0; // check if better

        if (dfa < minGrad) {
            bestX = xa;
            minGrad = dfa;
        }
    }
    return bestX;
}


                                //Diff Eq
//Numerov - solves d^2y/dx^2 = - g(x)y(x) given y0 and dy0
double Numerov(std::vector<double> &results, std::function<double(double)> g,double y0, double dy0, double a = 0, double b = 10, int N=513,bool clearVector=true){
    double h = (b - a) / (N - 1);
    int Voffset=0;
    if(clearVector)
        results.clear();

    std::vector<double> result;
    //initial conditions
    result.push_back(y0);
    result.push_back(y0 +dy0*h);

    for (int n = 1; n <  N; n++) {
			double i = a + h * (n + 1);
			result.push_back((2*result[n]*(1-5*(g(i))/12*h*h)-result[n-1]*(1+g(i-h)/12*h*h))/(1 + g(i+h)/12*h*h));
			// if(y[(n)]>yMin && y[(n)]<yMax)
			// 	DrawLine(map(n-1, 1, N - 1, 0, SIZEX), map(y[(n)], yMin, yMax, SIZEY, 0), map(n, 1, N - 1, 0, SIZEX), map(y[(n + 1)], yMin, yMax, SIZEY, 0), color);
		}

        // std::cout<<a + h * (N)+h;
    if(b<a){
        std::reverse(result.begin(),result.end());
    }
    for(auto a : result){
        results.push_back(a);
    }
    return abs(h);
}

//Variational - solves d^2y/dx^2 = -S(x) given y0 and y_end using iter iterations
double Variate(std::vector<double>& results, std::function<double(double)> S, double y0, double y_end, double a = 0, double b = 10, int N = 513, int iter = 512, bool clearVector = true) {
    double h = (b - a) / (N - 1);
    
    if (clearVector) {
        results.clear();
        //initial conditions
        for (int i = 0; i < N - 1; i++)
            results.push_back(map(i,0,N-1,y0,y_end));
        results.push_back(y_end);
    }

    for (int iteration = 0; iteration < iter; iteration++) {
        for (int i = 1; i < N-1; i++) {
            results[i] = (results[i + 1] + results[i - 1] + h * h * S(a + i * h)) / 2;
        }
    }


    return abs(h);
}

//Relaxation Method - solves d^2y/dx^2 = -S(x) given y0 and y_end and parameter 0<w<2 returns w
double Relax(std::vector<double>& results, std::function<double(double)> S, double y0, double y_end,double w, double a = 0, double b = 10, int N = 513, int iter = 512, bool clearVector = true) {
    double h = (b - a) / (N - 1);
    
    if (clearVector) {
        results.clear();
        //initial conditions
        for (int i = 0; i < N - 1; i++)
            results.push_back(y0);
        results.push_back(y_end);
    }

    for (int iteration = 0; iteration < iter; iteration++) {
        for (int i = 1; i < N - 1; i++) {
            results[i] = (1-w)* results[i]+w*(results[i + 1] + results[i - 1] + h * h * S(a + i * h))/2;
        }
    }


    return h;
}

                                //Integral equations
//Finite sum method - solves Fredholm equation y(x) = f(x) + lam*integrate(K(s,x),{ds,a,b}). returns vector of size N.
std::vector<double> FiniteSum( std::function<double(double)> f, std::function<double(double, double)> K, double lam, double a = 0, double b = 10,int N = 513) {
    //simson is only for odd
    if ((N % 2)==0)
        N++;

    double dx = (b - a) / (N - 1);

    std::vector<double> intCoeff(N);
    int j = 1;  
    for (auto& i : intCoeff) {
        i = ((4.-2*(j%2)) / 3.) * dx;
        j++;
        //i = dx;
    }
    intCoeff[0] = dx / 3;
    intCoeff[N-1] = dx / 3;

    std::vector<std::vector<double>> lambda(N, std::vector<double>(N));
    std::vector<double> F(N);
    for (int i = 0; i < N; i++) {
        F[i] = f(a + i * dx);
        for (int j = 0; j < N; j++)
            lambda[i][j] = ((i == j) - lam * intCoeff[j] * K(a + i * dx, a + j * dx));
    }

    return GaussElimination(lambda, F);
}

                                //Random Numbers

//Von Neumann rejection - returns number of random number it needed to generate
long RandomDist(double (*g)(double),double a, double b, std::vector<double> &results, int N){
    results.clear();
    double maxprop =  g(FindMaxMax(g, a, b)) + 0.1; //find maximum of function g
    double y =  MTrand(mt)* maxprop; //(0,1)
    double x = MTrand(mt)*(b-a)+a;
    int iter=0;
    while(N>0){
        iter++;
        if(y<g(x)){
            results.push_back(x);
            N--;
        }

        y = MTrand(mt)*maxprop; //(0,1)
        x = MTrand(mt)*(b-a)+a;
    }
    return iter;
    //std::cout<<iter<<"\n";
}
//returns one random number from distribution g
double RandomFromDist(std::function<double(double)> g, double a, double b, double maxprop = 0) {
    if(maxprop == 0)
        maxprop = g(FindMaxMax(g, a, b)) + 0.1; //find maximum of function g
    double y = MTrand(mt) * maxprop; //(0,1)
    double x = MTrand(mt) * (b - a) + a;
    while (y >= g(x)) {
        y = MTrand(mt) * maxprop; //(0,1)
        x = MTrand(mt) * (b - a) + a;
    }
    return x;

}
//Metropolis algorithm to get random distribution given distribution function
double Mean(const std::vector<double>& vec);
void RandomDistMetropolis(double (*g)(double), double a, double b, std::vector<double>& results, int N) {
    results.clear();
    double x = 0;// MTrand(mt)* (b - a) + a;
    double xt = MTrand(mt) * (b - a) + a;
    double dx = abs(xt - x);
    std::vector<double> delta; 
    
    int iter = 0;
    while (N > 0) {
        iter++;
        double r = g(xt) / g(x);
        if (r<1) {
            if (MTrand(mt) < r) {
                results.push_back(xt);
                N--;
                delta.push_back(dx);
            }
        }
        else {
            results.push_back(xt);
            delta.push_back(dx);
            N--;
        }

        x = xt;
        //std::cout << iter << "\n";
        xt += (MTrand(mt) * 4 - 2) * Mean(delta);
        while (!(xt > a && xt < b)) {
            //std::cout << iter << "\n";
            xt = x + (MTrand(mt) - 0.5) *Mean(delta);

        }
        dx = abs(xt - x);

    }
    //std::cout << iter << "\n";
}

//Make histogram returns no of data in Nth interval, at the end of vector it appends a,b and dx
void Histogram(std::vector<double> data, double a, double b, double N,std::vector<double> &hist) {
    hist.clear();
		for (int i = 0; i < N; i++)
			hist.push_back(0);  

		double dx = (b - a) / N;
		for (auto i : data) {
			if ((int((i-a) / dx)) < int(hist.size()))
				hist[int((i-a) / dx)]+=1;
			else
				std::cout <<"index error: "<<(int((i-a) / dx)-1)<<" < "<<hist.size()<< "\n";
		}
        hist.push_back(a);
        hist.push_back(b);
        hist.push_back(dx);
}
//Save Histogram to file
void SaveHist(std::vector<double> hist,std::string filename){
	std::ofstream Log = std::ofstream("Histogram.txt");
    // std::cout<<hist.size();
    double dx = hist[hist.size()-1];
    // std::cout<<dx;
    double a = hist[hist.size()-3];
    hist.pop_back();
    hist.pop_back();
    hist.pop_back();

    for(auto i : hist){
        Log<<a<<", "<<i<<"\n";
        a+=dx;
    }
    Log.close();
    
}


                                                                //Vector Operations
//Print vector                                                                // 
void PrintVector(std::vector<double> vec) {
    for (auto i : vec)
        std::cout << i << ", ";
    std::cout << "\n";
}
//Max of vector
double Max(std::vector<double> vec) {
    if (vec.size()==0)
        throw "Empty Vector";
    double max = vec[0];
    for (auto x : vec)
        if (x > max)
            max = x;
    return max;
}
//Max of 2D vector
double Max(std::vector<std::vector<double>> vec) {
    if (vec.size() == 0)
        throw "Empty Vector";
    double max = vec[0][0];
    for ( auto y : vec)
        for( auto x : y)
            if (x > max)
                max = x;
    return max;
}
//Min of vector
double Min(std::vector<double> vec) {
    if (vec.size() == 0)
        throw "Empty Vector";
    double max = vec[0];
    for (auto x : vec)
        if (x < max)
            max = x;
    return max;
}
//Sum of vector
double Sum(const std::vector<double>& vec) {
    double sum = 0;
    for (double i : vec)
        sum += i;
    return sum;
}
//Sum of 2d vector
double Sum(const std::vector<std::vector<double>>& vec) {
    double sum = 0;
    for (const auto& i : vec)
        sum += Sum(i);
    return sum;
}
//Mean of vector
double Mean(const std::vector<double>& vec) {
    return Sum(vec)/vec.size();
}
//Mean of abs values of vector
std::vector<double> Abs(std::vector<double> X1);
double AbsMean(std::vector<double> vec) {
    return Sum(Abs(vec)) / vec.size();
}

//Scalar product. R^n->R
double ScalarProduct(std::vector<double> X1, std::vector<double> X2) {
    double sum = 0;
    if (X1.size() == X2.size()) {
        int i = 0;
        for (double x : X1) {
            sum += x * X2[i];
            i++;
        }
    }
    else {
        throw "Vectors have different sizes.";
    }
    return sum;
}

//Subtract element-wise
std::vector<double> Subtract(std::vector<double> X1, std::vector<double> X2) {
    std::vector<double> Y;
    
    if (X1.size() == X2.size()) {
        int i = 0;
        for (double x : X1) {
            Y.push_back(x - X2[i]);
            i++;
        }
    }
    else {
        throw "While subtracting vectors have different sizes.";
    }
    return Y;
}
//Add element-wise
std::vector<double> Add(std::vector<double> X1, std::vector<double> X2) {
    std::vector<double> Y;

    if (X1.size() == X2.size()) {
        int i = 0;
        for (double x : X1) {
            Y.push_back(x + X2[i]);
            i++;
        }
    }
    else {
        throw "While adding: vectors have different sizes.";
    }
    return Y;
}

//Multiply element-wise
std::vector<double> Mult(const std::vector<double>& X1,const std::vector<double>& X2) {
    std::vector<double> Y;

    if (X1.size() == X2.size()) {
        int i = 0;
        for (double x : X1) {
            Y.push_back(x * X2[i]);
            i++;
        }
    }
    else {
        throw "While multiplying: vectors have different sizes.";
    }
    return Y;
}

//Add constant to a vector
std::vector<double> Add(std::vector<double> X1, double c) {
    std::vector<double> Y;
        for (double x : X1) {
            Y.push_back(x + c);
        }
    return Y;
}
//Add constant to a vector but the vector is of pointers
std::vector<double> Add(std::vector<double*> X1, double c) {
    std::vector<double> Y;
    for (double *x : X1) {
        Y.push_back(*x + c);
    }
    return Y;
}


//Absolute value element-wise of a vector but the vector is of pointers
std::vector<double> Abs(std::vector<double> X1) {
    std::vector<double> Y;

    for (double x : X1)
        Y.push_back(abs(x));

    return Y;
}

//Magitude of vector
double Mag(std::vector<double> X) {
    return ScalarProduct(X, X);
}
//Multiply by a scalar modifying existing vector
void Mult(std::vector<double>& vec, double x) {
    for (double& i : vec)
        i *= x;
}

//Multiply by a scalar but returns vector
std::vector<double> MultRet(const std::vector<double>& vec, double x) {
    std::vector<double> Y;
    for (const double& i : vec)
        Y.push_back(i * x);
    return Y;
}
//Multiply by a scalar but returns vector but the vector is of pointers
std::vector<double> MultRet(const std::vector<double*>& vec, double x) {
    std::vector<double> Y;
    for (double* i : vec)
        Y.push_back(*i * x);
    return Y;
}
//Divide by a scalar
std::vector<double> Div(std::vector<double> vec, double x) {
    std::vector<double> Y;

    for (double a : vec)
        Y.push_back(a / x);

    return Y;
}
//Inverse vector and multiply by a scalar: xi=c/xi
std::vector<double> Inv(const std::vector<double>& vec, double x) {
    std::vector<double> Y;

    for (const double &a : vec)
        Y.push_back(x/a);

    return Y;
}
//Inverse vector and multiply by a scalar: xi=c/xi but the vector is of pointers
std::vector<double> Inv(const std::vector<double*>& vec, double x) {
    std::vector<double> Y;

    for (double* a : vec)
        Y.push_back(x / *a);

    return Y;
}
//Derivative of vector assuming equally spaced points
std::vector<double> Derivative(std::vector<double> X, double a, double b) {
    std::vector<double> Y;
    double dx = (b - a) / (X.size() - 1);

    Y.push_back((X[0] - X[1])/dx);//forward

    for (int i = 1; i<X.size()-1; i++) //middle point
        Y.push_back((X[i + 1] - X[i - 1])/(2*dx));
        
    Y.push_back((X[X.size() - 1] - X[X.size() - 2]) / dx); //backward

    return Y;
}
//Derivative of vector assuming equally spaced points
std::vector<double> Derivative(std::vector<double> X, double dx) {
    std::vector<double> Y;

    Y.push_back((X[0] - X[1])/dx);//forward

    for (int i = 1; i<X.size()-1; i++) //middle point
        Y.push_back((X[i + 1] - X[i - 1])/(2*dx));
        
    Y.push_back((X[X.size() - 1] - X[X.size() - 2]) / dx); //backward

    return Y;
}

//Averadge of a vector
double Avg(std::vector<double> X) {
    return Sum(X) / X.size();
}

//Smooth the vector making each point averadge of n neighbouring points in each direction.
std::vector<double> Smooth(std::vector<double> X, int N) {
    std::vector<double> Y = std::vector<double>(X.size(),0);

    //sum in Y
    for (int i = 0; i < X.size(); i++) {
        for (int di = -N; di <= N; di++) {
            if (i + di >= 0 && (i + di < X.size()))
                Y[i + di] += X[i];
        }
    }

    //divide by appropriate amount
    for (int i = 0; i < N; i++) {
        Y[i] /= (N + i + 1);
        Y[X.size()-i-1] /= (N + i + 1);
    }
    for (int i = N; i < X.size() - N - 1; i++)
        Y[i] /= (N + N + 1);
    

    return Y;
}

//Subtract funtion from vector assuming points are equally spaced
std::vector<double> Subtract(std::vector<double> X, std::function<double(double)> f, double xmin, double xmax) {
    std::vector<double> Y;
    int N = X.size();
    double dx = (xmax - xmin) / (N-1);
    for (int i = 0; i < N; i++) {
        Y.push_back(X[i] - f(xmin + i * dx));
    }
    return Y;
}

//Log vector to a file
void LogVec(std::vector<double> vec, std::string filename) {
    std::ofstream Log = std::ofstream(filename);
    for (const auto& b : vec) {
        Log << b << "\n";
    }

}

//Log vector to a file with timestamps
void LogVec(std::vector<double> vec, std::string filename, double t0, double dt) {
    std::ofstream Log = std::ofstream(filename);
    for (const auto& b : vec) {
        Log <<t0<<", "<< b << "\n";
        t0 += dt;
    }
}

//mathematical product of difference of a constant and vector
double Product(double p, std::vector<double> X) {
    double prod = 1;
    for (double xi : X) {
        prod *= (p - xi);
    }
    return prod;
}



float colMatrix[7][4][3] = {
//         a             b          c          d
    {{0.5,0.5,0.5},{0.5,0.5,0.5},{1,1,1},{0,0.33,0.67}},
    {{0.5,0.5,0.5},{0.5,0.5,0.5},{1,1,1},{0.,0.1,0.2}},
    {{0.5,0.5,0.5},{0.5,0.5,0.5},{1,1,1},{0.3,0.2,0.2}},
    {{0.5,0.5,0.5},{0.5,0.5,0.5},{1,1,0.5},{0.8,0.9,0.3}},
    {{0.5,0.5,0.5},{0.5,0.5,0.5},{1,0.7,0.4},{0,0.15,0.2}},
    {{0.5,0.5,0.5},{0.5,0.5,0.5},{2,1,0},{0.5,0.2,0.25}},
    {{0.8,0.5,0.4},{0.2,0.4,0.2},{2,1,1},{0,0.25,0.25}}
};
//https://iquilezles.org/articles/palettes/

// std::vector<int> Colour(double t, int i = 0) {
    
//     return std::vector<int>({
//         colMatrix[i][0][0] + colMatrix[i][1][0] * cos(2 * M_PI * (colMatrix[i][2][0] * t + colMatrix[i][3][0])),
//         colMatrix[i][0][1] + colMatrix[i][1][1] * cos(2 * M_PI * (colMatrix[i][2][1] * t + colMatrix[i][3][1])),
//         colMatrix[i][0][2] + colMatrix[i][1][2] * cos(2 * M_PI * (colMatrix[i][2][2] * t + colMatrix[i][3][2]))}
//     );
// }