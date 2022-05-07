/* Copyright (c) 2021 Evgeniy Vodolazskiy (waterlaz)  */

#pragma once
#include <Eigen/Dense>

template <typename T, int n>
class DValue {
public:
    typedef Eigen::Matrix<T, n, 1> Vector;
    T value;
    Vector gradient;
    DValue(T _value, Vector _gradient) : 
        value{_value},
        gradient{_gradient}
    {
    }
    DValue(){ }
    DValue(T _value) : 
        value{_value},
        gradient{Vector::Zero()} 
    {
    }
};

template <int n, typename T, int rows, int cols>
Eigen::Matrix<DValue<T, n>, rows, cols> toDValue(const Eigen::Matrix<T, rows, cols>& M){
    return M.template cast<DValue<T, n>>();
}

namespace Eigen {

template <typename T, int n> 
struct NumTraits<DValue<T, n> > : NumTraits<T> {
  typedef DValue<T, n>  Real;
  typedef DValue<T, n>  NonInteger;
  typedef DValue<T, n>  Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

}// namespace Eigen

template <typename T, int m, int n>
Eigen::Matrix<T, m, n> jacobianFromVector(
    const Eigen::Matrix<DValue<T, n>, m, 1>& v)
{
    int rows = v.rows();
    int cols = v[0].gradient.rows();
    Eigen::Matrix<T, m, n> J(rows, cols);
    for(int i=0; i<rows; i++){
        J.row(i) = v[i].gradient.transpose();
    }
    return J;
}

template<typename T, int n>
DValue<T, n> DVariable(int i, T x){
    typename DValue<T, n>::Vector gradient 
        = DValue<T, n>::Vector::Zero();
    gradient[i] = 1;
    return DValue<T, n>(x, gradient);
}

// Arithmetic with scalars

template<typename T, int n> 
DValue<T, n> operator+ ( const T& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(a + b.value,
                        b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator+ ( const DValue<T, n>& b,
                         const T& a)
{
    return DValue<T, n>(a + b.value,
                        b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator- ( const T& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(a - b.value,
                        -b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator- ( const DValue<T, n>& b,
                         const T& a)
{
    return DValue<T, n>(b.value - a,
                        b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator* ( const T& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(a * b.value,
                        a*b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator* ( const DValue<T, n>& b,
                         const T& a)
{
    return DValue<T, n>(a*b.value,
                        a*b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator/ ( const T& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(a / b.value,
                        -a / b.value / b.value * b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator/ ( const DValue<T, n>& b,
                         const T& a)
{
    return DValue<T, n>(b.value / a,
                        b.gradient / a);
}

template<typename T, int n>
DValue<T, n>& operator+= (DValue<T, n>& a, const T& b)
{
    a = a+b;
    return a; 
}

template<typename T, int n>
DValue<T, n>& operator-= (DValue<T, n>& a, const T& b)
{
    a = a-b;
    return a; 
}

template<typename T, int n>
DValue<T, n>& operator*= (DValue<T, n>& a, const T& b)
{
    a = a*b;
    return a; 
}

template<typename T, int n>
DValue<T, n>& operator/= (DValue<T, n>& a, const T& b)
{
    a = a/b;
    return a; 
}

// Arithmetic between DValue

template<typename T, int n> 
DValue<T, n> operator+ ( const DValue<T, n>& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(a.value + b.value,
                  a.gradient + b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator- ( const DValue<T, n>& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(
        a.value - b.value,
        a.gradient - b.gradient);
}

template<typename T, int n> 
DValue<T, n> operator- ( const DValue<T, n>& a)
{
    return DValue<T, n>(
        -a.value,
        -a.gradient);
}


template<typename T, int n> 
DValue<T, n> operator* ( const DValue<T, n>& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(
        a.value * b.value,
        a.value * b.gradient 
      + b.value * a.gradient);
}

template<typename T, int n> 
DValue<T, n> operator/ ( const DValue<T, n>& a, 
                         const DValue<T, n>& b )
{
    return DValue<T, n>(
        a.value / b.value,
        (  b.value * a.gradient 
         - a.value * b.gradient) 
            / (b.value*b.value) );
}

template<typename T, int n>
DValue<T, n>& operator+= (DValue<T, n>& a, const DValue<T, n>& b)
{
    a = a+b;
    return a; 
}

template<typename T, int n>
DValue<T, n>& operator-= (DValue<T, n>& a, const DValue<T, n>& b)
{
    a = a-b;
    return a; 
}

template<typename T, int n>
DValue<T, n>& operator*= (DValue<T, n>& a, const DValue<T, n>& b)
{
    a = a*b;
    return a; 
}

template<typename T, int n>
DValue<T, n>& operator/= (DValue<T, n>& a, const DValue<T, n>& b)
{
    a = a/b;
    return a; 
}

// Trigonometry

template<typename T, int n>
DValue<T, n> sin(DValue<T, n> x){
    return DValue<T, n>(sin(x.value),
                        cos(x.value) * x.gradient);
}

template<typename T, int n>
DValue<T, n> cos(DValue<T, n> x){
    return DValue<T, n>(cos(x.value),
                        -sin(x.value) * x.gradient);
}

template<typename T, int n>
DValue<T, n> atan(DValue<T, n> x){
    return DValue<T, n>(atan(x.value),
                        1.0/(1+x.value*x.value) * x.gradient);
}

template<typename T, int n>
DValue<T, n> sqrt(DValue<T, n> x){
    return DValue<T, n>(sqrt(x.value),
                        1.0/2.0/sqrt(x.value) * x.gradient);
}
