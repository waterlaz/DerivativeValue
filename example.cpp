#include <iostream>
#include <cmath>

#include "DerivativeValue.hpp"


int main(){
    // create variables x = 1.0 and y = 2.0
    auto x = DVariable<float, 2>(0, 1.0);
    auto y = DVariable<float, 2>(1, 2.0);
    
    // create A matrix with expressions from x and y
    Eigen::Matrix<DValue<float, 2>, 2, 2> A;
    A<<x, x+y,
       x-y, y;
    
    // create two vectors with expressions from x and y
    Eigen::Matrix<DValue<float, 2>, 2, 1> a(sin(x), cos(y));
    Eigen::Matrix<DValue<float, 2>, 2, 1> b(cos(x), sin(y));

    // compute numeric value depending on A, a and b
    auto val = a.dot(A*b);
    
    // print gradient:
    std::cout<<val.gradient<<"\n";
    
    // when the result of computation is some vector v
    // one can conveniently get the jacobian
    auto r = sqrt(x*x + y*y);
    auto angle = atan(y/x);
    Eigen::Matrix<DValue<float, 2>, 2, 1> v(r, angle);
    std::cout<<jacobian(v)<<"\n";
}
