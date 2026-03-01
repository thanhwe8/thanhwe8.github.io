---
layout: post
title:  "Newton-Raphson's method"
date:   2018-10-10 20:27:22 +0200
categories: Financial Engineering
---
> In this post, we will give you overview of Newton-Raphson's algorithm which is used massively in Option Pricing.

### 1. BASICS OF NEWTON-RAPHSON'S METHOD
Let start with Taylor's theorem:

$$
f(x) = f(a) + f'(a)(x-a) + \displaystyle\frac{f''(a)}{2!}(x-a)^2 + \dots + \displaystyle\frac{f^{(k)}}{k!}(x-a)^k + h_k(x)(x-a)^k
$$

The main idea of the Newton-Raphson's method or Newton's method is to approximate the function $$f(x)$$ by using the straight line or linear function. Newton's method applies the tangent of $$f(x)$$  at the intial point $$x_0$$, find the cross of the tangent line at x-axis at $$x_1$$. Perform recurring process until the solution of $$f(x)$$ is found.

From Taylor's theorem, assume that $$f(x)=0$$, replace $$a=x_0$$ and the series is only dependent on the first-order derivatives, then we have:

$$
f(x) - f(x_0) = 0 - f(x_0) =  f'(x_0)(x-x_0) 
$$

Re-arrange the equation, we have:

$$
x = x_0 - \frac{f(x_0)}{f'(x_0)}
$$

We start with initial guess $$x_0$$, then the algorithm will repeat and find the solution based on that.

Example 1: Finding the root of the function

$$f(x) = x^2 - 4$$

We implement it in Python as below:

{% highlight python %}

f = lambda x: x**2 - 4
df = lambda x: 2*x
x = 9
counter = 0
tole = 0.0000001
while(abs(f(x)) > tole):
    x = x - f(x)/df(x)
    counter += 1
    print("New x is: {x} and counter is: {counter}".format(x=x,counter=counter))

# Ouput
New x is: 4.722222222222222 and counter is: 1
New x is: 2.784640522875817 and counter is: 2
New x is: 2.110545821818145 and counter is: 3
New x is: 2.002895075433833 and counter is: 4
New x is: 2.0000020923367057 and counter is: 5
New x is: 2.0000000000010947 and counter is: 6
{% endhighlight%}

### 2. ISSUES WITH NEWTON-RAPHSON'S METHOD
#### Example 1:
Let start with the function  

$$ f(x) = tan(x) $$

we have  

$$ f'(x) = tan'(x) = \frac{1}{cos^2(x)} $$

Let try with initial guess $$ x0 = pi/2$$  

{% highlight python %}
import math
f = lambda x: math.tan(x)
df = lambda x: 1/(math.cos(x)**2)
x = math.pi/2
counter = 0
tole = 0.0000001
while(abs(f(x)) > tole):
    x = x - f(x)/df(x)
    counter += 1
    print("New x is: {x} and counter is: {counter}".format(x=x,counter=counter))

#Output
New x is: 1.5707963267948966 and counter is: 1
New x is: 1.5707963267948966 and counter is: 2
New x is: 1.5707963267948966 and counter is: 3
New x is: 1.5707963267948966 and counter is: 4
New x is: 1.5707963267948966 and counter is: 5
{% endhighlight%}

The function will run forever as the initial guess is bad and $$f(x)/f'(x)$$ is kept constant in python with $$ f(x) = 1.633123935319537e+16 $$ and $$f'(x) = 3.749399456654644e-33$$.  

#### Example 2:  
Let try function below:

$$
f(x) = x^3 - 2x + 2
$$

with derivatives  

$$
f'(x) = 3x^2 - 2
$$

If we take the initial guess $$x_0 = 0$$ then the loop will enter the 0-1 cycle without converging to a true root as below code:

{% highlight python %}
import math
f = lambda x: x**3 - 2*x + 2
df = lambda x: 3*x**2 - 2
x = 0
counter = 0
tole = 0.0000001
while(abs(f(x)) > tole):
    x = x - f(x)/df(x)
    counter += 1
    print("New x is: {x} and counter is: {counter}".format(x=x,counter=counter))

#part of the output
New x is: 0.0 and counter is: 8988
New x is: 1.0 and counter is: 8989
New x is: 0.0 and counter is: 8990
New x is: 1.0 and counter is: 8991
New x is: 0.0 and counter is: 8992
New x is: 1.0 and counter is: 8993
New x is: 0.0 and counter is: 8994
New x is: 1.0 and counter is: 8995
New x is: 0.0 and counter is: 8996
New x is: 1.0 and counter is: 8997

{% endhighlight%}  

So with Newton's method, we might be able to get to solution to due problems with derivatives of the function or divergence. Make sure to include the exception handling when coding.

### 3. Black-Scholes Implied Volatility using Newton-Raphson method:

Remember that we have the following Black-Scholes formula for call option (no dividend):

$$
C = SN(d_1) - K e^{-rT}N(d_2)
$$

where

$$
d_1 = \frac{ln(S/K) + (r+\sigma^2/2)T}{\sigma \sqrt{T}}
$$

and 

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

Assuming that $$C,P,S,K,T$$ are known in formula above. What we want to find is $$ sigma $$ which is implied volatility.

Set for the call option

$$
f(x) = SN(d_1(x)) - K e^{-rT}N(d_2(x)) - C
$$

where 

$$
d_1 = \frac{ln(S/K) + (r+x^2/2)T}{x \sqrt{T}}
$$

We know that the first-order derivatives of $$f(x)$$ will be the vega of the call option as below:

$$
f'(x) = S\sqrt{T}\phi(d_1) = S \sqrt{T} \frac{1}{\sqrt{2\pi}}  \exp\bigg(-\frac{d_1(x)^2}{2}\bigg)
$$

Then we can apply the Newton-Raphson to find the root which is implied volatility. We can implement it in Python as function below:

{% highlight python %}

def ImpVol(S,K,T,r,C,x0):
    x = x0 = 0.25
    fx = 1
    time_sqrt = math.sqrt(T)
    counter = 0
    tole = 1e-7

    while (abs(fx)>tole):

        d1 = (math.log(S/K)+r*T)/(x*time_sqrt)+0.5*x*time_sqrt
        d2 = d1 - (x*time_sqrt)

        fx = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2) - C
        dfx = S*time_sqrt*norm.pdf(d1)

        x = x - fx/dfx
        counter += 1
        print("new x is: {x} and counter is: {counter}".format(x=x,counter=counter))

    return(x)


{% endhighlight %}  

 
We use the example of Professor Dan Stefanica with: $$C=7, \hspace{0.5cm} S = 25, \hspace{0.5cm}T = 1, \hspace{0.5cm} K = 20, \hspace{0.5cm}T = 1, \hspace{0.5cm}r = 0.05$$

The result will be:

{% highlight python %}
>>> a = ImpVol(25,20,1,0.05,7,0.25)
new x is: 0.386108807417462 and counter is: 1
new x is: 0.3634007015195324 and counter is: 2
new x is: 0.36306326393251365 and counter is: 3
new x is: 0.3630631804856223 and counter is: 4
new x is: 0.3630631804856171 and counter is: 5
{% endhighlight %}  
