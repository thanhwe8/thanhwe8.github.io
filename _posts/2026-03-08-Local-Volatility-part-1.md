---
layout: post
title: "Local Volatility - Part 1"
date: 2026-03-08
categories: [Quantitative Finance]
---

Under the risk neutral-measure, for European call option, we have

$$
C(K,T)=e^{-rT}\mathbb{E}\big[(S_T-K)^+\big].
$$

## First derivative with respect to strike

Assume stock price $$S_T$$ has a density function $$p(s,T)$$, then the above definition can be re-written as

$$
C(K,T)=e^{-rT}\int_0^\infty (s-K)^+\,p(s,T)\,ds.
$$

We need to truncate where $$s \le K$$

$$
C(K,T)=e^{-rT}\int_K^\infty (s-K)\,p(s,T)\,ds.
$$

We now can differentiate with respect to $$K$$ where

$$f(s,K) = (s-K)p(s,T).$$

We use the Leibniz's rule for differentiating an integral with variable lower bound.

Recall that for any function $$F(x)$$ defined by the integral

$$
F(x) = \int_{a(x)}^{b(x)} f(t,x)\,dt
$$

the Leibniz's rule says

$$
\frac{dF}{dx}
=
f(b(x),x)\,b'(x)
-
f(a(x),x)\,a'(x)
+
\int_{a(x)}^{b(x)} \frac{\partial f}{\partial x}(t,x)\,dt
$$

In our case, we have

$$
F(K)=\int_K^\infty f(s,K) ds
$$

where

$$
a(K)=K,\qquad b(K)=\infty
$$

and 

Since the upper bound is constant, we have $$a'(K)=1, b'(K)=0$$. Apply Leibniz's rule

$$
\frac{d}{dK}\int_K^\infty f(s,K)\,ds
=
- f(K,K)
+
\int_K^\infty \frac{\partial f}{\partial K}(s,K)\,ds.
$$

We can observe that

$$
f(K,K) = (K-K)p(K,T)=0
$$

and

$$
\frac{\partial f}{\partial K}(s,K) = -p(s,T)
$$

The differentiation equation can be re-written as

$$
\frac{\partial C}{\partial K}(K,T)
=
e^{-rT}\left[ -0+\int_K^\infty (-p(s,T))\,ds\right]
=
-e^{-rT}\int_K^\infty p(s,T)\,ds
= 
-e^{-rT}\mathbb{Q}(S_T \gt K). 
$$

## Second derivative with respect to strike

From first derivative, we differentiate again w.r.t $$K$$ given $$e^{-rT}$$

$$
\frac{\partial^2 C}{\partial K^2}(K,T)
=
-e^{-rT}\frac{d}{dK}\left(\int_K^\infty p(s,T)\,ds\right).
$$

Recall again the basic rule for integral with a moving boundary for function $$F(K)$$

$$
\frac{d}{dx} \int_{u(x)}^{v(x)} f(t) dt = f(v(x)).v'(x) - f(u(x)).u'(x).
$$

If $$a$$ is constant, we have

$$
F(x) = \int_{a}^{x}f(t)dt
$$

then we have 

$$
\frac{dF(x)}{dx} = \frac{d}{dx} \int_{a}^{x}f(t)dt = f(x)
$$

Back to our particular case, apply the same rule

$$
\frac{d}{dK}\int_K^\infty p(s,T)\,ds = -p(K,T).
$$

then the second derivative w.r.t $$K$$ can be re-written as:

$$
\frac{\partial^2 C}{\partial K^2}(K,T) = -e^{-rT} (-p(K,T)) = e^{-rT}p(K,T). 
$$

## Understand those derivatives intuitively

From above derivation, we can observe that a call price as a function of strike is 
- decreasing in $$K$$ (first derivative)
- convex in $$K$$ (second derivative)

The slope will tell the tail probability and the curvature will tell how quickly that tail probability changes with strike, which is the density function

$$
\frac{\partial C}{\partial K}=-e^{-rT}\mathbb{Q}(S_T \gt K).
$$

$$
\frac{\partial^2 C}{\partial K^2}=e^{-rT}p(K,T).
$$

or in most of the journal, it is rearranged in this form

$$
p(K,T)=f(K,T)=e^{rT}\frac{\partial^2 C}{\partial K^2}
$$

where $$f(K,T)$$ is risk-neutral probability of $$S_T$$. We just completed derivation of the famous formula: Breeden-Litzenberger formula

## Quick toy-example

We have
$$r=5\%$$, 
$$T=1$$, 
$$f(K,T)=0.04$$, 

Then the discounted density is:

$$
e^{-rT}f(K,T)=0.9512 \times 0.04
\approx 0.038
$$

so we got

$$
\frac{\partial^2 C}{\partial K^2}=0.038
$$

the risk-netrual density is

$$
f(K,T)=e^{0.05}\times 0.038 \approx 0.04
$$

## Forward-equation for density

Assuming we have local volatility model as 

$$
dS_t = (r-q)S_t\,dt + \sigma_{\text{loc}}(t,S_t)S_t\,dW_t
$$

The transition density $$p(S,t)$$ satisfies the Fokker-Planck forward PDE (or the forward Kolmogorove equation) which describes

$$
\frac{\partial p}{\partial t}
=
-\frac{\partial}{\partial S}\left((r-q)Sp\right)
+\frac{1}{2}\frac{\partial^2}{\partial S^2}\left(\sigma_{\text{loc}}^2(t,S)S^2 p\right)
$$

Recall from previous section, we have

$$
C(K,T)=e^{-rT}\int_K^\infty (S-K)p(S,T)\,dS
$$

Differentiate w.r.t $$T$$:

$$
\frac{\partial C}{\partial T}
=
-rC + e^{-rT}\int_K^\infty (S-K)\frac{\partial p}{\partial T}(S,T)\,dS
$$

Now substitute the forward PDE for $$\displaystyle\frac{\partial p}{\partial T}$$

$$
\frac{\partial p}{\partial T}
=
-\frac{\partial}{\partial S}\left((r-q)Sp\right)
+\frac{1}{2}\frac{\partial^2}{\partial S^2}\left(\sigma_{\text{loc}}^2(T,S)S^2p\right)
$$

Note that we use $$T$$ instead of $$t$$ in this formula as we evaluate the derivative w.r.t maturity

We have very long equation as below

$$
\frac{\partial C}{\partial T}
=
-rC
+ e^{-rT}\int_K^\infty (S-K)\left[
-\frac{\partial}{\partial S}\left((r-q)Sp\right)
+\frac{1}{2}\frac{\partial^2}{\partial S^2}\left(\sigma_{\text{loc}}^2(T,S)S^2p\right)
\right]dS
$$

Look closely, we can bifurcate the equation into 2 component, the drift term and the diffusion term. We we take care each one separately

### The Drift term

For this integral

$$
\frac{\partial C}{\partial T}
=
-rC
+ e^{-rT}\int_K^\infty (S-K)\left[
-\frac{\partial}{\partial S}\left((r-q)Sp\right)
+\frac{1}{2}\frac{\partial^2}{\partial S^2}\left(\sigma_{\text{loc}}^2(T,S)S^2p\right)
\right]dS
$$

First, we can rewrite $$p(S,T)$$ as just $$p(S)$$ fore readability.

Since $$r-q$$ is constant in S, so we rewrite above equation as 

$$
I_1=-\int_K^\infty (S - K) \frac{d}{dS} A(S) dS
$$

where 

$$
A(S):=(r-q)S p(S).
$$

We can apply integration by parts with following setup

$$
u=S-K,
\qquad
dv=\frac{d}{dS}A(S)\,dS.
$$

Then we have

$$
du=dS,
\qquad
v=A(S)=(r-q)Sp(S)
$$

Using integration by parts formula:

$$
\int u\,dv = uv-\int v\,du.
$$

Our drift integral becomes

$$
I_1
=
-\left[ (S-K)A(S)\right]_K^\infty
+
\int_K^\infty A(S)\,dS.
$$

Substitute back $$A(S) = (r-q)Sp(S)$$ we got

$$
I_1
=
-\left[(S-K)(r-q)Sp(S)\right]_K^\infty
+
\int_K^\infty (r-q)Sp(S)\,dS.
$$

We can evaluate the lower and upper bound for $$uv$$ term as below

At lower bound, we have

$$
(S-K)(r-q)Sp(S)\Big|_{S=K}
=
(K-K)(r-q)Kp(K) = 0
$$

At upper bound, we need to evaluate

$$
\lim_{S\to\infty}(S-K)(r-q)Sp(S)=0.
$$

We have an assumption that the call price is finite (otherwise the model is broken), then $$p(S)$$ must decay fast enough so when $$S\to\infty$$, $$p(S)$$ is 0.

At the end, after the boundary terms vanish, the integral becomes

$$
I_1=\int_K^\infty (r-q)Sp(S)\,dS.
$$

We perform a trick here to decompose $$S$$ into $$S-K + K$$, the integral becomes

$$
\int_K^\infty Sp\,dS
=
\int_K^\infty (S-K)p\,dS + K\int_K^\infty p\,dS
$$

It is clear that the first integral equal to $$C(S,T)$$ and the second integral 

$$
-\frac{\partial C}{\partial K}=e^{-rT}\int_K^\infty p(S,T)\,dS
$$

We get

$$
e^{-rT} I_1 = (r-q)\left(C - K\frac{\partial C}{\partial K}\right)
$$

### The Diffusion term
For the second term

$$
I_2 = \frac{1}{2}\int_K^\infty (S-K)\frac{\partial^2}{\partial S^2}
\left(\sigma_{\text{loc}}^2(T,S)S^2p\right)dS
$$

Denote

$$
g(S):=\sigma_{\text{loc}}^2(T,S)S^2p(S,T).
$$

then the integral can be written 

$$
I_2=\frac12\int_K^\infty (S-K)g''(S)\,dS.
$$

We will apply integration by parts technique here. Let 

$$
u=S-K,\qquad dv=g''(S)\,dS.
$$

then 

$$
du=dS,\qquad v=g'(S).
$$

so the interal can be written as

$$
\int_K^\infty (S-K)g''(S)\,dS
=
\Big[(S-K)g'(S)\Big]_K^\infty
-
\int_K^\infty g'(S)dS.
$$

then we have


$$
I_2
=
\frac12\left(
\Big[(S-K)g'(S)\Big]_K^\infty
-
\int_K^\infty g'(S)\,dS
\right).
$$

**For the first term:**

At lower boundary $$S=K$$, we have

$$
(S-K)g'(S)\big|_{S=K}=(K-K)g'(K)=0.
$$

At the upper boundary $$S\to\infty$$, we need a little bit of articulation for the term involving $$g(S)$$

$$
g(S)=\sigma_{\text{loc}}^2(T,S)S^2p(S,T).
$$

Applying the generalizations version of product rule

$$
\frac{d(uvw)}{x} = \frac{du}{dx}vw + u\frac{dv}{dx} + uv\frac{dw}{dx}.
$$

In our case we have

$$
g'(S)
=
\big(\partial_S \sigma_{\text{loc}}^2(T,S)\big)S^2p(S,T)
+
\sigma_{\text{loc}}^2(T,S)\cdot 2S\,p(S,T)
+
\sigma_{\text{loc}}^2(T,S)S^2\,\partial_S p(S,T).
$$

From asympsotic analysis, we can see that all 3 terms will go to 0 at upper boundary so the term vanish.

**For the second term:**

We apply again an integration by parts technique with

$$
u=1,\qquad dv=g'(S)\,dS.
$$

and 

$$
du=0,\qquad v=g(S).
$$

$$
\int_K^\infty g'(S)\,dS
=
\Big[1\cdot g(S)\Big]_K^\infty
-
\int_K^\infty g(S)\cdot 0\,dS
=
\Big[g(S)\Big]_K^\infty.
$$

again we can get rid of the first term in similar fashion from previous parts. The second term can be wrritten as 

$$
\Big[g(S)\Big]_K^\infty
=
0-g(K)
=
-g(K).
$$

then 

$$
\int_K^\infty (S-K)g''(S)\,dS
=
g(K).
$$

**Final result**

We can aggregate first and second term after evaulating boundaries as

$$
I_2=\frac12 g(K)=\frac12 \sigma_{\text{loc}}^2(T,K)K^2p(K,T).
$$

Apply the result from previous section

$$
p(K,T)=e^{rT}\frac{\partial^2 C}{\partial K^2}
$$

then we have

$$
e^{-rT}I_2=\frac{1}{2}\sigma_{\text{loc}}^2(T,K)K^2
\frac{\partial^2 C}{\partial K^2}
$$


## Final Derivation

Putting everything together now, we have

$$
e^{-rT}I_2=\frac{1}{2}\sigma_{\text{loc}}^2(T,K)K^2
\frac{\partial^2 C}{\partial K^2}
$$

$$
\frac{\partial C}{\partial T}
=
-qC -(r-q)K\frac{\partial C}{\partial K}
+\frac{1}{2}\sigma_{\text{loc}}^2(T,K)K^2\frac{\partial^2 C}{\partial K^2}
$$

$$
\frac{\partial C}{\partial T}
=
-qC -(r-q)K\frac{\partial C}{\partial K}
+\frac{1}{2}\sigma_{\text{loc}}^2(T,K)K^2\frac{\partial^2 C}{\partial K^2}
$$

Rearrange a little bit, we got final result

$$
\frac{1}{2}\sigma_{\text{loc}}^2(T,K)K^2\frac{\partial^2 C}{\partial K^2}
=
\frac{\partial C}{\partial T}
+ qC + (r-q)K\frac{\partial C}{\partial K}
$$

For special case where $$q=0$$, it becomes:

$$
\boxed{
\sigma_{\text{loc}}^2(T,K)=
\frac{
\frac{\partial C}{\partial T}
+ qC + (r-q)K\frac{\partial C}{\partial K}
}{
\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}
}
}
$$




































































