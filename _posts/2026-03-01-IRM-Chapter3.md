---
layout: post
title:  "Interest Rate Models - Chapter 3: Vasicek Calibration"
date:   2026-03-01 
categories: [Interest Rate Models]
---

We discuss equivalent martingale measure via simple numerical example.

Table of Contents
--- 

<!-- TOC -->

- [Interest rate models - Chapter 2](#Interest-rate-models)
  - [Table of Contents](#table-of-contents)
- [Proposition 2.1.1](#P211)

<!-- /TOC -->

# Model and bond pricing formula:

Vasicek under risk-neutral measure $$Q$$: 

$$dr_t=\kappa(\theta-r_t)\,dt+\sigma\,dW_t$$

where
- $$\kappa \gt 0$$: mean reversion speed
- $$\theta$$: long-run mean level (under $$Q$$)
- $$\sigma \gt 0$$: volatility
- $$r_0$$; initial short rate

We have zero-coupon bond price with affine form:

$$ P(0,T)=A(T)\,e^{-B(T)r_0}$$

where 

$$ B(T)=\frac{1-e^{-\kappa T}}{\kappa} $$

and 

$$\ln A(T)=\left(\theta-\frac{\sigma^2}{2\kappa^2}\right)\left(B(T)-T\right)-\frac{\sigma^2}{4\kappa}B(T)^2$$

The the model zero rate/yield can be defined as:

$$y^{\text{model}}(0,T)=-\frac{1}{T}\ln P(0,T)$$


Suppose that the curve cooking team gives you the following information on zero rate/yield:

| (T) (years) | $$y^{\text{mkt}}(0,T)$$ |
| ----------: | --------------------: |
|         0.5 |                 4.80% |
|         1.0 |                 4.70% |
|         2.0 |                 4.50% |
|         3.0 |                 4.35% |
|         5.0 |                 4.20% |
|         7.0 |                 4.15% |
|        10.0 |                 4.10% |

you can easily convert each zero rate into a ZCB price accordingly with formula:

$$P^{\text{mkt}}(0,T)=e^{-y^{\text{mkt}}(0,T)\,T}$$

For example:

$$ P^{\text{mkt}}(0,2)=e^{-0.045\cdot2}=e^{-0.09}\approx0.91393$$

As can be seen from the formula itself, we can calibrate 4 parameters $$(\kappa,\theta,\sigma,r_0)$$ from the zero-rate curve or zero-rate. The common choice for practitioners to use are:
- Set $$r_0$$ to the instantaneous short rate (or closest money-market rate) or treat it as free parameter
- Fit $$\kappa,\theta,\sigma$$ by minimizing pricing errors using least squares for example. As the $$\sigma$$ are weakly identified given we don't use any derivative such as cap/floor/swaption, there are some degrees of freedom involved (many combos can be fitted into the same curve).

For optimization, objective function can be chosen to fit either log prices or zero rate/yield as below.

For price:

$$ \min_{\kappa,\theta,\sigma,r_0}\sum_{i} w_i\left(\ln P^{\text{model}}(0,T_i)-\ln P^{\text{mkt}}(0,T_i)\right)^2 $$

For zero rate/yield:

$$ \min_{\kappa,\theta,\sigma,r_0}\sum_{i} w_i\left(y^{\text{model}}(0,T_i)-y^{\text{mkt}}(0,T_i)\right)^2$$

Parameter weights $$w_i$$ can be used to control the long-end/short-end in calibration process per your preference.

Assuming that the quant team gives us the following calibration results:

$$
\boxed{
\kappa=0.35,\quad
\theta=0.040,\quad
\sigma=0.012,\quad
r_0=0.049
}
$$

We can give an intepretation as below:
- $$r_0=4.9\%$$ starts near the short end (0.5y -> 4.8% from the inputs)
- $$\theta=4.0\%$$ gives a lower long-run level (long end of the curve ~ 4.1%)
- $$\kappa=0.35$$ means mean-reversion half-life $$\ln 2/0.35\approx 1.98$$
- $$\sigma=1.2\%$$ means modest volatility



# Vasicek calibration to a zero curve
```r
# Market zero curve input
T  <- c(0.5, 1, 2, 3, 5, 7, 10)                             
y  <- c(0.048, 0.047, 0.045, 0.0435, 0.042, 0.0415, 0.041) 

# Market zero coupon bond prices
P_mkt <- exp(-y * T)
logP_mkt <- log(P_mkt)

# Vasicek ZCB fucntions
B_vas <- function(kappa, T)
{
  (1 - exp(-kappa*T))/kappa
}

logA_vas <- function(kappa, theta, sigma, T)
{
  B <- B_vas(kappa_T)
  term1 <- (theta-(sigma^2)/(2*kappa^2))*(B-T)
  term2 <- -(sigma^2)*(B^2)/(4*kappa)
  term1+term2
}

```