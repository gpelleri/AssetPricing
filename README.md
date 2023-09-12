# AssetPricing
This repository aims to produce a simple Python pricer for various assets. It only implements equity assets for now but hopefully it will include others later on.
The final goal is to be able to extract data from yahoo finance (or others) and price asset accordingly. 
I used to have a few different scripts to price some basic options and my current objectives is to regroup all of them and adding new products or models using OOP.
This is obviously a fairly simple pricer that aims only for educational & learning purposes.

## Products  
It currently allows to price EU & US options, as well as barrier option and digital option.
I am also able to price basic options strategies such as spreads, straddles and strangles. I'm willing to produce in the near future notebooks to disclose how & showcase their greeks

## Volatility Models   
I've made some progress on implementing a local volatility model but not on Heston.
I have tried the following approach :

The first is a "Discrete" computation of local vol, using dupire formula and finite difference partial derivatives, which was unsuccessfull.

My second approach uses a parametrized approach to get $\sigma^2 (K,T)$ .
I am using a 5 factor model to fit and calibrate the function. For very liquid stocks, I am now able to sucessfully draw my Implied Volatility Surface
I am also able to compute local vol for a certain (K,T), but I still need to make a monte carlo pricing function that uses it



