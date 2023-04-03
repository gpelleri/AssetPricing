# AssetPricing
This repository aims to produce a simple Python pricer for various assets. It only implements equity assets for now but hopefully it will include others later on.
The final goal is to be able to extract data from yahoo finance (or others) and price asset accordingly. 
I used to have a few different scripts to price some basic options and my current objectives is to regroup all of them and adding new products or models using OOP.
This is obviously a fairly simple pricer that aims only for educational & learning purposes.

Products  
It currently allows to price EU & US options, as well as barrier option and digital option.
I've made some work on a basic autocall pricing, but I do not have any test/reference data yet ...

Volatility Models   
I've made some progress on implementing a local volatility model but not on Heston.
I've just finished a first version of plotting my own implied_volatility surface.  
I still need to make progress on it, otherwise I'm afraid I'll have to use yfinance implied volatility directly to get it.  


