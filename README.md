# AssetPricing
This repository aims to produce a simple Python pricer for various assets.
The goal is to be able to extract data from yahoo finance (or others) and price derivatives accordingly. Autocall pricing would also be a final goal

It currently implements BS & Binomial tree pricing for EU options and CRR tree pricing for US options.  
Barrier option pricing and montecarlo simulations should come very quickly once i've fixed actual issues, meaning i'm not uploading them at the moment.
I'm also working on local volatility model and Heston model.

The project tries to use OOP and not to be script-based. 
