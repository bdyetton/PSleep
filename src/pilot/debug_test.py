import pymc3 as pm

print('I can break here')

with pm.Model() as model:
    prior = pm.Normal('prior', mu=1, sd=1)
    pm.Normal('obs', mu=prior, sd=1, observed=[1,0,-1,0])
    pm.sample()

print('I cannot break here')
