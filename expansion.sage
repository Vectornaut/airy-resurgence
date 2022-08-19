# computes rescaled bessel 1/3 asymptotics using the saddle point approximation
cutoff = 9
R = PolynomialRing(QQ, ','.join(['a' + str(k) for k in reversed(range(1, cutoff+1))]))
a = list(reversed(R.gens()))
S.<t> = PowerSeriesRing(R)
w = sum(a[n-1]*t^n for n in range(1, cutoff+1)) + O(t^(cutoff+1))
rel = t^2 - w^2 - (2/3)*w^3
rel_ideal = R.ideal(rel.coefficients())
sol = rel_ideal.groebner_basis()
w_sol = a[0]*t + sum(-sol[-n].coefficients()[-1]*sol[-n].monomials()[-1]*t^(n+1) for n in range(1,cutoff))
jac = w_sol.derivative()
