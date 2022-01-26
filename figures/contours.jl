module Contours

# for plots
using Compose, Colors, LinearAlgebra, Roots
import Cairo, Fontconfig

# for integral checks
using QuadGK, SpecialFunctions, HypergeometricFunctions
import Plots

# === paths ===

u_path(θ, off) = t -> 1/6 * exp(-im*θ/3) * (cosh(2/3*π*im - t) - dot(off, [exp(2/3*π*im), exp(-2/3*π*im)]))
u_jet(θ, off) = t -> (u_path(θ, off)(t), -1/6 * exp(-im*θ/3) * sinh(2/3*π*im - t))

function ζ(u)
  if length(u) == 1
    return 4u^3 - 3u
  else
    return (4u[1]^3 - 3u[1], 3*(4u[1]^2 - 1)*u[2])
  end
end

# === plots ===

function arrowhead(curve, time, size, color; eps = 1e-3)
  jet = curve(time)
  if length(jet) == 1
    place = jet
    dir = angle(conj(curve(time + eps) - curve(time - eps)))
  else
    place = jet[1]
    dir = angle(conj(jet[2]))
  end
  ctx = context(
    real(place)*cx - size, imag(place)*cy - size, 2size, 2size,
    units = UnitBox(-1, -1, 2, 2),
    rotation = Rotation(dir, 0, 0)
  )
  return compose(ctx,
    polygon([(0.6, 0), (-0.7, 0.7), (-0.4, 0), (-0.7, -0.7)]),
    fill(color)
  )
end

taxinorm(z) = max(abs(real(z)), abs(imag(z)))

# for a nice contour with θ at π or 4π, set `off` to (1.6, 0)
function plotcontours(θ = 0, off = (0, 0); test = true)
  dark = RGB(0, 0.5, 0.7)
  light = RGB(0.6, 0.9, 1.0)
  
  u_range = [find_zero(t -> taxinorm(u_path(θ, off)(t)) - 1.01, search, Bisection()) for search in [(-3, 0), (0, 3)]]
  u_mark = find_zero(t -> taxinorm(u_path(θ, off)(t)) - 0.5, (0, 3), Bisection())
  u_window = context(units = UnitBox(-1, -1, 2, 2), mirror = Mirror(0, 0, 0))
  u_frame = compose(context(),
    line([(-1, 0), (1, 0)]),
    line([(-1/sqrt(3), -1), (1/sqrt(3), 1)]),
    line([(1/sqrt(3), -1), (-1/sqrt(3), 1)]),
    linewidth(1w/300), stroke(Gray(0.6))
  )
  u_contour = compose(u_window,
    (context(), circle(-0.25, 0, 0.015w), fill(dark)),
    (context(), circle(0.25, 0, 0.015w), fill(light)),
    arrowhead(u_jet(θ, off), u_mark, 0.04w, "black"),
    (context(), line(reim.([u_path(θ, off)(t) for t in LinRange(u_range..., 60)])), stroke("black")),
    u_frame
  )
  
  ζ_range = [find_zero(t -> taxinorm(ζ(u_path(θ, off)(t))) - 6.01, search, Bisection()) for search in [(-3, 0), (0, 3)]]
  ζ_mark = find_zero(t -> taxinorm(ζ(u_path(θ, off)(t))) - 3, (0, 3), Bisection())
  ζ_window = context(units = UnitBox(-6, -6, 12, 12), mirror = Mirror(0, 0, 0))
  ζ_frame = compose(context(),
    line([(-6, 0), (6, 0)]),
    line([(0, -6), (0, 6)]),
    linewidth(1w/300), stroke(Gray(0.6))
  )
  ζ_contour = compose(ζ_window,
    (context(), circle(ζ(-0.25), 0, 0.015w), fill(dark)),
    (context(), circle(ζ(0.25), 0, 0.015w), fill(light)),
    arrowhead(t -> ζ(u_jet(θ, off)(t)), ζ_mark, 0.04w, "black"),
    (context(), line(reim.(ζ.([u_path(θ, off)(t) for t in LinRange(ζ_range..., 60)]))), stroke("black")),
    ζ_frame
  )
  
  # === output ===
  
  tag = test ? "_test" : ""
  draw(PDF(string("u_contour", tag, ".pdf"), 6cm, 6cm), u_contour)
  draw(PDF(string("zeta_contour", tag, ".pdf"), 6cm, 6cm), ζ_contour)
end

# === contour integrals ===

besselk_form(z) = t -> begin
  u = u_jet(t)
  im*sqrt(3) * exp(-z*(4u[1]^3 - 3u[1])) * u[2]
end

besselk_integral(z) = quadgk(besselk_form(z), -3, 3, rtol = 1e-8)

besselk_hankel_form(z) = t -> begin
  ζ0 = ζ(u_jet(t))
  1/(im*sqrt(3)) * exp(-z*ζ0[1]) * _₂F₁(1/3, 2/3, 1/2, ζ0[1]^2) * ζ0[2]
end

besselk_hankel_integral(z) = quadgk(besselk_hankel_form(z), -3, 3, rtol = 1e-8)

function f0_hat(ζ0)
  ξ = (1-ζ0)/2
  ξ^(-1/2) * _₂F₁(1/6, 5/6, 1/2, ξ)
end

function f1_hat(ζ0)
  ξ = (1-ζ0)/2
  (1-ξ)^(-1/2) * _₂F₁(1/6, 5/6, 1/2, 1-ξ)
end

g1_hat(ζ0) = _₂F₁(2/3, 4/3, 3/2, (1-ζ0)/2)
g0_hat(ζ0) = _₂F₁(2/3, 4/3, 3/2, (1+ζ0)/2)

besselk_sing_form(z) = t -> begin
  ζ0 = ζ(u_jet(t))
  1/(3im*sqrt(3)) * exp(-z*ζ0[1]) * g0_hat(ζ0[1]) * ζ0[2]
end

besselk_sing_integral(z) = quadgk(besselk_sing_form(z), -3, 3, rtol = 1e-8)

besselk_holo_form(z) = t -> begin
  ζ0 = ζ(u_jet(t))
  1/(3im*sqrt(3)) * exp(-z*ζ0[1]) * g1_hat(ζ0[1]) * ζ0[2]
end

besselk_holo_integral(z) = quadgk(besselk_holo_form(z), -3, 3, rtol = 1e-8)

# === laplace transforms ===

function kappa_hat(ζ1)
  1/sqrt(2) * ζ1^(-1/2) * _₂F₁(1/6, 5/6, 1/2, -ζ1/2)
end

laplace(f_hat; rcutoff = 10, rtol = 1e-8) =
  z -> quadgk(s -> exp(-z*s) * f_hat(s), 0, rcutoff / real(z), rtol = rtol)

besselkappa = laplace(kappa_hat)
besselk_laplace(z) = exp(-z) * besselkappa(z)[1]

function integral_test_plot()
  mesh = LinRange(1/2, 2, 100)
  Plots.plot(mesh, [
    besselk.(1/3, mesh),
    real.(first.(besselk_integral.(mesh))),
    real.(besselk_laplace.(mesh)),
    sqrt(pi/2) * exp.(-mesh) .* mesh.^(-1/2)
  ])
end

end
