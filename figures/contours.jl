using Compose, Colors
import Cairo, Fontconfig

# === paths ===

u_path(t) = -(0.02 + im*t)^(2/3)

ζ(u) = 4u^3 - 3u

# === plots ===

function arrowhead(curve, time, size, color, eps = 1e-3)
  place = curve(time)
  dir = angle(conj(curve(time + eps) - curve(time - eps)))
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

u_window = context(units = UnitBox(-1, -1, 2, 2), mirror = Mirror(0, 0, 0))
u_frame = compose(context(),
  line([(-1, 0), (1, 0)]),
  line([(-1/sqrt(3), -1), (1/sqrt(3), 1)]),
  line([(1/sqrt(3), -1), (-1/sqrt(3), 1)]),
  linewidth(1w/300), stroke(Gray(0.6))
)
u_contour = compose(u_window,
  (context(), circle(-0.25, 0, 0.015w), fill("orangered")),
  arrowhead(u_path, 0.5, 0.04w, "black"),
  (context(), line(reim.([u_path(t^3) for t in LinRange(-1.09, 1.09, 60)])), stroke("black")),
  u_frame
)

ζ_window = context(units = UnitBox(-6, -6, 12, 12), mirror = Mirror(0, 0, 0))
ζ_frame = compose(context(),
  line([(-6, 0), (6, 0)]),
  line([(0, -6), (0, 6)]),
  linewidth(1w/300), stroke(Gray(0.6))
)
ζ_contour = compose(ζ_window,
  (context(), circle(ζ(-0.25), 0, 0.015w), fill("orangered")),
  arrowhead(t -> ζ(u_path(t)), 0.7, 0.04w, "black"),
  (context(), line(reim.(ζ.([u_path(sign(t)*t^2) for t in LinRange(-1.03, 1.03, 60)]))), stroke("black")),
  ζ_frame
)

# === output ===

draw(PDF("u_contour.pdf", 6cm, 6cm), u_contour)
draw(PDF("zeta_contour.pdf", 6cm, 6cm), ζ_contour)
