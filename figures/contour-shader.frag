//----------------------sRGB----------------------
// from nmz's 3d color space visualization
// https://www.shadertoy.com/view/XddGRN

// map colors from RGB space to sRGB space. in RGB space, color value is
// proportional to light intensity, so linear color-vector interpolation
// corresponds to physical light mixing. in sRGB space, the color encoding
// used by many monitors, we use more of the value interval to represent low
// intensities, and less of the interval to represent high intensities. this
// improves color quantization. see explore-lab/explore-lab-l.frag to learn more

float sRGB(float t){ return mix(1.055*pow(t, 1./2.4) - 0.055, 12.92*t, step(t, 0.0031308)); }
vec3 sRGB(in vec3 c) { return vec3 (sRGB(c.x), sRGB(c.y), sRGB(c.z)); }

//----------------------CIE Lab----------------------
// from nmz's 3d color space visualization
// https://www.shadertoy.com/view/XddGRN

// map colors from Lab space to RGB space. see explore-lab/explore-lab-l.frag
// to learn more

const vec3 wref =  vec3(.95047, 1.0, 1.08883);

float xyzR(float t){ return mix(t*t*t , 0.1284185*(t - 0.139731), step(t,0.20689655)); }

vec3 lab2rgb(in vec3 c)
{
    float lg = 1./116.*(c.x + 16.);
    vec3 xyz = vec3(wref.x*xyzR(lg + 0.002*c.y),
                    wref.y*xyzR(lg),
                    wref.z*xyzR(lg - 0.005*c.z));
    vec3 rgb = xyz*mat3( 3.2406, -1.5372,-0.4986,
                        -0.9689,  1.8758, 0.0415,
                         0.0557, -0.2040, 1.0570);
    return rgb;
}

// --- automatic differentiation ---

// a 1-jet of a map R^2 --> R^2, with image point `pt` and derivative `push`
struct jet2 {
    vec2 pt;
    mat2 push;
};

// a 1-jet of a map R^2 --> R, with image point `pt` and derivative `push`
struct jet21 {
    float pt;
    vec2 push;
};

jet2 scale(float a, jet2 f) {
    return jet2(a*f.pt, a*f.push);
}

jet21 scale(float a, jet21 f) {
    return jet21(a*f.pt, a*f.push);
}

jet21 proj_x(jet2 f) {
    return jet21(f.pt.x, transpose(f.push)[0]);
}

jet21 proj_y(jet2 f) {
    return jet21(f.pt.y, transpose(f.push)[1]);
}

jet21 add(jet21 f, float c) {
    return jet21(f.pt + c, f.push);
}

jet21 dmod(jet21 t, float period) {
    return jet21(mod(t.pt, period), t.push);
}

// --- complex arithmetic ---

const vec2 ONE = vec2(1., 0.);

//  the complex conjugate of `z`
vec2 conj(vec2 z) {
    return vec2(z.x, -z.y);
}

// the sum of `z` and `w`
jet2 add(jet2 z, vec2 c) {
    return jet2(z.pt + c, z.push);
}

// the sum of `z` and `w`
jet2 add(jet2 z, jet2 w) {
    return jet2(z.pt + w.pt, z.push + w.push);
}

// multiplication by `z`
mat2 mul(vec2 z) {
    return mat2(z, conj(z).yx);
}

// the product of `z` and `w`
vec2 mul(vec2 z, vec2 w) {
    return mul(z) * w;
}

jet2 mul(vec2 a, jet2 z) {
    mat2 mul_a = mul(a);
    mat2 mul_z = mul(z.pt);
    return jet2(mul_a*z.pt, mul_a*z.push);
}

jet2 mul(jet2 z, jet2 w) {
    mat2 mul_z = mul(z.pt);
    mat2 mul_w = mul(w.pt);
    return jet2(mul_z*w.pt, mul_z*w.push + mul_w*z.push);
}

// the reciprocal of `z`
vec2 rcp(vec2 z) {
  // 1/z = z'/(z'*z) = z'/|z|^2
  return conj(z) / dot(z, z);
}

// --- complex square root ---
//
// from the complex arithmetic code listing in Appendix C of _Numerical Recipes_
//
// William Press, Saul Teukolsky, William Vetterling, and Brian Flannery,
// _Numerical Recipes in C_, 2nd edition. Cambridge University Press, 1992
//
vec2 csqrt(vec2 z) {
    // sqrt(0) = 0
    if (z.x == 0. && z.y == 0.) {
        return vec2(0.);
    }
    
    // calculate w
    vec2 a = abs(z);
    float w;
    if (a.x >= a.y) {
        float sl = a.y / a.x;
        w = sqrt(a.x) * sqrt(0.5*(1. + sqrt(1. + sl*sl)));
    } else {
        float sl = a.x / a.y;
        w = sqrt(a.y) * sqrt(0.5*(sl + sqrt(1. + sl*sl)));
    }
    
    // construct output
    if (z.x >= 0.) {
        return vec2(w, z.y / (2.*w));
    } else if (z.y >= 0.) {
        return vec2(z.y/(2.*w), w);
    } else {
        return -vec2(z.y/(2.*w), w);
    }
}

jet2 csqrt(jet2 z) {
    vec2 out_pt = csqrt(z.pt);
    return jet2(out_pt, mul(0.5*rcp(out_pt))*z.push);
}

// --- pixel sampling ---

const float A1 = 0.278393;
const float A2 = 0.230389;
const float A3 = 0.000972;
const float A4 = 0.078108;

// Abramowitz and Stegun, equation 7.1.27
float erfc_appx(float t) {
  float r = abs(t);
  float p = 1. + A1*(r + A2*(r + A3*(r + A4*r)));
  float p_sq = p*p;
  float erfc_r = 1. / (p_sq*p_sq);
  return t < 0. ? (2. - erfc_r) : erfc_r;
}

// how much of a pixel's sampling distribution falls on the negative side of an
// edge. `disp` is the pixel's displacement from the edge in pattern space
float neg_part(float pattern_disp, float scaling, float r_px) {
  // find the displacement to the edge in the screen tangent space
  float screen_disp = pattern_disp / scaling;
  
  // integrate our pixel's sampling distribution on the screen tangent space to
  // find out how much of the pixel falls on the negative side of the edge
  return 0.5*erfc_appx(screen_disp / r_px);
}

// find the color of a pixel near an edge between two colored regions.
// `neg` and `pos` are the colors on the negative and positive sides of the
// edge. `disp` is the displacement from the edge

float edge_mix(float neg, float pos, float pattern_disp, float scaling, float r_px) {
  return mix(pos, neg, neg_part(pattern_disp, scaling, r_px));
}

vec3 edge_mix(vec3 neg, vec3 pos, float pattern_disp, float scaling, float r_px) {
  return mix(pos, neg, neg_part(pattern_disp, scaling, r_px));
}

// how much of a pixel's sampling distribution falls on a thickened line.
// `width` is the line thickness, in pixels. `pattern_disp` is the pixel's
// displacement from the line in pattern space
float line_part(float width, float pattern_disp, float scaling, float r_px) {
  // find the displacement to the edge in the screen tangent space
  float screen_disp = pattern_disp / scaling;
  float screen_disp_px = screen_disp / r_px;
  
  // integrate our pixel's sampling distribution on the screen tangent space to
  // find out how much of the pixel falls within `width/2` of the line
  float lower = erfc_appx(screen_disp_px - 0.5*width);
  float upper = erfc_appx(screen_disp_px + 0.5*width);
  return 0.5*(lower - upper);
}

/*float line_mix(float stroke, float bg, float width, float pattern_disp, float scaling, float r_px) {
  return mix(bg, stroke, line_part(width, pattern_disp, scaling, r_px));
}*/

vec3 line_mix(vec3 stroke, vec3 bg, float width, float pattern_disp, float scaling, float r_px) {
  return mix(bg, stroke, line_part(width, pattern_disp, scaling, r_px));
}

/*vec4 line_mix(vec4 stroke, vec4 bg, float width, float pattern_disp, float scaling, float r_px) {
  return mix(bg, stroke, line_part(width, pattern_disp, scaling, r_px));
}*/

// --- antialiased stripe pattern ---

const int N = 4;

/*vec3 stripe(jet2 f, float r_px) {
    // set up stripe colors
    vec3 colors [N];
    colors[0] = vec3(0.82, 0.77, 0.71);
    colors[1] = vec3(0.18, 0.09, 0.00);
    colors[2] = vec3(0.02, 0.36, 0.51);
    colors[3] = vec3(0.24, 0.62, 0.67);
    
    // find the displacement to the nearest stripe edge in the pattern space
    jet21 y = proj_y(f);
    y.pt += 0.5;
    jet21 t = dmod(y, float(N)); // the pattern coordinate
    int n = int(t.pt); // the index of the nearest stripe edge
    float pattern_disp = t.pt - (float(n) + 0.5);
    
    // the edges of the stripes on the screen are level sets of the pattern
    // coordinate `t`. linearizing, we get stripes on the screen tangent space,
    // whose edges are level sets of `t.push`. find the distance to the nearest
    // stripe edge in the screen tangent space
    float screen_dist = abs(pattern_disp) / length(t.push);
    
    // now we can integrate our pixel's sampling distribution on the screen
    // tangent space to find out how much of the pixel falls on the other side
    // of the nearest edge
    float overflow = 0.5*erfc_appx(screen_dist / r_px);
    return mix(colors[n], colors[(n+1)%N], pattern_disp < 0. ? overflow : 1.-overflow);
}*/

const float GRAT = 0.5;

/*const vec3 lower_bg = vec3(0.18, 0.09, 0.00);
const vec3 upper_bg = vec3(0.82, 0.77, 0.71);
const vec3 lower_band = mix(lower_main, upper_main, 0.3);
const vec3 upper_band = mix(lower_main, upper_main, 0.8);*/

const vec2 axis = 0.5*vec2(1., sqrt(3.));

vec3 surface_color(float end_zone, jet2 zeta, float r_px) {
    jet21 y = proj_y(zeta);
    float scaling = length(y.push);
    
    // paint surface blue where exp(-f) is small and orange where it's large
    float growth = edge_mix(-1.,0., zeta.pt.x + end_zone, scaling, r_px);
    growth = edge_mix(growth, 1., zeta.pt.x - end_zone, scaling, r_px);
    
    // paint surface light in upper half-plane and dark in the lower half-plane
    float width = min(2., 0.05 / (scaling * r_px));
    vec3 color = edge_mix(lab2rgb(vec3(70., 38.4*growth*axis)), lab2rgb(vec3(80., 29.8*growth*axis)), y.pt, scaling, r_px);
    if (y.pt < -0.5*GRAT) {
        color = line_mix(lab2rgb(vec3(60., 33.9*growth*axis)), color, width, mod(y.pt - 0.5*GRAT, GRAT) - 0.5*GRAT, scaling, r_px);
    } else if (0.5*GRAT < y.pt) {
        color = line_mix(lab2rgb(vec3(88., 16.8*growth*axis)), color, width, mod(y.pt - 0.5*GRAT, GRAT) - 0.5*GRAT, scaling, r_px);
    }
    return color;
}

vec3 contour_color(vec2 position, float angle, vec3 pen_color, vec3 bg, jet2 zeta, float r_px) {
    vec2 phase_rcp = vec2(cos(-angle), sin(-angle));
    jet2 dis = mul(-phase_rcp, add(zeta, -position));
    jet21 t = proj_x(csqrt(dis));
    float width = max(6., 0.008*min(iResolution.x, iResolution.y));
    return line_mix(pen_color, bg, width, t.pt, length(t.push), r_px);
}

// --- test image ---

jet2 chebyshev(jet2 z, int n) {
    jet2 curr = z;
    jet2 prev = jet2(ONE, mat2(0.));
    for (int k = 1; k < n; k++) {
        jet2 temp = curr;
        curr = add(mul(scale(2., z), curr), scale(-1., prev));
        prev = temp;
    }
    return curr;
}

jet2 alg_cosh(jet2 z) {
    vec2 pt = z.pt;
    return jet2(
        pt + rcp(pt),
        mat2(1.) - mul(rcp(mul(pt, pt)))
    );
}

vec3 chebyshev_plot(int n, float angle, float view, vec2 fragCoord) {
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = view / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 u = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    
    // get pixel color
    jet2 zeta = chebyshev(u, n);
    vec3 color = surface_color(0.75*pow(1.5, float(n)), zeta, r_px);
    color = contour_color( ONE, angle, vec3(0.40, 0.00, 0.10), color, zeta, r_px);
    color = contour_color(-ONE, angle, vec3(0.05, 0.00, 0.10), color, zeta, r_px);
    return color;
}

vec3 alg_cosh_plot(float angle, float view, vec2 fragCoord) {
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = view / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 u = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    
    // get pixel color
    jet2 zeta = alg_cosh(u);
    vec3 color = surface_color(2., zeta, r_px);
    color = contour_color( 2.*ONE, angle, vec3(0.40, 0.00, 0.10), color, zeta, r_px);
    color = contour_color(-2.*ONE, angle, vec3(0.05, 0.00, 0.10), color, zeta, r_px);
    return color;
}

const float PI = 3.141592653589793;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // try setting the first argument of chebyshev_plot to 1, 2, 3, 4, 5...
    float angle = 4.*PI*(iMouse.x / iResolution.x + 1./3.);
    vec3 color = chebyshev_plot(5, angle, 0.8, fragCoord);
    /*vec3 color = alg_cosh_plot(angle, 1.2, fragCoord);*/
    fragColor = vec4(sRGB(color), 1.);
}
