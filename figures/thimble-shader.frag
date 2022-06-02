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

// --- chromasphere ---
// adapted from endolith's complex_colormap code
// https://github.com/endolith/complex_colormap/tree/master/complex_colormap

// endolith created this palette for phase plots of complex functions, like the
// ones in this article. (the article uses a different palette---possibly based
// on a CMYK color wheel?)
// https://www.ams.org/notices/201106/rtx110600768p.pdf

// at each lightness `l`, we want to use a constant-chroma color wheel with the
// most saturated colors an RGB monitor can display. this function approximates
// the radius of that color wheel using an efficient piecewise-linear formula
float appx_chromawheel(in float l) {
    if (l <= 0.0 || 100.0 <= l) return 0.0;
    float lpts [5];
    float cpts [5];
    lpts[0] =   0.0; lpts[1] =   9.2; lpts[2] =  73.8; lpts[3] =  90.0; lpts[4] = 100.0;
    cpts[0] =   0.0; cpts[1] =  10.8; cpts[2] =  39.9; cpts[3] =  12.5; cpts[4] =   0.0;
    float lbot; float ltop;
    float cbot; float ctop;
    if      (l < lpts[1]) { lbot = lpts[0]; ltop = lpts[1]; cbot = cpts[0]; ctop = cpts[1]; }
    else if (l < lpts[2]) { lbot = lpts[1]; ltop = lpts[2]; cbot = cpts[1]; ctop = cpts[2]; }
    else if (l < lpts[3]) { lbot = lpts[2]; ltop = lpts[3]; cbot = cpts[2]; ctop = cpts[3]; }
    else                  { lbot = lpts[3]; ltop = lpts[4]; cbot = cpts[3]; ctop = cpts[4]; }
    float t = (l - lbot) / (ltop - lbot);
    return (1.0-t)*cbot + t*ctop;
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

vec3 line_mix(vec3 stroke, vec3 bg, float width, float pattern_disp, float scaling, float r_px) {
  return mix(bg, stroke, line_part(width, pattern_disp, scaling, r_px));
}

// --- thimble coloring ---

const int N = 4;

const float GRAT = 0.5;

const vec2 axis = 0.5*vec2(1., sqrt(3.));

vec3 surface_color(float end_zone, jet2 zeta, float r_px) {
    jet21 y = proj_y(zeta);
    float scaling = length(y.push);
    
    // paint the surface blue where exp(-f) is small and orange where it's large
    float growth = edge_mix(0., 1., -zeta.pt.x + end_zone, scaling, r_px);
    
    // paint the surface light in upper half-plane and dark in the lower half-plane
    float width = min(2., 0.05 / (scaling * r_px));
    float shading = mix(0.59, 1., growth);
    vec3 grid = lab2rgb(vec3(90., -4., -9.));
    const float check = 0.92;
    vec3 color = lab2rgb(vec3(edge_mix(shading*check*95., shading*95., y.pt, scaling, r_px), 0., 0.));
    if (y.pt < -0.5*GRAT) {
        color = line_mix(lab2rgb(vec3(shading*check*90., -4., -9.)), color, width, mod(y.pt - 0.5*GRAT, GRAT) - 0.5*GRAT, scaling, r_px);
    } else if (0.5*GRAT < y.pt) {
        color = line_mix(lab2rgb(vec3(shading*90., -4., -9.)), color, width, mod(y.pt - 0.5*GRAT, GRAT) - 0.5*GRAT, scaling, r_px);
    }
    return color;
}

vec3 thimble_color(vec2 crit_val, vec2 phase_rcp, vec3 pen_color, vec3 bg, jet2 zeta, float r_px) {
    jet2 dis = mul(phase_rcp, add(zeta, -crit_val));
    jet21 t = proj_x(csqrt(dis));
    float width = max(6., 0.008*min(iResolution.x, iResolution.y));
    return line_mix(pen_color, bg, width, t.pt, length(t.push), r_px);
}

// --- maps ---

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

jet2 dcosh(jet2 z) {
    vec2 cs_x = vec2(cosh(z.pt.x), sinh(z.pt.x));
    vec2 cs_y = vec2(cos(z.pt.y), sin(z.pt.y));
    vec2 cosh_z = vec2(cs_x.x * cs_y.x, cs_x.y * cs_y.y);
    vec2 sinh_z = vec2(cs_x.y * cs_y.x, cs_x.x * cs_y.y);
    return jet2(cosh_z, mul(sinh_z) * z.push);
}

jet2 alg_cosh(jet2 z) {
    vec2 pt = z.pt;
    return scale(0.5, jet2(
        pt + rcp(pt),
        mat2(1.) - mul(rcp(mul(pt, pt)))
    ));
}

// (n-1)^(n-1)*z^n - n*z
jet2 critigon(jet2 z, int n) {
    jet2 lead = z;
    for (int k = 0; k < n-2; k++) {
        lead = mul(z, lead);
    }
    float nf = float(n);
    return mul(z, add(scale(pow(nf-1., nf-1.), lead), -nf*ONE));
}

// --- root finding ---

vec2 newton_step(jet21 f) {
    return (-f.pt / dot(f.push, f.push)) * f.push;
}

const int CHEBYSHEV = 0;
const int CRITIGON = 1;

vec2 horizontal_flow(
    vec2 start_pt,
    vec2 target_val,
    vec2 phase_rcp,
    int fn_name,
    int fn_index,
    float speed,
    float tol,
    int step_max
) {
    jet2 u = jet2(start_pt, mat2(1.));
    for (int step_cnt = 0; step_cnt < step_max; step_cnt++) {
        // evaluate the map
        jet2 zeta;
        if (fn_name == CHEBYSHEV) zeta = scale(-1., chebyshev(u, fn_index));
        else zeta = scale(-1., critigon(u, fn_index));
        
        // find the horizontal displacement to the target critical value. if
        // we're close enough, stop. if not, step closer
        jet21 dis_h = proj_x(mul(phase_rcp, add(zeta, -target_val)));
        if (abs(dis_h.pt) < tol) {
            break;
        } else {
            u = add(u, speed * newton_step(dis_h));
        }
    }
    return u.pt;
}

// --- thimble plots ---

const float PI = 3.141592653589793;

const vec3 blue_lab = vec3(62., -15., -38.);
const vec3 aqua_lab = vec3(40., 30., -71.);

vec3 airy_lucas_plot(int n, vec2 phase_rcp, float view, vec2 fragCoord) {
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = view / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 u = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    
    // color surface
    jet2 zeta = scale(-1., chebyshev(u, n));
    vec3 color = surface_color(0.75*pow(1.5, float(n)), zeta, r_px);
    
    // color thimbles over 1
    /*color = thimble_color(ONE, phase_rcp, lab2rgb(blue_lab), color, zeta, r_px);*/
    float penultimate = cos(PI/float(n));
    vec2 flowed = horizontal_flow(u.pt, ONE, phase_rcp, CHEBYSHEV, n, 0.1, 0.01, 40);
    if (-penultimate < flowed.x || n % 2 == 0) {
        /*color = mix(color, lab2rgb(blue_lab), 0.8);*/
        color = thimble_color(ONE, phase_rcp, lab2rgb(blue_lab), color, zeta, r_px);
    }
    
    // color thimbles over -1
    /*color = thimble_color(-ONE, phase_rcp, lab2rgb(aqua_lab), color, zeta, r_px);*/
    flowed = horizontal_flow(u.pt, -ONE, phase_rcp, CHEBYSHEV, n, 0.1, 0.01, 40);
    if ((-penultimate < flowed.x || n % 2 == 1) && flowed.x < penultimate) {
        /*color = mix(color, lab2rgb(aqua_lab), 0.8);*/
        color = thimble_color(-ONE, phase_rcp, lab2rgb(aqua_lab), color, zeta, r_px);
    }
    
    return color;
}

vec3 bessel_zero_unrolled_plot(vec2 phase_rcp, float view, vec2 fragCoord) {
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = view / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 u = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    
    // color surface
    jet2 zeta = dcosh(u);
    vec3 color = surface_color(535.5, zeta, r_px);
    
    // color thimbles
    color = thimble_color( ONE, phase_rcp, lab2rgb(blue_lab), color, zeta, r_px);
    color = thimble_color(-ONE, phase_rcp, lab2rgb(aqua_lab), color, zeta, r_px);
    
    return color;
}

vec3 bessel_zero_cylinder_plot(vec2 phase_rcp, float view, vec2 fragCoord) {
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = view / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 u = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    
    // get pixel color
    jet2 zeta = alg_cosh(u);
    vec3 color = surface_color(1.1, zeta, r_px);
    color = thimble_color( ONE, phase_rcp, lab2rgb(blue_lab), color, zeta, r_px);
    color = thimble_color(-ONE, phase_rcp, lab2rgb(aqua_lab), color, zeta, r_px);
    return color;
}

vec3 higher_airy_plot(int n, vec2 phase_rcp, float view, vec2 fragCoord) {
    float crit_dens = 1. / float(n-1);
    float crit_angle = 2.*PI*crit_dens;
    vec2 root = vec2(cos(crit_angle), sin(crit_angle));
    vec2 crit_pt_disp = crit_dens*(root - ONE);
    float crit_pt_sep_sq = dot(crit_pt_disp, crit_pt_disp);
    
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = view * crit_dens / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 u = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    
    // get pixel color
    jet2 zeta = scale(-1., critigon(u, n));
    vec3 color = surface_color(2.53, zeta, r_px);
    vec2 crit_val = (n % 2 == 0 ? 1. : -1.) * root;
    for (int k = 0; k < n-1; k++) {
        /*vec3 label = lab2rgb(vec3(49., 29.*crit_val));
        color = thimble_color(crit_val, phase_rcp, label, color, zeta, r_px);*/
        vec2 flowed = horizontal_flow(u.pt, crit_val, phase_rcp, CRITIGON, n, 0.1, 0.01, 40);
        vec2 flow_disp = flowed - crit_dens*crit_val;
        if (dot(flow_disp, flow_disp) < crit_pt_sep_sq) {
            float lite = mix(65., 80., float(k)*crit_dens);
            vec3 label = lab2rgb(vec3(lite, appx_chromawheel(lite)*crit_val));
            color = thimble_color(crit_val, phase_rcp, label, color, zeta, r_px);
            /*if (k == 0) color = mix(color, label, 0.8);*/
        }
        crit_val = mul(root, crit_val);
    }
    
    return color;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // try setting the first argument of chebyshev_plot to 1, 2, 3, 4, 5...
    float angle = 4.*PI*(iMouse.x / iResolution.x + 1./3.);
    vec2 phase_rcp = vec2(cos(-angle), sin(-angle));
    vec3 color = airy_lucas_plot(3, phase_rcp, 0.8, fragCoord);
    /*vec3 color = bessel_zero_unrolled_plot(phase_rcp, 2.75*PI, fragCoord);*/
    /*vec3 color = bessel_zero_cylinder_plot(phase_rcp, 1.2, fragCoord);*/
    /*vec3 color = higher_airy_plot(6, phase_rcp, 2., fragCoord);*/
    fragColor = vec4(sRGB(color), 1.);
}
