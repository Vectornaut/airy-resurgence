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

jet2 add(jet2 f, vec2 c) {
    return jet2(f.pt + c, f.push);
}

jet21 proj_y(jet2 f) {
    return jet21(f.pt.y, vec2(f.push[0].y, f.push[1].y));
}

jet21 dabs(jet21 f) {
    float sgn = f.pt < 0. ? -1. : 1.;
    return jet21(sgn*f.pt, sgn*f.push);
}

// --- complex arithmetic ---

const vec2 ONE = vec2(1., 0.);

//  the complex conjugate of `z`
vec2 conj(vec2 z) {
    return vec2(z.x, -z.y);
}

// multiplication by `z`
mat2 mul(vec2 z) {
    return mat2(z, conj(z).yx);
}

// the product of `z` and `w`
vec2 mul(vec2 z, vec2 w) {
    return mul(z) * w;
}

// the reciprocal of `z`
vec2 rcp(vec2 z) {
    // 1/z = z'/(z'*z) = z'/|z|^2
    return conj(z) / dot(z, z);
}

// the square root of `z`, from the complex arithmetic code listing in
// Appendix C of _Numerical Recipes in C_
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
    vec2 pt = csqrt(z.pt);
    return jet2(pt, mul(0.5*rcp(pt))*z.push);
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
vec3 edge_mix(vec3 neg, vec3 pos, float pattern_disp, float scaling, float r_px) {
  return mix(pos, neg, neg_part(pattern_disp, scaling, r_px));
}

// --- antialiased stripe pattern ---

const int N = 4;

vec3 stripe(jet2 f, float r_px) {
    // set up stripe colors
    vec3 colors [N];
    colors[0] = vec3(0.82, 0.77, 0.71);
    colors[1] = vec3(0.18, 0.09, 0.00);
    colors[2] = vec3(0.02, 0.36, 0.51);
    colors[3] = vec3(0.24, 0.62, 0.67);
    
    // find the displacement to the nearest stripe edge in the pattern space
    jet21 y = proj_y(f);
    float edge = round(y.pt); // the position of the nearest stripe edge
    float pattern_disp = y.pt - edge;
    int n = int(edge)%N; // the index of the color below the edge
    float scaling = length(y.push);
    
    // sample nearest colors
    return edge_mix(colors[n], colors[(n+1)%N], pattern_disp, scaling, r_px);
}

// --- test image ---

const float VIEW = 1.5;

jet2 f(jet2 z) {
    vec2 pt = z.pt;
    vec2 pt_sq = mul(pt, pt);
    return jet2(
        mul(pt, 4.*pt_sq - 3.*ONE),
        3.*mul(4.*pt_sq - ONE) * z.push
    );
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = VIEW / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 u = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    jet2 zeta = f(u);
    
    // color stripes
    vec3 color = stripe(scale(2./3., zeta), r_px);
    
    // color offset strips
    jet21 h = dabs(proj_y(csqrt(add(zeta, -ONE))));
    h.pt -= 0.1;
    vec3 highlight = mix(color, vec3(1., 0., 0.), 0.5);
    color = edge_mix(highlight, color, h.pt, length(h.push), r_px);
    
    fragColor = vec4(color, 1.);
}