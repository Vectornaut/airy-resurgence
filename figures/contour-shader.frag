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

jet21 proj_y(jet2 f) {
    return jet21(f.pt.y, vec2(f.push[0].y, f.push[1].y));
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

// multiplication by `z`
mat2 mul(vec2 z) {
    return mat2(z, conj(z).yx);
}

// the product of `z` and `w`
vec2 mul(vec2 z, vec2 w) {
    return mul(z) * w;
}

// --- approximate error function ---

const float A1 = 0.278393;
const float A2 = 0.230389;
const float A3 = 0.000972;
const float A4 = 0.078108;

// Abramowitz and Stegun, equation 7.1.27
float erfc_appx(float t) {
  float p = 1. + A1*(t + A2*(t + A3*(t + A4*t)));
  float p_sq = p*p;
  return 1. / (p_sq*p_sq);
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
}

// --- test image ---

const float VIEW = 3.;

jet2 f(jet2 z) {
    vec2 pt = z.pt;
    vec2 pt_sq = mul(pt, pt);
    return jet2(
        mul(pt, mul(ONE/3., pt_sq) - ONE),
        mul(pt_sq - ONE) * z.push
    );
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // find screen point
    float small_dim = min(iResolution.x, iResolution.y);
    float r_px = VIEW / small_dim; // the inner radius of a pixel in the Euclidean metric of the screen
    jet2 z = jet2(r_px * (2.*fragCoord - iResolution.xy), mat2(1.));
    
    // get pixel color
    /*vec3 color = stripe(mul(z, z), r_px);*/
    vec3 color = stripe(f(z), r_px);
    fragColor = vec4(color, 1.);
}
