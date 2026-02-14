// Ripple project main.c
// To enable image loading with stb_image: place stb_image.h in src/ and uncomment the next two lines.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

const char* vert_glsl = "#version 330 core\n"
"layout(location = 0) in vec2 aPos;\n"
"layout(location = 1) in vec2 aTex;\n"
"out vec2 vTex;\n"
"void main(){ vTex = aTex; gl_Position = vec4(aPos,0.0,1.0); }\n"
"\n";

const char* display_glsl = "#version 330 core\n"
"in vec2 vTex;\n"
"out vec4 outColor;\n"
"\n"
"uniform sampler2D uTex;       // height-field (simulation)\n"
"uniform sampler2D uCaustics;  // caustics texture (grayscale)\n"
"uniform vec2 uRes;\n"
"\n"
"uniform float uTime;\n"
"uniform float uWaveFreq;\n"
"uniform float uWaveSpeed;\n"
"uniform float uWaveWidth;\n"
"\n"
"uniform float uDistort;       // distortion strength\n"
"uniform float uCausticsScale; // brightness of caustics\n"
"\n"
"float hash12(vec2 p) {\n"
"    p = fract(p * vec2(127.1, 311.7));\n"
"    p += dot(p, p + 34.345);\n"
"    return fract(p.x * p.y);\n"
"}\n"
"\n"
"float noise2(vec2 p) {\n"
"    vec2 i = floor(p);\n"
"    vec2 f = fract(p);\n"
"    float a = hash12(i + vec2(0.0, 0.0));\n"
"    float b = hash12(i + vec2(1.0, 0.0));\n"
"    float c = hash12(i + vec2(0.0, 1.0));\n"
"    float d = hash12(i + vec2(1.0, 1.0));\n"
"    vec2 u = f * f * (3.0 - 2.0 * f); // cubic hermite\n"
"    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);\n"
"}\n"
"vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}\n"
"vec2 fade(vec2 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}\n"
"\n"
"float cnoise(vec2 P){\n"
"  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);\n"
"  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);\n"
"  Pi = mod(Pi, 289.0); // To avoid truncation effects in permutation\n"
"  vec4 ix = Pi.xzxz;\n"
"  vec4 iy = Pi.yyww;\n"
"  vec4 fx = Pf.xzxz;\n"
"  vec4 fy = Pf.yyww;\n"
"  vec4 i = permute(permute(ix) + iy);\n"
"  vec4 gx = 2.0 * fract(i * 0.0243902439) - 1.0; // 1/41 = 0.024...\n"
"  vec4 gy = abs(gx) - 0.5;\n"
"  vec4 tx = floor(gx + 0.5);\n"
"  gx = gx - tx;\n"
"  vec2 g00 = vec2(gx.x,gy.x);\n"
"  vec2 g10 = vec2(gx.y,gy.y);\n"
"  vec2 g01 = vec2(gx.z,gy.z);\n"
"  vec2 g11 = vec2(gx.w,gy.w);\n"
"  vec4 norm = 1.79284291400159 - 0.85373472095314 * \n"
"    vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));\n"
"  g00 *= norm.x;\n"
"  g01 *= norm.y;\n"
"  g10 *= norm.z;\n"
"  g11 *= norm.w;\n"
"  float n00 = dot(g00, vec2(fx.x, fy.x));\n"
"  float n10 = dot(g10, vec2(fx.y, fy.y));\n"
"  float n01 = dot(g01, vec2(fx.z, fy.z));\n"
"  float n11 = dot(g11, vec2(fx.w, fy.w));\n"
"  vec2 fade_xy = fade(Pf.xy);\n"
"  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);\n"
"  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);\n"
"  return 2.3 * n_xy;\n"
"}\n"
"mat2 rot(float a) {\n"
"    float s = sin(a), c = cos(a);\n"
"    return mat2(c, -s, s, c);\n"
"}\n"
"\n"
"float fbm_morph(vec2 p, float time) {\n"
"    float v = 0.0;\n"
"    float amp = 0.6;\n"
"    float freq = 1.0;\n"
"    // 4 octaves is enough for smooth blobs\n"
"    for (int i = 0; i < 4; ++i) {\n"
"        // small rotation per-octave dependent on time (tiny angle)\n"
"        float angle = time * (0.03 + float(i) * 0.01);\n"
"        vec2 rp = rot(angle) * p * freq;\n"
"        v += amp * cnoise(rp);\n"
"        freq *= 2.0;\n"
"        amp *= 0.5;\n"
"    }\n"
"    return v;\n"
"}\n"
"\n"
"float computeDepthVal(vec2 uv) {\n"
"    float along = ((1.0 - uv.x) + (1.0 - uv.y)) * 0.5;\n"
"    float phase = along * uWaveFreq - uTime * uWaveSpeed;\n"
"    float cycle = fract(phase);\n"
"    float d = abs(cycle - 0.5);\n"
"    float sigma = max(uWaveWidth, 0.0001);\n"
"    float depthVal = exp(- (d * d) / (sigma * sigma));\n"
"    return clamp(depthVal, 0.0, 1.0);\n"
"}\n"
"\n"
"void main() {\n"
"    float h = texture(uTex, vTex).r;\n"
"    vec2 texel = 1.0 / uRes;\n"
"    float hl = texture(uTex, vTex + vec2(-texel.x, 0.0)).r;\n"
"    float hr = texture(uTex, vTex + vec2(texel.x, 0.0)).r;\n"
"    float hd = texture(uTex, vTex + vec2(0.0, -texel.y)).r;\n"
"    float hu = texture(uTex, vTex + vec2(0.0, texel.y)).r;\n"
"    vec2 grad = vec2((hr - hl) * 0.5, (hu - hd) * 0.5);\n"
"\n"
"    float jitter = sin((vTex.x * 25.0 + uTime * 3.3)) * cos((vTex.y * 18.0 + uTime * 2.1)) * 0.01;\n"
"\n"
"    vec2 offset = grad * uDistort * 0.5 + vec2(jitter);\n"
"    offset += vec2(sin(uTime * 0.8 + vTex.y * 3.0), cos(uTime * 0.9 + vTex.x * 2.0)) * 0.002;\n"
"\n"
"    float scale = 1.05;\n"
"    vec2 uv = vTex - 0.5;\n"
"    uv /= scale;\n"
"    uv += 0.5;\n"
"\n"
"    uv += offset;\n"
"\n"
"    uv = clamp(uv, 0.0, 1.0);\n"
"\n"
"    float ca = texture(uCaustics, uv).r;\n"
"    float ripple_bright = clamp(h, 0.0, 1.0);\n"
"    float base_brightness = ca * uCausticsScale * 0.7 + ripple_bright * 0.8;\n"
"    base_brightness = clamp(base_brightness, 0.0, 1.0);\n"
"\n"
"    float blobScale = 3.2;    // larger -> fewer, larger blobs\n"
"    float threshold = 0.4;   // higher -> sparser blobs\n"
"    float softness = 0.5;    // edge softness (0..0.5)\n"
"    float intensity = 10.5;   // how much the blobs brighten (0..1)\n"
"    float morphTime = uTime * 0.5; // slow morph speed\n"
"\n"
"    vec2 p = uv * blobScale;\n"
"\n"
"    vec2 warp = vec2(\n"
"        fbm_morph(p * 0.6 + vec2(12.3, 7.7), morphTime),\n"
"        fbm_morph(p * 0.6 + vec2(3.1, 9.9), morphTime + 5.0)\n"
"    ) * 0.55; // small warp amplitude\n"
"\n"
"    p += warp;\n"
"\n"
"    float blobNoise = fbm_morph(p, morphTime); // 0..~1\n"
"\n"
"    float blobNoise2 = fbm_morph(p * 1.7 + vec2(8.2, 2.5), morphTime * 0.9) * 0.6;\n"
"    float n = clamp(blobNoise * 0.8 + blobNoise2 * 0.2, 0.0, 1.0);\n"
"\n"
"    float localJitter = fbm_morph(uv * (blobScale * 0.35) + vec2(4.2,1.7), morphTime * 0.7) * 0.08 - 0.04;\n"
"    float t = threshold + localJitter;\n"
"\n"
"    float mask = smoothstep(t - softness, t + softness, n);\n"
"\n"
"    float radial = length((uv - 0.5));\n"
"    float edgeFade = smoothstep(0.85, 0.95, radial);\n"
"    mask *= (1.0 - edgeFade);\n"
"\n"
"    float depthVal = computeDepthVal(vTex);\n"
"    float depthBoost = mix(0.6, 1.6, depthVal);\n"
"\n"
"    float multiplyFactor = 1.0 + intensity * depthBoost * mask;\n"
"    multiplyFactor = clamp(multiplyFactor, 1.0, 1.0 + intensity * 2.0);\n"
"\n"
"    float final_brightness = base_brightness * multiplyFactor;\n"
"\n"
"    outColor = vec4(vec3(final_brightness), 1.0);\n"
"}\n";

const char* sim_glsl =  "#version 330 core\n"
"in vec2 vTex;\n"
"out float fragColor;\n"
"\n"
"uniform sampler2D uPrev; // previous state\n"
"uniform sampler2D uCurr; // current state\n"
"uniform vec2 uRes;\n"
"uniform vec2 uMouse;\n"
"uniform int uDrop;\n"
"uniform float uTime;\n"
"uniform float uDamping;\n"
"\n"
"void main() {\n"
"    vec2 texel = 1.0 / uRes;\n"
"    float left  = texture(uCurr, vTex + vec2(-texel.x, 0)).r;\n"
"    float right = texture(uCurr, vTex + vec2(texel.x, 0)).r;\n"
"    float up    = texture(uCurr, vTex + vec2(0, texel.y)).r;\n"
"    float down  = texture(uCurr, vTex + vec2(0, -texel.y)).r;\n"
"    float center= texture(uCurr, vTex).r;\n"
"    float prev  = texture(uPrev, vTex).r;\n"
"\n"
"    float lap = (left + right + up + down - 4.0 * center);\n"
"\n"
"\n"
"    float next = (2.0 * center - prev) + 0.5*lap;\n"
"\n"
"    // injection: add a gaussian to next where click happens\n"
"    if (uDrop == 1) {\n"
"        float d = distance(vTex, uMouse);\n"
"        float radius = 0.03;\n"
"        float amp = 0.9;\n"
"        float add = amp * exp(- (d * d) / (radius * radius));\n"
"        next += add;\n"
"    }\n"
"\n"
"    next *= uDamping;\n"
"    next = clamp(next, -1.0, 1.0);\n"
"    fragColor = next;\n"
"}\n";
static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, 1, (size_t)sz, f);
    buf[r] = ' ';
    fclose(f);
    return buf;
}

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192]; glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "Shader compile error: %s", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint link_program(GLuint vert, GLuint frag) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vert);
    glAttachShader(p, frag);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192]; glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "Program link error: %s", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// Input state
static double mouse_x = 0.0, mouse_y = 0.0;
static int mouse_down = 0;
static int win_w = 1024, win_h = 768;
static void cursor_cb(GLFWwindow* w, double x, double y) { mouse_x = x; mouse_y = y; }
static void mouse_button_cb(GLFWwindow* w, int button, int action, int mods) { if (button == GLFW_MOUSE_BUTTON_LEFT) mouse_down = (action == GLFW_PRESS || action == GLFW_REPEAT); }

// Image loader (optional stb_image). If stb_image.h isn't available, we generate procedural textures.
static GLuint load_gray_texture(const char* path, int* out_w, int* out_h) {
#ifdef STB_IMAGE_IMPLEMENTATION
    int w,h,comp; unsigned char* data = stbi_load(path, &w, &h, &comp, 1);
    if (!data) return 0;
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    stbi_image_free(data);
    if (out_w) *out_w = w; if (out_h) *out_h = h; return t;
#else
    // Procedural fallback: simple pattern
    int w = 1024, h = 768;
    unsigned char* data = (unsigned char*)malloc(w * h);
    if (!data) return 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float nx = (float)x / w * 10.0f;
            float ny = (float)y / h * 8.0f;
            float v = fabsf(sinf(nx * 6.2831853f) * cosf(ny * 6.2831853f));
            data[y * w + x] = (unsigned char)(fminf(1.0f, v) * 255.0f);
        }
    }
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    free(data);
    if (out_w) *out_w = w; if (out_h) *out_h = h; return t;
#endif
}

int main(void) {
    if (!glfwInit()) { fprintf(stderr, "Failed to init GLFW"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(win_w, win_h, "Ripple (depth-aware)", NULL, NULL);
    if (!window) { fprintf(stderr, "Failed to create window"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, cursor_cb);
    glfwSetMouseButtonCallback(window, mouse_button_cb);

    glewExperimental = GL_TRUE; if (glewInit() != GLEW_OK) { fprintf(stderr, "Failed to init GLEW"); return 1; }
    glGetError();

    // Fullscreen quad (positions + texcoords)
    float quad_verts[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
    };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));


    GLuint vert = compile_shader(GL_VERTEX_SHADER, vert_glsl);
    GLuint sim_frag = compile_shader(GL_FRAGMENT_SHADER, sim_glsl);
    GLuint disp_frag = compile_shader(GL_FRAGMENT_SHADER, display_glsl);
    GLuint sim_prog = link_program(vert, sim_frag);
    GLuint disp_prog = link_program(vert, disp_frag);
    glDeleteShader(vert); glDeleteShader(sim_frag); glDeleteShader(disp_frag);

    // Simulation textures (3) for stable ping-pong and to avoid sampling destination
    const int sim_w = 512, sim_h = 512;
    GLuint sim_tex[3]; glGenTextures(3, sim_tex);
    for (int i = 0; i < 3; ++i) {
        glBindTexture(GL_TEXTURE_2D, sim_tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sim_w, sim_h, 0, GL_RED, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    GLuint fbos[3]; glGenFramebuffers(3, fbos);
    for (int i = 0; i < 3; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbos[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sim_tex[i], 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "FBO %d incomplete", i); return 1; }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Load caustics and depth textures
    int ca_w = 0, ca_h = 0, depth_w = 0, depth_h = 0;
    GLuint causticsTex = load_gray_texture("../texture/caustics-tex.png", &ca_w, &ca_h);

    // Clear sim textures to zero
    glViewport(0,0,sim_w,sim_h);
    for (int i = 0; i < 3; ++i) { glBindFramebuffer(GL_FRAMEBUFFER, fbos[i]); glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT); }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Uniform locations (simulation)
    GLint sim_uPrev = glGetUniformLocation(sim_prog, "uPrev");
    GLint sim_uCurr = glGetUniformLocation(sim_prog, "uCurr");
    GLint sim_uRes  = glGetUniformLocation(sim_prog, "uRes");
    GLint sim_uMouse = glGetUniformLocation(sim_prog, "uMouse");
    GLint sim_uDrop  = glGetUniformLocation(sim_prog, "uDrop");
    GLint sim_uDamp  = glGetUniformLocation(sim_prog, "uDamping");

    // Uniform locations (display)
    GLint disp_uTex = glGetUniformLocation(disp_prog, "uTex");
    GLint disp_uCaustics = glGetUniformLocation(disp_prog, "uCaustics");
    GLint disp_uRes = glGetUniformLocation(disp_prog, "uRes");
    GLint disp_uTime = glGetUniformLocation(disp_prog, "uTime");
    GLint disp_uDistort = glGetUniformLocation(disp_prog, "uDistort");
    GLint disp_uCausticsScale = glGetUniformLocation(disp_prog, "uCausticsScale");

    int iPrev = 0, iCurr = 1, iNext = 2;
    double start = glfwGetTime(), last = start;
    const double fixed_step = 1.0 / 120.0;
    const int max_substeps = 8;
    double sim_accumulator = 0.0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        double now = glfwGetTime();
        double dt = now - last;
        if (dt < 0.0) dt = 0.0;
        if (dt > 0.25) dt = 0.25;
        last = now;
        sim_accumulator += dt;
        float t = (float)(now - start);

        int ww, hh;
        glfwGetWindowSize(window, &ww, &hh);
        float nx = (float)(mouse_x / ww), ny = 1.0f - (float)(mouse_y / hh);

        int substeps = 0;
        while (sim_accumulator >= fixed_step && substeps < max_substeps) {
            // SIMULATION PASS
            glUseProgram(sim_prog);
            glBindFramebuffer(GL_FRAMEBUFFER, fbos[iNext]);
            glViewport(0, 0, sim_w, sim_h);

            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, sim_tex[iPrev]); glUniform1i(sim_uPrev, 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, sim_tex[iCurr]); glUniform1i(sim_uCurr, 1);

            glUniform2f(sim_uRes, (float)sim_w, (float)sim_h);
            glUniform2f(sim_uMouse, nx, ny);
            glUniform1i(sim_uDrop, mouse_down ? 1 : 0);
            glUniform1f(sim_uDamp, 0.998f);

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            // rotate ping-pong indices
            int old = iPrev;
            iPrev = iCurr;
            iCurr = iNext;
            iNext = old;

            sim_accumulator -= fixed_step;
            substeps++;
        }
        if (substeps == max_substeps) {
            sim_accumulator = 0.0;
        }

        // DISPLAY PASS
        glUseProgram(disp_prog);
        glViewport(0, 0, win_w, win_h);
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, sim_tex[iCurr]); glUniform1i(disp_uTex, 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, causticsTex); glUniform1i(disp_uCaustics, 1);

        glUniform2f(disp_uRes, (float)sim_w, (float)sim_h);
        glUniform1f(disp_uTime, t);
        glUniform1f(disp_uDistort, 0.9f);
        glUniform1f(disp_uCausticsScale, 0.8f);

        glBindVertexArray(vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);

        int new_w, new_h; glfwGetFramebufferSize(window, &new_w, &new_h);
        if (new_w != win_w || new_h != win_h) { win_w = new_w; win_h = new_h; glViewport(0, 0, win_w, win_h); }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
