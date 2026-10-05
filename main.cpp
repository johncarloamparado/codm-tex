#include <cstdarg>
#include <cerrno>
#include <jni.h>
#include <dlfcn.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <EGL/egl.h>
#include <android/native_window.h>
#include <android/dlext.h>
#include <GLES3/gl3.h>
#include "zygisk.hpp"
#include "lsplt.hpp"

#define PKG "com.garena.game.codm"
#define DIR "/data/data/" PKG "/files/"
#define LOGF DIR "codm_tex.log"
#define CFGF DIR "bias.txt"
#define FPSF DIR "fps.txt"
#define SCLF DIR "scale.txt"
#define MODEF DIR "mode.txt"

static std::mutex mu;
static FILE *lf = nullptr;
static void L(const char *fmt, ...) {
    std::lock_guard<std::mutex> g(mu);
    if (!lf) lf = fopen(LOGF, "a");
    if (!lf) return;
    va_list a; va_start(a, fmt);
    vfprintf(lf, fmt, a);
    va_end(a);
    fflush(lf);
}

/* ---------- Stage 2: mip bias ---------- */
static std::atomic<int> g_scale{65};       /* 0 = patay; 20-99 = % ng native */
static std::atomic<int> g_bias{2};
static std::atomic<int> g_applied{0};
static void (*real_tp)(GLenum, GLenum, GLint) = nullptr;

static bool is_cmp(GLenum f) {
    return (f >= 0x9270 && f <= 0x9279) || (f >= 0x93b0 && f <= 0x93dd);
}

static void (*o_st)(GLenum, GLsizei, GLenum, GLsizei, GLsizei) = nullptr;
static void h_st(GLenum t, GLsizei lv, GLenum fmt, GLsizei w, GLsizei h) {
    if (o_st) o_st(t, lv, fmt, w, h);
    int b = g_bias.load();
    if (b > 0 && real_tp && t == GL_TEXTURE_2D && lv >= 4 &&
        std::max(w, h) >= 512 && is_cmp(fmt)) {
        int base = std::min(b, (int)lv - 2);
        real_tp(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, base);
        int n = ++g_applied;
        if (n <= 100 || n % 500 == 0)
            L("APPLIED #%d %dx%d lv=%d fmt=0x%x base=%d\n", n, w, h, lv, fmt, base);
    }
}

/* ---------- Stage 3: frame-time logger + FPS cap ---------- */
static std::atomic<int> g_fps{0};          /* 0 = walang cap, logger lang */
static std::atomic<pid_t> g_rt{0};         /* render thread (unang tumawag ng swap) */

static inline int64_t now_ns() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static EGLBoolean (*o_swap)(EGLDisplay, EGLSurface) = nullptr;
static EGLBoolean h_swap(EGLDisplay d, EGLSurface s) {
    EGLBoolean r = o_swap ? o_swap(d, s) : EGL_FALSE;

    pid_t tid = (pid_t)syscall(SYS_gettid);
    pid_t exp = 0;
    g_rt.compare_exchange_strong(exp, tid);
    if (g_rt.load() != tid) return r;       /* ibang thread: huwag pakialaman */

    static int64_t last = 0;                /* oras ng nakaraang frame (pagkatapos ng cap) */
    static int64_t deadline = 0;            /* kailan dapat matapos ang kasalukuyang frame */
    static int64_t win_start = 0;
    static std::vector<float> ft;           /* frame times (ms) sa 5-segundong window */

    int64_t t = now_ns();

    /* FPS cap: tulugan ang natitirang oras para pantay ang pagitan ng frames */
    int cap = g_fps.load();
    if (cap > 0) {
        int64_t iv = 1000000000LL / cap;
        if (deadline == 0) deadline = t;
        deadline += iv;
        if (t > deadline + iv) deadline = t;          /* nahuli ng malaki: mag-reset */
        else if (t < deadline) {
            timespec ts;
            ts.tv_sec = deadline / 1000000000LL;
            ts.tv_nsec = deadline % 1000000000LL;
            while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
            t = now_ns();
        }
    }

    if (last != 0) {
        float ms = (float)((t - last) / 1e6);
        if (ms < 1000.f) ft.push_back(ms);           /* huwag isama ang loading pauses */
    }
    last = t;
    if (win_start == 0) win_start = t;

    if (t - win_start >= 5000000000LL && ft.size() >= 10) {
        std::sort(ft.begin(), ft.end());
        double sum = 0; for (float v : ft) sum += v;
        size_t n = ft.size();
        float p50 = ft[n / 2], p99 = ft[std::min(n - 1, (size_t)(n * 0.99))], mx = ft[n - 1];
        size_t s33 = 0, s50 = 0;
        for (float v : ft) { if (v > 33.4f) s33++; if (v > 50.f) s50++; }
        L("FT fps=%.1f avg=%.1fms p50=%.1f p99=%.1f max=%.1f >33ms=%zu >50ms=%zu n=%zu cap=%d bias=%d scale=%d\n",
          1000.0 * n / sum, sum / n, p50, p99, mx, s33, s50, n, cap, g_bias.load(), g_scale.load());
        ft.clear();
        win_start = t;
    }
    return r;
}

/* ---------- Stage 4: render scale (liitan ang buffer ng window) ---------- */
static std::atomic<int> g_fullw{1600}, g_fullh{720};   /* seed: native ng phone mo; lalaki lang, hindi liliit */
static std::atomic<int> g_sbgn{0};

static EGLSurface (*o_cws)(EGLDisplay, EGLConfig, EGLNativeWindowType, const EGLint *) = nullptr;
static EGLSurface h_cws(EGLDisplay d, EGLConfig c, EGLNativeWindowType win, const EGLint *at) {
    int s = g_scale.load();
    ANativeWindow *nwin = (ANativeWindow *)win;
    if (s > 0 && nwin) {
        int w = ANativeWindow_getWidth(nwin), h = ANativeWindow_getHeight(nwin);
        if (w > 0 && h > 0) {
            if (w < h) {
                L("SURFACE portrait %dx%d: skip\n", w, h);
            } else {
                int fw = g_fullw.load(), fh = g_fullh.load();
                if (w > fw) { g_fullw = fw = w; g_fullh = fh = h; }
                int nw = fw * s / 100, nh = fh * s / 100;
                int r = ANativeWindow_setBuffersGeometry(nwin, nw, nh, 0);
                L("SURFACE cur=%dx%d full=%dx%d -> %dx%d r=%d\n", w, h, fw, fh, nw, nh, r);
            }
        }
    }
    return o_cws ? o_cws(d, c, win, at) : EGL_NO_SURFACE;
}

static int32_t (*o_sbg)(ANativeWindow *, int32_t, int32_t, int32_t) = nullptr;
static int32_t h_sbg(ANativeWindow *w, int32_t ww, int32_t hh, int32_t f) {
    int s = g_scale.load(), fw = g_fullw.load(), fh = g_fullh.load();
    int n = ++g_sbgn;
    if (n <= 20) L("SBG call %dx%d fmt=%d\n", ww, hh, f);
    if (s > 0 && fw > 0 && (ww == 0 || hh == 0 || (ww >= fw && hh >= fh))) {
        ww = fw * s / 100; hh = fh * s / 100;
        if (n <= 20) L("SBG overridden -> %dx%d\n", ww, hh);
    }
    return o_sbg ? o_sbg(w, ww, hh, f) : ANativeWindow_setBuffersGeometry(w, ww, hh, f);
}

static void *wrap(const char *n, void *r) {
    if (!n || !r) return r;
    if (!strcmp(n, "glTexParameteri")) { if (!real_tp) real_tp = (decltype(real_tp))r; return r; }
    if (!strcmp(n, "glTexStorage2D")) { if (!o_st) o_st = (decltype(o_st))r; return (void *)h_st; }
    if (!strcmp(n, "eglSwapBuffers")) { if (!o_swap) o_swap = (decltype(o_swap))r; return (void *)h_swap; }
    if (!strcmp(n, "eglCreateWindowSurface")) { if (!o_cws) o_cws = (decltype(o_cws))r; return (void *)h_cws; }
    if (!strcmp(n, "ANativeWindow_setBuffersGeometry")) { if (!o_sbg) o_sbg = (decltype(o_sbg))r; return (void *)h_sbg; }
    return r;
}

static void *(*o_dlsym)(void *, const char *) = nullptr;
static void *h_dlsym(void *h, const char *n) {
    void *r = o_dlsym(h, n);
    return wrap(n, r);
}
static void *(*o_egl)(const char *) = nullptr;
static void *h_egl(const char *n) {
    void *r = o_egl(n);
    return wrap(n, r);
}

static void load_cfg() {
    FILE *f = fopen(CFGF, "r");
    if (f) {
        int v = 2;
        if (fscanf(f, "%d", &v) == 1) g_bias = std::max(0, std::min(v, 4));
        fclose(f);
    } else {
        f = fopen(CFGF, "w");
        if (f) { fprintf(f, "2\n"); fclose(f); }
    }
    f = fopen(FPSF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_fps = (v >= 20 && v <= 120) ? v : 0;
        fclose(f);
    } else {
        f = fopen(FPSF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    f = fopen(SCLF, "r");
    if (f) {
        int v = 65;
        if (fscanf(f, "%d", &v) == 1) g_scale = (v >= 20 && v <= 99) ? v : 0;
        fclose(f);
    } else {
        f = fopen(SCLF, "w");
        if (f) { fprintf(f, "65\n"); fclose(f); }
    }
}

/* mode.txt: 0 = walang ginagawa | 1 = thread lang | 2 = thread + hanap libunity (walang hook) | 3 = buong hook (polling) | 4 = hook sa mismong pag-load ng libunity (walang thread) */
static int read_mode() {
    int v = 0;
    FILE *f = fopen(MODEF, "r");
    if (f) {
        if (fscanf(f, "%d", &v) != 1) v = 0;
        fclose(f);
    } else {
        f = fopen(MODEF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    return std::max(0, std::min(v, 4));
}

/* ---------- Mode 4: i-hook ang libunity ilang sandali LANG matapos itong ma-load ---------- */
static std::atomic<bool> g_hooked{false};
static void *(*o_adle)(const char *, int, const android_dlextinfo *) = nullptr;

static void install_unity_hooks() {
    if (g_hooked.exchange(true)) return;
    for (auto &m : lsplt::MapInfo::Scan()) {
        const std::string &p = m.path;
        if (p.size() < 12 || p.compare(p.size() - 12, 12, "/libunity.so") != 0) continue;
        lsplt::RegisterHook(m.dev, m.inode, "dlsym", (void *)h_dlsym, (void **)&o_dlsym);
        lsplt::RegisterHook(m.dev, m.inode, "eglGetProcAddress", (void *)h_egl, (void **)&o_egl);
        lsplt::RegisterHook(m.dev, m.inode, "glTexStorage2D", (void *)h_st, (void **)&o_st);
        lsplt::RegisterHook(m.dev, m.inode, "eglSwapBuffers", (void *)h_swap, (void **)&o_swap);
        lsplt::RegisterHook(m.dev, m.inode, "eglCreateWindowSurface", (void *)h_cws, (void **)&o_cws);
        lsplt::RegisterHook(m.dev, m.inode, "ANativeWindow_setBuffersGeometry", (void *)h_sbg, (void **)&o_sbg);
        L("pre-commit (unity)\n");
        bool ok = lsplt::CommitHook();
        L("commit=%d (unity)\n", (int)ok);
        return;
    }
    L("libunity hindi nakita sa maps\n");
    g_hooked = false;
}

static void *h_adle(const char *path, int flags, const android_dlextinfo *info) {
    void *r = o_adle ? o_adle(path, flags, info) : android_dlopen_ext(path, flags, info);
    if (r && path) {
        size_t n = strlen(path);
        if (n >= 11 && !strcmp(path + n - 11, "libunity.so")) {
            L("dlopen libunity tapos na: %s\n", path);
            install_unity_hooks();
        }
    }
    return r;
}

static void hook_nativeloader() {
    for (auto &m : lsplt::MapInfo::Scan()) {
        const std::string &p = m.path;
        if (p.size() < 19 || p.compare(p.size() - 19, 19, "/libnativeloader.so") != 0) continue;
        lsplt::RegisterHook(m.dev, m.inode, "android_dlopen_ext", (void *)h_adle, (void **)&o_adle);
        bool ok = lsplt::CommitHook();
        L("nativeloader hook commit=%d\n", (int)ok);
        return;
    }
    L("libnativeloader hindi nakita\n");
}

static void run(int mode) {
    load_cfg();
    std::set<std::pair<dev_t, ino_t>> done;
    std::map<std::pair<dev_t, ino_t>, int64_t> first;   /* kailan unang nakita ang libunity */
    L("=== start v9 mode=%d bias=%d fpscap=%d scale=%d ===\n", mode, g_bias.load(), g_fps.load(), g_scale.load());
    int total = (mode == 1) ? 60 : 1200;
    for (int i = 0; i < total; i++) {
        if (mode >= 2) {
            bool added = false;
            for (auto &m : lsplt::MapInfo::Scan()) {
                const std::string &p = m.path;
                if (p.size() < 12 || p.compare(p.size() - 12, 12, "/libunity.so") != 0) continue;
                std::pair<dev_t, ino_t> key{m.dev, m.inode};
                if (done.count(key)) continue;
                auto it = first.find(key);
                if (it == first.end()) { first[key] = now_ns(); L("SEEN libunity i=%d\n", i); continue; }
                if (now_ns() - it->second < 3000000000LL) continue;   /* hintayin matapos ang load */
                done.insert(key);
                if (mode == 2) { L("mode2: stable, walang hook\n"); continue; }
                lsplt::RegisterHook(m.dev, m.inode, "dlsym", (void *)h_dlsym, (void **)&o_dlsym);
                lsplt::RegisterHook(m.dev, m.inode, "eglGetProcAddress", (void *)h_egl, (void **)&o_egl);
                lsplt::RegisterHook(m.dev, m.inode, "glTexStorage2D", (void *)h_st, (void **)&o_st);
                lsplt::RegisterHook(m.dev, m.inode, "eglSwapBuffers", (void *)h_swap, (void **)&o_swap);
                lsplt::RegisterHook(m.dev, m.inode, "eglCreateWindowSurface", (void *)h_cws, (void **)&o_cws);
                lsplt::RegisterHook(m.dev, m.inode, "ANativeWindow_setBuffersGeometry", (void *)h_sbg, (void **)&o_sbg);
                added = true;
            }
            if (added) {
                L("pre-commit\n");
                bool ok = lsplt::CommitHook();
                L("commit=%d\n", (int)ok);
            }
        }
        sleep(1);
    }
}

class M : public zygisk::ModuleBase {
    zygisk::Api *api; JNIEnv *env; bool target = false;
public:
    void onLoad(zygisk::Api *a, JNIEnv *e) override { api = a; env = e; }
    void preAppSpecialize(zygisk::AppSpecializeArgs *args) override {
        const char *n = env->GetStringUTFChars(args->nice_name, nullptr);
        target = n && strcmp(n, PKG) == 0;
        env->ReleaseStringUTFChars(args->nice_name, n);
        if (!target) api->setOption(zygisk::DLCLOSE_MODULE_LIBRARY);
    }
    void postAppSpecialize(const zygisk::AppSpecializeArgs *) override {
        if (!target) return;
        int mode = read_mode();
        if (mode == 4) {
            load_cfg();
            L("=== start v9 mode=4 bias=%d fpscap=%d scale=%d ===\n", g_bias.load(), g_fps.load(), g_scale.load());
            hook_nativeloader();
            return;
        }
        if (mode >= 1) std::thread(run, mode).detach();
    }
};
REGISTER_ZYGISK_MODULE(M)
