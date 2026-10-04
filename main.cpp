#include <cstdarg>
#include <jni.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <set>
#include <string>
#include <GLES3/gl3.h>
#include "zygisk.hpp"
#include "lsplt.hpp"

#define PKG "com.garena.game.codm"
#define DIR "/data/data/" PKG "/files/"
#define LOGF DIR "codm_tex.log"
#define CFGF DIR "bias.txt"

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

static void *wrap(const char *n, void *r) {
    if (!n || !r) return r;
    if (!strcmp(n, "glTexParameteri")) { if (!real_tp) real_tp = (decltype(real_tp))r; return r; }
    if (!strcmp(n, "glTexStorage2D")) { if (!o_st) o_st = (decltype(o_st))r; return (void *)h_st; }
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
}

static void run() {
    load_cfg();
    std::set<std::pair<dev_t, ino_t>> done;
    L("=== start v3 bias=%d ===\n", g_bias.load());
    for (int i = 0; i < 600; i++) {
        bool added = false;
        for (auto &m : lsplt::MapInfo::Scan()) {
            const std::string &p = m.path;
            if (p.size() < 12 || p.compare(p.size() - 12, 12, "/libunity.so") != 0) continue;
            if (!done.insert({m.dev, m.inode}).second) continue;
            lsplt::RegisterHook(m.dev, m.inode, "dlsym", (void *)h_dlsym, (void **)&o_dlsym);
            lsplt::RegisterHook(m.dev, m.inode, "eglGetProcAddress", (void *)h_egl, (void **)&o_egl);
            lsplt::RegisterHook(m.dev, m.inode, "glTexStorage2D", (void *)h_st, (void **)&o_st);
            added = true;
        }
        if (added) L("commit=%d\n", lsplt::CommitHook());
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
        if (target) std::thread(run).detach();
    }
};
REGISTER_ZYGISK_MODULE(M)
