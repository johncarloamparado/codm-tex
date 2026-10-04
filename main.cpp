#include <cstdarg>
#include <jni.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <mutex>
#include <set>
#include <string>
#include <GLES3/gl3.h>
#include "zygisk.hpp"
#include "lsplt.hpp"

#define PKG "com.garena.game.codm"
#define LOGF "/data/data/" PKG "/files/codm_tex.log"

static std::mutex mu, mu2;
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

static void (*o_st)(GLenum, GLsizei, GLenum, GLsizei, GLsizei) = nullptr;
static void h_st(GLenum t, GLsizei lv, GLenum fmt, GLsizei w, GLsizei h) {
    L("STORAGE %dx%d lv=%d fmt=0x%x\n", w, h, lv, fmt);
    if (o_st) o_st(t, lv, fmt, w, h);
}
static void (*o_ct)(GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void *) = nullptr;
static void h_ct(GLenum t, GLint lv, GLenum fmt, GLsizei w, GLsizei h, GLint b, GLsizei sz, const void *d) {
    if (lv == 0) L("COMPRESSED %dx%d fmt=0x%x size=%d\n", w, h, fmt, sz);
    if (o_ct) o_ct(t, lv, fmt, w, h, b, sz, d);
}
static void (*o_ti)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *) = nullptr;
static void h_ti(GLenum t, GLint lv, GLint fmt, GLsizei w, GLsizei h, GLint b, GLenum f, GLenum ty, const void *d) {
    if (lv == 0) L("TEXIMAGE %dx%d fmt=0x%x\n", w, h, fmt);
    if (o_ti) o_ti(t, lv, fmt, w, h, b, f, ty, d);
}

static std::set<std::string> seen;
static void *wrap(const char *n, void *r) {
    if (!n || !r) return r;
    if (!strcmp(n, "glTexStorage2D")) { o_st = (decltype(o_st))r; return (void *)h_st; }
    if (!strcmp(n, "glCompressedTexImage2D")) { o_ct = (decltype(o_ct))r; return (void *)h_ct; }
    if (!strcmp(n, "glTexImage2D")) { o_ti = (decltype(o_ti))r; return (void *)h_ti; }
    if (strncmp(n, "vk", 2) == 0 || strstr(n, "Tex")) {
        bool isnew;
        {
            std::lock_guard<std::mutex> g(mu2);
            isnew = seen.size() < 300 && seen.insert(n).second;
        }
        if (isnew) L("REQ %s\n", n);
    }
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

static void run() {
    std::set<std::pair<dev_t, ino_t>> done;
    L("=== start v2 ===\n");
    for (int i = 0; i < 600; i++) {
        bool added = false;
        for (auto &m : lsplt::MapInfo::Scan()) {
            const std::string &p = m.path;
            if (p.size() < 12 || p.compare(p.size() - 11, 11, "/libunity.so") != 0) continue;
            if (!done.insert({m.dev, m.inode}).second) continue;
            lsplt::RegisterHook(m.dev, m.inode, "dlsym", (void *)h_dlsym, (void **)&o_dlsym);
            lsplt::RegisterHook(m.dev, m.inode, "eglGetProcAddress", (void *)h_egl, (void **)&o_egl);
            lsplt::RegisterHook(m.dev, m.inode, "glTexStorage2D", (void *)h_st, (void **)&o_st);
            lsplt::RegisterHook(m.dev, m.inode, "glCompressedTexImage2D", (void *)h_ct, (void **)&o_ct);
            lsplt::RegisterHook(m.dev, m.inode, "glTexImage2D", (void *)h_ti, (void **)&o_ti);
            L("hooked %s\n", p.c_str());
            added = true;
        }
        if (added) {
            bool ok = lsplt::CommitHook();
            L("commit=%d dlsym=%p egl=%p\n", ok, (void *)o_dlsym, (void *)o_egl);
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
        if (target) std::thread(run).detach();
    }
};
REGISTER_ZYGISK_MODULE(M)
