#include <cstdarg>
#include <jni.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <set>
#include <string>
#include <GLES3/gl3.h>
#include "zygisk.hpp"
#include "lsplt.hpp"

#define PKG "com.garena.game.codm"
#define LOGF "/data/data/" PKG "/files/codm_tex.log"

static void L(const char *fmt, ...) {
    FILE *f = fopen(LOGF, "a");
    if (!f) return;
    va_list a; va_start(a, fmt);
    vfprintf(f, fmt, a);
    va_end(a); fclose(f);
}

static void (*o_st)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
static void h_st(GLenum t, GLsizei lv, GLenum fmt, GLsizei w, GLsizei h) {
    L("STORAGE %dx%d lv=%d fmt=0x%x\n", w, h, lv, fmt);
    o_st(t, lv, fmt, w, h);
}
static void (*o_ct)(GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void *);
static void h_ct(GLenum t, GLint lv, GLenum fmt, GLsizei w, GLsizei h, GLint b, GLsizei sz, const void *d) {
    if (lv == 0) L("COMPRESSED %dx%d fmt=0x%x size=%d\n", w, h, fmt, sz);
    o_ct(t, lv, fmt, w, h, b, sz, d);
}
static void (*o_ti)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *);
static void h_ti(GLenum t, GLint lv, GLint fmt, GLsizei w, GLsizei h, GLint b, GLenum f, GLenum ty, const void *d) {
    if (lv == 0) L("TEXIMAGE %dx%d fmt=0x%x\n", w, h, fmt);
    o_ti(t, lv, fmt, w, h, b, f, ty, d);
}

static void run() {
    std::set<std::pair<dev_t, ino_t>> done;
    L("=== start ===\n");
    for (int i = 0; i < 300; i++) {
        bool added = false;
        for (auto &m : lsplt::MapInfo::Scan()) {
            if (m.path.rfind("/data/app", 0) != 0) continue;
            if (m.path.size() < 3 || m.path.compare(m.path.size() - 3, 3, ".so") != 0) continue;
            if (!done.insert({m.dev, m.inode}).second) continue;
            lsplt::RegisterHook(m.dev, m.inode, "glTexStorage2D", (void *)h_st, (void **)&o_st);
            lsplt::RegisterHook(m.dev, m.inode, "glCompressedTexImage2D", (void *)h_ct, (void **)&o_ct);
            lsplt::RegisterHook(m.dev, m.inode, "glTexImage2D", (void *)h_ti, (void **)&o_ti);
            L("hooked %s\n", m.path.c_str());
            added = true;
        }
        if (added) lsplt::CommitHook();
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
