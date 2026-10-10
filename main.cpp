#include <cstdarg>
#include <cerrno>
#include <jni.h>
#include <dlfcn.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <dirent.h>
#include <fcntl.h>
#include <sched.h>
#include <cstdio>
#include <ctime>
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

#ifndef RUSAGE_THREAD
#define RUSAGE_THREAD 1
#endif

#define PKG "com.garena.game.codm"
#define DIR "/data/data/" PKG "/files/"
#define LOGF DIR "codm_tex.log"
#define CFGF DIR "bias.txt"
#define FPSF DIR "fps.txt"
#define SCLF DIR "scale.txt"
#define MODEF DIR "mode.txt"
#define FZF DIR "fz.txt"
#define THF DIR "th.txt"
#define COREF DIR "core.txt"
#define NEARF DIR "near.txt"
#define BLKF DIR "blk.txt"
#define MINF DIR "minsz.txt"
#define DLYF DIR "delay.txt"
#define SKIPF DIR "skip.txt"
#define STYF DIR "style.txt"
#define SEDF DIR "sedge.txt"
#define FLATF DIR "flat.txt"
#define TTEXF DIR "testtex.txt"
#define TTSZF DIR "ttsize.txt"
#define TTSMF DIR "ttsm.txt"
#define TTAXF DIR "ttaniso.txt"

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
static std::atomic<int> g_scale{65};       /* 0 = off; 20-99 = % of native resolution */
static std::atomic<int> g_bias{2};
static std::atomic<int> g_applied{0};
static void (*real_tp)(GLenum, GLenum, GLint) = nullptr;

static bool is_cmp(GLenum f) {
    return (f >= 0x9270 && f <= 0x9279) || (f >= 0x93b0 && f <= 0x93dd);
}

/* ---------- Stage 8: blocky texture (near.txt) ---------- */
/* near.txt: 0 = off | 1 = on (default). Only matters when block/bias path is active.
   Flagged textures are forced to NEAREST filtering = mosaic blocks. */
static std::atomic<int> g_near{1};
static std::atomic<int> g_blk{16};      /* blk.txt: target size (px) per texture; 0 = use bias.txt */
static std::atomic<int> g_flagged{0}, g_forced{0}, g_smp{0};
static const GLuint MAXTEX = 1u << 18;
static std::atomic<uint8_t> g_tf[1u << 18];        /* 1 = texture forced to NEAREST */
static thread_local GLuint tl_bound[32];
static thread_local int tl_unit = 0;

static void (*o_act)(GLenum) = nullptr;
static void h_act(GLenum u) {
    int i = (int)u - 0x84C0;
    tl_unit = (i >= 0 && i < 32) ? i : 0;
    if (o_act) o_act(u);
}

static void (*o_bind)(GLenum, GLuint) = nullptr;
static void h_bind(GLenum t, GLuint id) {
    if (t == GL_TEXTURE_2D) tl_bound[tl_unit] = id;
    if (o_bind) o_bind(t, id);
}

static void (*o_del)(GLsizei, const GLuint *) = nullptr;
static void h_del(GLsizei n, const GLuint *ids) {
    if (ids) {
        for (GLsizei i = 0; i < n; i++) {
            if (ids[i] < MAXTEX) g_tf[ids[i]] = 0;
            for (int u = 0; u < 32; u++) if (tl_bound[u] == ids[i]) tl_bound[u] = 0;
        }
    }
    if (o_del) o_del(n, ids);
}

/* ---------- v16: Detail Cut (style.txt = 0 off | 1..4 = min LOD; sedge.txt = 1 -> sharp MAG edges) ---------- */
/* ---------- v19: Flat Color (flat.txt = 0 off | 1..4 = strength). Locks texture to tiny mips = near-solid color. */
static std::atomic<int> g_style{0};
static std::atomic<int> g_sedge{1};
static std::atomic<int> g_flat{0};
static std::atomic<int> g_ttsize{0};   /* ttsize.txt: -1 finer | 0 normal | 1..4 bigger pixels (2x,4x,8x,16x) */
static std::atomic<int> g_ttsm{0};     /* ttsm.txt: 0 sharp | 1 soft mip blend */
static std::atomic<int> g_ttax{0};     /* ttaniso.txt: 0 force anisotropy 1 | 1 leave the game's anisotropy (Test Tex only) */
static std::atomic<int> g_testtex{0};  /* testtex.txt: 0 off | 1 = Test Tex (NEAREST only, full mips) */
#define TEX_MIN_LOD 0x813A
#define TEX_MAX_LEVEL 0x813D
#define TEX_ANISO 0x84FE
#ifndef GL_NEAREST_MIPMAP_NEAREST
#define GL_NEAREST_MIPMAP_NEAREST 0x2700
#endif

#ifndef GL_NEAREST_MIPMAP_LINEAR
#define GL_NEAREST_MIPMAP_LINEAR 0x2702
#endif
#ifndef GL_LINEAR_MIPMAP_NEAREST
#define GL_LINEAR_MIPMAP_NEAREST 0x2701
#endif
/* Test Tex filters from ttsize / ttsm */
static void tt_filters(GLint &mn, GLint &mg) {
    int sz = g_ttsize.load(), sm = g_ttsm.load();
    bool nomip = sz < 0;
    if (sm >= 1) { mn = nomip ? GL_NEAREST : GL_NEAREST_MIPMAP_LINEAR; mg = GL_NEAREST; }
    else              { mn = nomip ? GL_NEAREST : GL_NEAREST_MIPMAP_NEAREST; mg = GL_NEAREST; }
}

static void h_tp(GLenum t, GLenum p, GLint v) {
    int st = g_style.load();
    int fl = g_flat.load();
    int tt = g_testtex.load();
    if ((st > 0 || fl > 0 || (tt > 0 && !g_ttax.load())) && p == TEX_ANISO && v > 1) v = 1;
    if (t == GL_TEXTURE_2D && (p == GL_TEXTURE_MIN_FILTER || p == GL_TEXTURE_MAG_FILTER)) {
        GLuint id = tl_bound[tl_unit];
        if (id > 0 && id < MAXTEX && g_tf[id].load()) {
            if (tt > 0) {
                /* Test Tex: sharp texels, keep full resolution mips */
                GLint mn, mg; tt_filters(mn, mg);
                v = (p == GL_TEXTURE_MAG_FILTER) ? mg : mn;
                g_forced++;
            } else if (fl > 0) {
                /* Flat: both filters NEAREST so the tiny mip stays solid */
                v = GL_NEAREST;
                g_forced++;
            } else if (st > 0) {
                /* Detail Cut: MAG only */
                if (p == GL_TEXTURE_MAG_FILTER) { v = GL_NEAREST; g_forced++; }
            } else {
                v = GL_NEAREST;
                g_forced++;
            }
        }
    }
    if (real_tp) real_tp(t, p, v);
}

static void (*o_tpf)(GLenum, GLenum, GLfloat) = nullptr;
static void h_tpf(GLenum t, GLenum p, GLfloat v) {
    if ((g_style.load() > 0 || g_flat.load() > 0 || (g_testtex.load() > 0 && !g_ttax.load())) && p == TEX_ANISO && v > 1.0f) v = 1.0f;
    if (o_tpf) o_tpf(t, p, v);
}

static void (*o_smp)(GLuint, GLenum, GLint) = nullptr;
static void h_smp(GLuint s, GLenum p, GLint v) {
    g_smp++;
    if (o_smp) o_smp(s, p, v);
}

/* ---------- Stage 9: texture groups, skip list, min size, start delay ---------- */
static inline int64_t now_ns();
static std::atomic<int> g_minsz{64};                 /* minsz.txt: smaller than this = leave untouched */
static std::atomic<int> g_delay{0};                  /* delay.txt: seconds after launch with no texture changes */
static std::atomic<int64_t> g_tstart{0};
struct Sk { int fmt, a, b; };
static std::vector<Sk> g_skip;                       /* skip.txt: "fmt size" (size 0 = all) or "fmt w h"; read at startup only */
struct Grp { int fmt; int sz; int n; int ap; };
static const int MAXG = 96;
static Grp g_grp[MAXG];
static int g_ng = 0;
static std::mutex g_gm;
static std::atomic<int> g_gdirty{0};

static void grp_note(int fmt, int sz, bool applied) {
    std::lock_guard<std::mutex> lk(g_gm);
    for (int i = 0; i < g_ng; i++)
        if (g_grp[i].fmt == fmt && g_grp[i].sz == sz) { g_grp[i].n++; if (applied) g_grp[i].ap++; g_gdirty = 1; return; }
    if (g_ng < MAXG) { g_grp[g_ng].fmt = fmt; g_grp[g_ng].sz = sz; g_grp[g_ng].n = 1; g_grp[g_ng].ap = applied ? 1 : 0; g_ng++; g_gdirty = 1; }
}

/* Permanent skips (compiled in; Reset/skip.txt cannot remove these).
   37497 256x512 = Lava Remix scope/reticle atlas.
   37493/37492/37497 128x128 = partial character-detail protection. */
static const Sk g_builtin[] = {
    {37497, 256, 512},
    {37493, 128, 128},
    {37492, 128, 128},
    {37497, 128, 128},
};

static bool in_skip(int fmt, int w, int h) {
    int sz = std::max(w, h);
    for (auto &p : g_builtin)
        if (p.fmt == fmt && p.a == w && p.b == h) return true;
    for (auto &p : g_skip) {
        if (p.fmt != fmt) continue;
        if (p.b == 0) { if (p.a == 0 || p.a == sz) return true; }
        else if (p.a == w && p.b == h) return true;
    }
    return false;
}

/* v17: TEXN = every created texture (phone time) — compare two logs to identify scopes/reddots */
static std::atomic<int> g_texn{0};
static void texn_log(int w, int h, int lv, int fmt, bool cand) {
    int n = ++g_texn;
    if (n > 6000) return;
    time_t tt = time(nullptr); struct tm tmv; localtime_r(&tt, &tmv);
    L("TEXN %02d:%02d:%02d t=%.1f %dx%d lv=%d fmt=0x%x cand=%d\n", tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
      (now_ns() - g_tstart.load()) / 1e9, w, h, lv, fmt, cand ? 1 : 0);
}

static void (*o_st)(GLenum, GLsizei, GLenum, GLsizei, GLsizei) = nullptr;
static void h_st(GLenum t, GLsizei lv, GLenum fmt, GLsizei w, GLsizei h) {
    if (o_st) o_st(t, lv, fmt, w, h);
    int b = g_bias.load(), k = g_blk.load();
    int mx = std::max(w, h);
    bool cand = (t == GL_TEXTURE_2D && lv >= 3 && mx >= 64 && is_cmp(fmt));
    if (t == GL_TEXTURE_2D && mx >= 16) texn_log((int)w, (int)h, (int)lv, (int)fmt, cand);
    if (!cand) return;
    bool skip = in_skip((int)fmt, (int)w, (int)h) || mx < g_minsz.load();
    int dl = g_delay.load();
    if (!skip && dl > 0 && now_ns() - g_tstart.load() < (int64_t)dl * 1000000000LL) skip = true;

    /* Test Tex: force NEAREST sampling only — full texture data, sharp pixel look. */
    int tt = g_testtex.load();
    if (tt > 0) {
        grp_note((int)fmt, mx, !skip);
        if (!skip && real_tp) {
            GLuint id = tl_bound[tl_unit];
            if (id > 0 && id < MAXTEX) {
                g_tf[id] = 1; g_flagged++;
                GLint mn, mg; tt_filters(mn, mg);
                real_tp(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mn);
                real_tp(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mg);
                int sz = g_ttsize.load();
                if (sz > 0) {
                    /* bigger pixels: start from a coarser mip, keep at least 8 px */
                    int base = sz;
                    while (base > 0 && (mx >> base) < 8) base--;
                    if (base > (int)lv - 1) base = (int)lv - 1;
                    if (base > 0) real_tp(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, base);
                }
            }
            int n = ++g_applied;
            if (n <= 100 || n % 500 == 0)
                L("TESTTEX #%d %dx%d lv=%d fmt=0x%x size=%d smooth=%d aniso=%d\n", n, w, h, lv, fmt, g_ttsize.load(), g_ttsm.load(), g_ttax.load());
        }
        return;
    }

    /* v19 Flat Color: lock to tiny mips so the surface is near-solid color (lighting still applies). */
    int fl = g_flat.load();
    if (fl > 0) {
        /* strength 1..4 → target remaining size ~8, 4, 2, 1 px */
        static const int FTGT[] = {0, 8, 4, 2, 1};
        int tgt = FTGT[fl < 4 ? fl : 4];
        int base = 0;
        if (!skip && mx > tgt)
            while ((mx >> (base + 1)) >= tgt && base < (int)lv - 1) base++;
        grp_note((int)fmt, mx, base > 0);
        if (base > 0 && real_tp) {
            real_tp(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, base);
            real_tp(GL_TEXTURE_2D, TEX_MAX_LEVEL, base);   /* lock: no higher mips */
            GLuint id = tl_bound[tl_unit];
            if (id > 0 && id < MAXTEX) {
                g_tf[id] = 1; g_flagged++;
                real_tp(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                real_tp(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            }
            int n = ++g_applied;
            if (n <= 100 || n % 500 == 0)
                L("FLAT #%d %dx%d lv=%d fmt=0x%x base=%d tgt=%d\n", n, w, h, lv, fmt, base, tgt);
        }
        return;
    }

    int sty = g_style.load();
    if (sty > 0) {
        /* Detail Cut: raise min LOD only; original filters kept except optional sharp MAG */
        int lod = skip ? 0 : std::min(sty, (int)lv - 2);
        grp_note((int)fmt, mx, lod > 0);
        if (lod > 0 && real_tp) {
            real_tp(GL_TEXTURE_2D, TEX_MIN_LOD, lod);
            if (g_sedge.load()) {
                GLuint id = tl_bound[tl_unit];
                if (id > 0 && id < MAXTEX) {
                    g_tf[id] = 1; g_flagged++;
                    real_tp(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                }
            }
            int n = ++g_applied;
            if (n <= 100 || n % 500 == 0)
                L("STYLE #%d %dx%d lv=%d fmt=0x%x lod=%d edge=%d\n", n, w, h, lv, fmt, lod, g_sedge.load());
        }
        return;
    }
    int base = 0;
    if (!skip) {
        if (k > 0) {
            if (mx > k)
                while ((mx >> (base + 1)) >= k && base < (int)lv - 2) base++;
        } else if (b > 0 && lv >= 4 && mx >= 512) {
            base = std::min(b, (int)lv - 2);
        }
    }
    grp_note((int)fmt, mx, base > 0);
    if (base > 0 && real_tp) {
        real_tp(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, base);
        if (g_near.load()) {
            GLuint id = tl_bound[tl_unit];
            if (id > 0 && id < MAXTEX) {
                g_tf[id] = 1; g_flagged++;
                real_tp(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                real_tp(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            }
        }
        int n = ++g_applied;
        if (n <= 100 || n % 500 == 0)
            L("APPLIED #%d %dx%d lv=%d fmt=0x%x base=%d blk=%d\n", n, w, h, lv, fmt, base, k);
    }
}

static void grp_dump(double t_s) {
    Grp tmp[MAXG]; int n;
    { std::lock_guard<std::mutex> lk(g_gm); n = g_ng; for (int i = 0; i < n; i++) tmp[i] = g_grp[i]; g_gdirty = 0; }
    for (int i = 0; i < n; i++)
        L("TEXG t=%.0f fmt=%d hex=0x%x sz=%d n=%d ap=%d\n", t_s, tmp[i].fmt, tmp[i].fmt, tmp[i].sz, tmp[i].n, tmp[i].ap);
}

/* ---------- Stage 3: frame-time logger + FPS cap ---------- */
static std::atomic<int> g_fps{0};          /* 0 = no cap, logger only */
static std::atomic<pid_t> g_rt{0};         /* render thread (first caller of swap) */

static inline int64_t now_ns() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ---------- Stage 5: freeze logger (logging only, does not change gameplay) ---------- */
/* fz.txt: 0 = off (no extra hooks) | 1 = on (default) */
static std::atomic<int> g_fz{1};
static std::atomic<int64_t> g_t0{0};       /* module start time for timestamps */

enum { K_COMP, K_LINK, K_PBIN, K_BUF, K_TEX, K_N };
static std::atomic<int> c_cnt[2][K_N];     /* [0] = render thread, [1] = ibang thread */
static std::atomic<int64_t> c_ns[2][K_N];

static thread_local pid_t tl_tid = 0;
static inline bool on_rt() {
    if (!tl_tid) tl_tid = (pid_t)syscall(SYS_gettid);
    return tl_tid == g_rt.load();
}

#define TIMED(K, call) do { \
    int64_t t0_ = now_ns(); call; int64_t d_ = now_ns() - t0_; \
    int i_ = on_rt() ? 0 : 1; c_cnt[i_][K]++; c_ns[i_][K] += d_; } while (0)

static void (*o_comp)(GLuint) = nullptr;
static void h_comp(GLuint s) { if (!o_comp) return; TIMED(K_COMP, o_comp(s)); }

static void (*o_link)(GLuint) = nullptr;
static void h_link(GLuint p) { if (!o_link) return; TIMED(K_LINK, o_link(p)); }

static void (*o_pbin)(GLuint, GLenum, const void *, GLsizei) = nullptr;
static void h_pbin(GLuint p, GLenum f, const void *b, GLsizei n) {
    if (!o_pbin) return; TIMED(K_PBIN, o_pbin(p, f, b, n));
}

static void (*o_buf)(GLenum, GLsizeiptr, const void *, GLenum) = nullptr;
static void h_buf(GLenum t, GLsizeiptr sz, const void *d, GLenum u) {
    if (!o_buf) return; TIMED(K_BUF, o_buf(t, sz, d, u));
}

static void (*o_ti2)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *) = nullptr;
static void h_ti2(GLenum t, GLint l, GLint ifmt, GLsizei w, GLsizei h, GLint b, GLenum f, GLenum ty, const void *p) {
    if (!o_ti2) return; TIMED(K_TEX, o_ti2(t, l, ifmt, w, h, b, f, ty, p));
}

static void (*o_cti2)(GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void *) = nullptr;
static void h_cti2(GLenum t, GLint l, GLenum f, GLsizei w, GLsizei h, GLint b, GLsizei n, const void *p) {
    if (!o_cti2) return; TIMED(K_TEX, o_cti2(t, l, f, w, h, b, n, p));
}

static void (*o_tsi2)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void *) = nullptr;
static void h_tsi2(GLenum t, GLint l, GLint x, GLint y, GLsizei w, GLsizei h, GLenum f, GLenum ty, const void *p) {
    if (!o_tsi2) return; TIMED(K_TEX, o_tsi2(t, l, x, y, w, h, f, ty, p));
}

static void (*o_ctsi2)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLsizei, const void *) = nullptr;
static void h_ctsi2(GLenum t, GLint l, GLint x, GLint y, GLsizei w, GLsizei h, GLenum f, GLsizei n, const void *p) {
    if (!o_ctsi2) return; TIMED(K_TEX, o_ctsi2(t, l, x, y, w, h, f, n, p));
}

static void (*o_ts3)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLsizei) = nullptr;
static void h_ts3(GLenum t, GLsizei lv, GLenum f, GLsizei w, GLsizei h, GLsizei d) {
    if (!o_ts3) return; TIMED(K_TEX, o_ts3(t, lv, f, w, h, d));
}

/* ---------- Stage 6: per-thread CPU (alin ang busy habang may freeze) ---------- */
/* th.txt: 0 = off | 1 = on (default). No new thread; only the render thread reads /proc. */
static std::atomic<int> g_th{1};
static int g_sched_mode = -1;              /* 1 = schedstat (ns) | 0 = stat (ticks) */
struct Thr { pid_t tid; int fd; int64_t prev, cur, wprev, wcur; char name[20]; };
static std::vector<Thr> g_thr;             /* only the render thread handles this */
static int g_nthr = 0;
static const size_t MAXT = 48;

/* ---------- Stage 7: core logger (logging lang) ---------- */
/* core.txt: 0 = off | 1 = on (default). Only works when fz.txt = 1 at th.txt = 1. */
static std::atomic<int> g_core{1};
static const int NC = 8;
static std::vector<std::pair<pid_t, int>> g_um;    /* UnityMain: tid, fd ng stat */
static int g_um_now = -1, g_um_prev = -1;
static int g_h_um[NC], g_h_rt[NC], g_h_fz[NC];
static int g_h_n = 0;

static int read_int_file(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char b[32];
    ssize_t n = read(fd, b, sizeof(b) - 1);
    close(fd);
    if (n <= 0) return -1;
    b[n] = 0;
    return atoi(b);
}

static int cpu_khz(int c) {
    char p[96];
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", c);
    return read_int_file(p);
}

/* field 39 ("processor") ng /proc/.../stat = huling core na tinakbuhan */
static int core_of(int fd) {
    char b[640];
    ssize_t n = pread(fd, b, sizeof(b) - 1, 0);
    if (n <= 0) return -1;
    b[n] = 0;
    char *p = strrchr(b, ')');
    if (!p || !p[1]) return -1;
    p += 2;
    for (int i = 0; i < 36; i++) {
        p = strchr(p, ' ');
        if (!p) return -1;
        p++;
    }
    return atoi(p);
}

static void core_info() {
    char buf[480]; int len = snprintf(buf, sizeof buf, "CORES");
    for (int c = 0; c < NC && len < (int)sizeof(buf) - 60; c++) {
        char p[96], a[16], m[16];
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpu_capacity", c);
        int cap = read_int_file(p);
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        int mx = read_int_file(p);
        if (cap >= 0) snprintf(a, sizeof a, "%d", cap); else strcpy(a, "n/a");
        if (mx >= 0) snprintf(m, sizeof m, "%d", mx); else strcpy(m, "n/a");
        len += snprintf(buf + len, sizeof(buf) - len, " cpu%d=cap:%s/max:%s", c, a, m);
    }
    L("%s\n", buf);
}

static void core_sample() {
    g_um_prev = g_um_now;
    int c = -1;
    for (auto &u : g_um) {
        if (u.second < 0) continue;
        int x = core_of(u.second);
        if (x >= 0) { c = x; break; }
    }
    g_um_now = c;
    int r = sched_getcpu();
    if (c >= 0 && c < NC) g_h_um[c]++;
    if (r >= 0 && r < NC) g_h_rt[r]++;
    g_h_n++;
}

static void core_freeze(long long frame, float ms, double t_s) {
    int c = g_um_now;
    if (c >= 0 && c < NC) g_h_fz[c]++;
    L("CORE t=%.2f f=%lld ms=%.0f um=%d prev=%d um_khz=%d rt=%d\n",
      t_s, frame, ms, c, g_um_prev, c >= 0 ? cpu_khz(c) : -1, sched_getcpu());
}

static int put_arr(char *b, size_t cap, const int *a) {
    int len = 0;
    for (int i = 0; i < NC; i++)
        len += snprintf(b + len, cap - len, "%s%d", i ? "," : "", a[i]);
    return len;
}

static void core_hist(double t_s) {
    if (g_h_n <= 0) return;
    char buf[300]; int len = snprintf(buf, sizeof buf, "CH t=%.1f n=%d um=", t_s, g_h_n);
    len += put_arr(buf + len, sizeof(buf) - len, g_h_um);
    len += snprintf(buf + len, sizeof(buf) - len, " rt=");
    len += put_arr(buf + len, sizeof(buf) - len, g_h_rt);
    len += snprintf(buf + len, sizeof(buf) - len, " umfz=");
    put_arr(buf + len, sizeof(buf) - len, g_h_fz);
    L("%s\n", buf);
    memset(g_h_um, 0, sizeof g_h_um);
    memset(g_h_rt, 0, sizeof g_h_rt);
    memset(g_h_fz, 0, sizeof g_h_fz);
    g_h_n = 0;
}

static bool read_cpu(int fd, int64_t &run, int64_t &wait) {
    char b[256];
    ssize_t n = pread(fd, b, sizeof(b) - 1, 0);
    if (n <= 0) return false;
    b[n] = 0;
    if (g_sched_mode == 1) {
        char *e = nullptr;
        run = strtoll(b, &e, 10);
        wait = (e && *e) ? strtoll(e, &e, 10) : 0;
        return true;
    }
    char *p = strrchr(b, ')');
    if (!p) return false;
    unsigned long ut = 0, st = 0;
    if (sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &ut, &st) != 2) return false;
    run = (int64_t)(ut + st) * 10000000LL;
    wait = 0;
    return true;
}

static void th_init() {
    char path[64];
    snprintf(path, sizeof path, "/proc/self/task/%d/schedstat", (int)getpid());
    g_sched_mode = 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        g_sched_mode = 1;
        int64_t r = 0, w = 0;
        if (!read_cpu(fd, r, w) || r <= 0) g_sched_mode = 0;
        close(fd);
    }
    L("THR mode=%s\n", g_sched_mode == 1 ? "schedstat" : "stat");
    if (g_core.load()) core_info();
}

static void th_rescan() {
    for (auto &t : g_thr) if (t.fd >= 0) close(t.fd);
    g_thr.clear();
    for (auto &u : g_um) if (u.second >= 0) close(u.second);
    g_um.clear();
    auto d = opendir("/proc/self/task");
    if (!d) return;
    std::vector<Thr> all;
    while (dirent *e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        pid_t tid = (pid_t)atoi(e->d_name);
        char path[64];
        snprintf(path, sizeof path, "/proc/self/task/%d/%s", (int)tid, g_sched_mode == 1 ? "schedstat" : "stat");
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        Thr t; memset(&t, 0, sizeof t);
        t.tid = tid; t.fd = fd;
        if (!read_cpu(fd, t.cur, t.wcur)) { close(fd); continue; }
        t.prev = t.cur; t.wprev = t.wcur;
        snprintf(path, sizeof path, "/proc/self/task/%d/comm", (int)tid);
        int cf = open(path, O_RDONLY | O_CLOEXEC);
        ssize_t n = 0;
        if (cf >= 0) { n = read(cf, t.name, 15); close(cf); }
        if (n > 0) {
            t.name[n] = 0;
            for (char *c = t.name; *c; c++) if (*c == '\n' || *c == ' ') *c = (*c == '\n') ? 0 : '_';
        } else strcpy(t.name, "?");
        all.push_back(t);
    }
    closedir(d);
    if (g_core.load()) {
        for (auto &a : all) {
            if (strcmp(a.name, "UnityMain") != 0 || g_um.size() >= 2) continue;
            char sp[64];
            snprintf(sp, sizeof sp, "/proc/self/task/%d/stat", (int)a.tid);
            int sf = open(sp, O_RDONLY | O_CLOEXEC);
            if (sf >= 0) g_um.push_back({a.tid, sf});
        }
    }
    g_nthr = (int)all.size();
    std::sort(all.begin(), all.end(), [](const Thr &a, const Thr &b) { return a.cur > b.cur; });
    pid_t me = getpid(), rt = g_rt.load();
    for (size_t i = 0; i < all.size(); i++) {
        if (i < MAXT || all[i].tid == me || all[i].tid == rt) g_thr.push_back(all[i]);
        else close(all[i].fd);
    }
}

static void th_snapshot() {
    for (auto &x : g_thr) {
        if (x.fd < 0) continue;
        x.prev = x.cur; x.wprev = x.wcur;
        if (!read_cpu(x.fd, x.cur, x.wcur)) { close(x.fd); x.fd = -1; x.cur = x.prev; x.wcur = x.wprev; }
    }
}

static void th_log(long long frame, float ms, double t_s) {
    pid_t me = getpid(), rt = g_rt.load();
    std::vector<std::pair<int64_t, size_t>> v;
    double sum = 0, rwait = 0;
    for (size_t i = 0; i < g_thr.size(); i++) {
        int64_t dl = g_thr[i].cur - g_thr[i].prev;
        if (dl < 0) dl = 0;
        sum += dl / 1e6;
        if (g_thr[i].tid == rt) rwait = (g_thr[i].wcur - g_thr[i].wprev) / 1e6;
        if (dl >= 3000000LL) v.push_back({dl, i});
    }
    std::sort(v.begin(), v.end(), [](const std::pair<int64_t, size_t> &a, const std::pair<int64_t, size_t> &b) { return a.first > b.first; });
    char buf[420]; int len = 0;
    len += snprintf(buf + len, sizeof(buf) - len, "THR t=%.2f f=%lld ms=%.0f sum=%.0f rwait=%.0f nthr=%d |", t_s, frame, ms, sum, rwait, g_nthr);
    for (size_t k = 0; k < v.size() && k < 6 && len < (int)sizeof(buf) - 40; k++) {
        const Thr &x = g_thr[v[k].second];
        len += snprintf(buf + len, sizeof(buf) - len, " %s%s=%.0f", x.name,
                        x.tid == rt ? "*" : (x.tid == me ? "!" : ""), v[k].first / 1e6);
    }
    L("%s\n", buf);
}

static inline double tv_ms(const timeval &a) { return a.tv_sec * 1000.0 + a.tv_usec / 1000.0; }

static void log_gl_info() {
    typedef const GLubyte *(*gs_t)(GLenum);
    gs_t gs = (gs_t)dlsym(RTLD_DEFAULT, "glGetString");
    if (!gs) { void *h = dlopen("libGLESv3.so", RTLD_NOW); if (h) gs = (gs_t)dlsym(h, "glGetString"); }
    if (!gs) { L("GLINFO glGetString n/a\n"); return; }
    const char *r = (const char *)gs(GL_RENDERER), *v = (const char *)gs(GL_VERSION), *e = (const char *)gs(GL_EXTENSIONS);
    L("GLINFO renderer=%s version=%s\n", r ? r : "?", v ? v : "?");
    L("GLINFO ext lod_bias=%d aniso=%d\n", (e && strstr(e, "lod_bias")) ? 1 : 0, (e && strstr(e, "anisotropic")) ? 1 : 0);
}

static EGLBoolean (*o_swap)(EGLDisplay, EGLSurface) = nullptr;
static EGLBoolean h_swap(EGLDisplay d, EGLSurface s) {
    int64_t t_pre = now_ns();
    EGLBoolean r = o_swap ? o_swap(d, s) : EGL_FALSE;
    int64_t t_post = now_ns();

    pid_t tid = (pid_t)syscall(SYS_gettid);
    pid_t exp = 0;
    g_rt.compare_exchange_strong(exp, tid);
    if (g_rt.load() != tid) return r;       /* other thread: leave alone */

    static int64_t last = 0;                /* time of previous frame (after cap) */
    static int64_t deadline = 0;            /* when the current frame should finish */
    static int64_t win_start = 0;
    static std::vector<float> ft;           /* frame times (ms) in a 5-second window */
    static int64_t frame_no = 0;
    static int win_fz = 0;                  /* freeze count in the stats window */
    static int fz_logged = 0;               /* FREEZE line limit per session */
    static rusage ru_last;
    static bool ru_ok = false;
    static int64_t next_scan = 0;

    int64_t t = t_post;
    if (g_t0.load() == 0) g_t0 = t;
    frame_no++;
    if (frame_no == 30) log_gl_info();

    /* FPS cap: sleep the remaining time so frame intervals stay even */
    int cap = g_fps.load();
    if (cap > 0) {
        int64_t iv = 1000000000LL / cap;
        if (deadline == 0) deadline = t;
        deadline += iv;
        if (t > deadline + iv) deadline = t;          /* fell far behind: reset deadline */
        else if (t < deadline) {
            timespec ts;
            ts.tv_sec = deadline / 1000000000LL;
            ts.tv_nsec = deadline % 1000000000LL;
            while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
            t = now_ns();
        }
    }

    float ms = 0.f;
    if (last != 0) {
        ms = (float)((t - last) / 1e6);
        if (ms < 1000.f) ft.push_back(ms);           /* skip loading pauses in stats */
    }
    last = t;
    if (win_start == 0) win_start = t;

    /* Freeze logger: read and reset counters each frame */
    if (g_fz.load()) {
        rusage ru; bool ok = (getrusage(RUSAGE_THREAD, &ru) == 0);
        bool th_on = g_th.load() != 0;
        if (th_on && g_sched_mode < 0) th_init();
        if (th_on && g_thr.empty()) th_rescan();
        if (th_on) th_snapshot();
        if (th_on && g_core.load()) core_sample();
        int cc[2][K_N]; int64_t cn[2][K_N];
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < K_N; k++) { cc[i][k] = c_cnt[i][k].exchange(0); cn[i][k] = c_ns[i][k].exchange(0); }
        if (last != 0 && ms >= 100.f) {
            win_fz++;
            if (fz_logged < 400) {
                fz_logged++;
                double cpu = 0, vcs = 0, ics = 0, minf = 0, majf = 0;
                if (ok && ru_ok) {
                    cpu = tv_ms(ru.ru_utime) + tv_ms(ru.ru_stime) - tv_ms(ru_last.ru_utime) - tv_ms(ru_last.ru_stime);
                    vcs = (double)(ru.ru_nvcsw - ru_last.ru_nvcsw);
                    ics = (double)(ru.ru_nivcsw - ru_last.ru_nivcsw);
                    minf = (double)(ru.ru_minflt - ru_last.ru_minflt);
                    majf = (double)(ru.ru_majflt - ru_last.ru_majflt);
                }
                L("FREEZE t=%.2f f=%lld ms=%.0f swap=%.0f cpu=%.0f vcs=%.0f ics=%.0f minf=%.0f majf=%.0f"
                  " | rt comp=%d/%.0f link=%d/%.0f pbin=%d/%.0f buf=%d/%.0f tex=%d/%.0f"
                  " | bg comp=%d/%.0f link=%d/%.0f pbin=%d/%.0f buf=%d/%.0f tex=%d/%.0f\n",
                  (t - g_t0.load()) / 1e9, (long long)frame_no, ms, (t_post - t_pre) / 1e6, cpu, vcs, ics, minf, majf,
                  cc[0][K_COMP], cn[0][K_COMP] / 1e6, cc[0][K_LINK], cn[0][K_LINK] / 1e6,
                  cc[0][K_PBIN], cn[0][K_PBIN] / 1e6, cc[0][K_BUF], cn[0][K_BUF] / 1e6,
                  cc[0][K_TEX], cn[0][K_TEX] / 1e6,
                  cc[1][K_COMP], cn[1][K_COMP] / 1e6, cc[1][K_LINK], cn[1][K_LINK] / 1e6,
                  cc[1][K_PBIN], cn[1][K_PBIN] / 1e6, cc[1][K_BUF], cn[1][K_BUF] / 1e6,
                  cc[1][K_TEX], cn[1][K_TEX] / 1e6);
                if (th_on) th_log((long long)frame_no, ms, (t - g_t0.load()) / 1e9);
                if (th_on && g_core.load()) core_freeze((long long)frame_no, ms, (t - g_t0.load()) / 1e9);
            }
        }
        if (ok) { ru_last = ru; ru_ok = true; }
        if (th_on && t >= next_scan) { th_rescan(); next_scan = t + 5000000000LL; }
    }

    if (t - win_start >= 5000000000LL && ft.size() >= 10) {
        std::sort(ft.begin(), ft.end());
        double sum = 0; for (float v : ft) sum += v;
        size_t n = ft.size();
        float p50 = ft[n / 2], p99 = ft[std::min(n - 1, (size_t)(n * 0.99))], mx = ft[n - 1];
        size_t s33 = 0, s50 = 0;
        for (float v : ft) { if (v > 33.4f) s33++; if (v > 50.f) s50++; }
        L("FT fps=%.1f avg=%.1fms p50=%.1f p99=%.1f max=%.1f >33ms=%zu >50ms=%zu n=%zu cap=%d bias=%d scale=%d t=%.1f fz=%d\n",
          1000.0 * n / sum, sum / n, p50, p99, mx, s33, s50, n, cap, g_bias.load(), g_scale.load(),
          (t - g_t0.load()) / 1e9, win_fz);
        if (g_near.load()) L("NEAR flagged=%d forced=%d smp=%d\n", g_flagged.load(), g_forced.load(), g_smp.load());
        {
            static int64_t last_gd = 0;
            if (g_gdirty.load() && t - last_gd >= 20000000000LL) { last_gd = t; grp_dump((t - g_t0.load()) / 1e9); }
        }
        if (g_fz.load() && g_th.load() && g_core.load()) core_hist((t - g_t0.load()) / 1e9);
        ft.clear();
        win_fz = 0;
        win_start = t;
    }
    return r;
}

/* ---------- Stage 4: render scale (shrink the window buffer) ---------- */
static std::atomic<int> g_fullw{1600}, g_fullh{720};   /* seed: device native; only grows, never shrinks */
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
    if (!strcmp(n, "glTexParameteri")) { if (!real_tp) real_tp = (decltype(real_tp))r; return (g_near.load() || g_style.load() || g_flat.load() || g_testtex.load()) ? (void *)h_tp : r; }
    if (!strcmp(n, "glTexParameterf")) { if (!o_tpf) o_tpf = (decltype(o_tpf))r; return (g_style.load() || g_flat.load() || g_testtex.load()) ? (void *)h_tpf : r; }
    if (g_near.load() || g_style.load() || g_flat.load() || g_testtex.load()) {
        if (!strcmp(n, "glActiveTexture")) { if (!o_act) o_act = (decltype(o_act))r; return (void *)h_act; }
        if (!strcmp(n, "glBindTexture")) { if (!o_bind) o_bind = (decltype(o_bind))r; return (void *)h_bind; }
        if (!strcmp(n, "glDeleteTextures")) { if (!o_del) o_del = (decltype(o_del))r; return (void *)h_del; }
        if (!strcmp(n, "glSamplerParameteri")) { if (!o_smp) o_smp = (decltype(o_smp))r; return (void *)h_smp; }
    }
    if (!strcmp(n, "glTexStorage2D")) { if (!o_st) o_st = (decltype(o_st))r; return (void *)h_st; }
    if (!strcmp(n, "eglSwapBuffers")) { if (!o_swap) o_swap = (decltype(o_swap))r; return (void *)h_swap; }
    if (!strcmp(n, "eglCreateWindowSurface")) { if (!o_cws) o_cws = (decltype(o_cws))r; return (void *)h_cws; }
    if (!strcmp(n, "ANativeWindow_setBuffersGeometry")) { if (!o_sbg) o_sbg = (decltype(o_sbg))r; return (void *)h_sbg; }
    if (g_fz.load()) {
        if (!strcmp(n, "glCompileShader")) { if (!o_comp) o_comp = (decltype(o_comp))r; return (void *)h_comp; }
        if (!strcmp(n, "glLinkProgram")) { if (!o_link) o_link = (decltype(o_link))r; return (void *)h_link; }
        if (!strcmp(n, "glProgramBinary")) { if (!o_pbin) o_pbin = (decltype(o_pbin))r; return (void *)h_pbin; }
        if (!strcmp(n, "glBufferData")) { if (!o_buf) o_buf = (decltype(o_buf))r; return (void *)h_buf; }
        if (!strcmp(n, "glTexImage2D")) { if (!o_ti2) o_ti2 = (decltype(o_ti2))r; return (void *)h_ti2; }
        if (!strcmp(n, "glCompressedTexImage2D")) { if (!o_cti2) o_cti2 = (decltype(o_cti2))r; return (void *)h_cti2; }
        if (!strcmp(n, "glTexSubImage2D")) { if (!o_tsi2) o_tsi2 = (decltype(o_tsi2))r; return (void *)h_tsi2; }
        if (!strcmp(n, "glCompressedTexSubImage2D")) { if (!o_ctsi2) o_ctsi2 = (decltype(o_ctsi2))r; return (void *)h_ctsi2; }
        if (!strcmp(n, "glTexStorage3D")) { if (!o_ts3) o_ts3 = (decltype(o_ts3))r; return (void *)h_ts3; }
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

static void load_cfg() {
    FILE *f = fopen(CFGF, "r");
    if (f) {
        int v = 2;
        if (fscanf(f, "%d", &v) == 1) g_bias = std::max(0, std::min(v, 6));
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
    f = fopen(FZF, "r");
    if (f) {
        int v = 1;
        if (fscanf(f, "%d", &v) == 1) g_fz = v ? 1 : 0;
        fclose(f);
    } else {
        f = fopen(FZF, "w");
        if (f) { fprintf(f, "1\n"); fclose(f); }
    }
    f = fopen(THF, "r");
    if (f) {
        int v = 1;
        if (fscanf(f, "%d", &v) == 1) g_th = v ? 1 : 0;
        fclose(f);
    } else {
        f = fopen(THF, "w");
        if (f) { fprintf(f, "1\n"); fclose(f); }
    }
    f = fopen(NEARF, "r");
    if (f) {
        int v = 1;
        if (fscanf(f, "%d", &v) == 1) g_near = v ? 1 : 0;
        fclose(f);
    } else {
        f = fopen(NEARF, "w");
        if (f) { fprintf(f, "1\n"); fclose(f); }
    }
    f = fopen(BLKF, "r");
    if (f) {
        int v = 16;
        if (fscanf(f, "%d", &v) == 1) g_blk = (v >= 4 && v <= 256) ? v : 0;
        fclose(f);
    } else {
        f = fopen(BLKF, "w");
        if (f) { fprintf(f, "16\n"); fclose(f); }
    }
    f = fopen(MINF, "r");
    if (f) {
        int v = 64;
        if (fscanf(f, "%d", &v) == 1) g_minsz = std::max(64, std::min(v, 2048));
        fclose(f);
    } else {
        f = fopen(MINF, "w");
        if (f) { fprintf(f, "64\n"); fclose(f); }
    }
    f = fopen(DLYF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_delay = std::max(0, std::min(v, 120));
        fclose(f);
    } else {
        f = fopen(DLYF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    g_skip.clear();
    f = fopen(SKIPF, "r");
    if (f) {
        char ln[96];
        while (g_skip.size() < 128 && fgets(ln, sizeof ln, f)) {
            int a = 0, b = 0, c = 0;
            int n = sscanf(ln, "%d %d %d", &a, &b, &c);
            if (n == 2) g_skip.push_back({a, b, 0});
            else if (n == 3) g_skip.push_back({a, b, c});
        }
        fclose(f);
    }
    f = fopen(STYF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_style = std::max(0, std::min(v, 4));
        fclose(f);
    } else {
        f = fopen(STYF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    f = fopen(SEDF, "r");
    if (f) {
        int v = 1;
        if (fscanf(f, "%d", &v) == 1) g_sedge = v ? 1 : 0;
        fclose(f);
    } else {
        f = fopen(SEDF, "w");
        if (f) { fprintf(f, "1\n"); fclose(f); }
    }
    f = fopen(FLATF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_flat = std::max(0, std::min(v, 4));
        fclose(f);
    } else {
        f = fopen(FLATF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    f = fopen(TTEXF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_testtex = v ? 1 : 0;
        fclose(f);
    } else {
        f = fopen(TTEXF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    f = fopen(TTSZF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_ttsize = std::max(-1, std::min(v, 4));
        fclose(f);
    } else {
        f = fopen(TTSZF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    f = fopen(TTAXF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_ttax = v ? 1 : 0;
        fclose(f);
    } else {
        f = fopen(TTAXF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    f = fopen(TTSMF, "r");
    if (f) {
        int v = 0;
        if (fscanf(f, "%d", &v) == 1) g_ttsm = std::max(0, std::min(v, 1));
        fclose(f);
    } else {
        f = fopen(TTSMF, "w");
        if (f) { fprintf(f, "0\n"); fclose(f); }
    }
    g_tstart = now_ns();
    f = fopen(COREF, "r");
    if (f) {
        int v = 1;
        if (fscanf(f, "%d", &v) == 1) g_core = v ? 1 : 0;
        fclose(f);
    } else {
        f = fopen(COREF, "w");
        if (f) { fprintf(f, "1\n"); fclose(f); }
    }
}

/* mode.txt: 0 = idle | 1 = thread only | 2 = thread + find libunity (no hooks) | 3 = full hooks (polling) | 4 = hook at libunity load (no poll thread) */
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

/* Register all hooks on one libunity mapping */
static void reg_hooks(dev_t dv, ino_t in) {
    lsplt::RegisterHook(dv, in, "dlsym", (void *)h_dlsym, (void **)&o_dlsym);
    lsplt::RegisterHook(dv, in, "eglGetProcAddress", (void *)h_egl, (void **)&o_egl);
    lsplt::RegisterHook(dv, in, "glTexStorage2D", (void *)h_st, (void **)&o_st);
    if (g_style.load() || g_flat.load() || g_testtex.load()) lsplt::RegisterHook(dv, in, "glTexParameterf", (void *)h_tpf, (void **)&o_tpf);
    if (g_near.load() || g_style.load() || g_flat.load() || g_testtex.load()) {
        lsplt::RegisterHook(dv, in, "glTexParameteri", (void *)h_tp, (void **)&real_tp);
        lsplt::RegisterHook(dv, in, "glActiveTexture", (void *)h_act, (void **)&o_act);
        lsplt::RegisterHook(dv, in, "glBindTexture", (void *)h_bind, (void **)&o_bind);
        lsplt::RegisterHook(dv, in, "glDeleteTextures", (void *)h_del, (void **)&o_del);
        lsplt::RegisterHook(dv, in, "glSamplerParameteri", (void *)h_smp, (void **)&o_smp);
    }
    lsplt::RegisterHook(dv, in, "eglSwapBuffers", (void *)h_swap, (void **)&o_swap);
    lsplt::RegisterHook(dv, in, "eglCreateWindowSurface", (void *)h_cws, (void **)&o_cws);
    lsplt::RegisterHook(dv, in, "ANativeWindow_setBuffersGeometry", (void *)h_sbg, (void **)&o_sbg);
    if (g_fz.load()) {
        lsplt::RegisterHook(dv, in, "glCompileShader", (void *)h_comp, (void **)&o_comp);
        lsplt::RegisterHook(dv, in, "glLinkProgram", (void *)h_link, (void **)&o_link);
        lsplt::RegisterHook(dv, in, "glProgramBinary", (void *)h_pbin, (void **)&o_pbin);
        lsplt::RegisterHook(dv, in, "glBufferData", (void *)h_buf, (void **)&o_buf);
        lsplt::RegisterHook(dv, in, "glTexImage2D", (void *)h_ti2, (void **)&o_ti2);
        lsplt::RegisterHook(dv, in, "glCompressedTexImage2D", (void *)h_cti2, (void **)&o_cti2);
        lsplt::RegisterHook(dv, in, "glTexSubImage2D", (void *)h_tsi2, (void **)&o_tsi2);
        lsplt::RegisterHook(dv, in, "glCompressedTexSubImage2D", (void *)h_ctsi2, (void **)&o_ctsi2);
        lsplt::RegisterHook(dv, in, "glTexStorage3D", (void *)h_ts3, (void **)&o_ts3);
    }
}

/* ---------- Mode 4: hook libunity only briefly right after it loads ---------- */
static std::atomic<bool> g_hooked{false};
static void *(*o_adle)(const char *, int, const android_dlextinfo *) = nullptr;

static void install_unity_hooks() {
    if (g_hooked.exchange(true)) return;
    for (auto &m : lsplt::MapInfo::Scan()) {
        const std::string &p = m.path;
        if (p.size() < 12 || p.compare(p.size() - 12, 12, "/libunity.so") != 0) continue;
        reg_hooks(m.dev, m.inode);
        L("pre-commit (unity)\n");
        bool ok = lsplt::CommitHook();
        L("commit=%d (unity)\n", (int)ok);
        return;
    }
    L("libunity not found in maps\n");
    g_hooked = false;
}

static void *h_adle(const char *path, int flags, const android_dlextinfo *info) {
    void *r = o_adle ? o_adle(path, flags, info) : android_dlopen_ext(path, flags, info);
    if (r && path) {
        size_t n = strlen(path);
        if (n >= 11 && !strcmp(path + n - 11, "libunity.so")) {
            L("dlopen libunity done: %s\n", path);
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
    L("libnativeloader not found\n");
}

static void run(int mode) {
    load_cfg();
    std::set<std::pair<dev_t, ino_t>> done;
    std::map<std::pair<dev_t, ino_t>, int64_t> first;   /* when libunity was first seen */
    L("=== start v23 mode=%d bias=%d fpscap=%d scale=%d fz=%d th=%d core=%d near=%d blk=%d min=%d delay=%d skip=%d style=%d sedge=%d flat=%d testtex=%d builtin=%d ===\n", mode, g_bias.load(), g_fps.load(), g_scale.load(), g_fz.load(), g_th.load(), g_core.load(), g_near.load(), g_blk.load(), g_minsz.load(), g_delay.load(), (int)g_skip.size(), g_style.load(), g_sedge.load(), g_flat.load(), g_testtex.load(), (int)(sizeof g_builtin / sizeof g_builtin[0]));
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
                if (now_ns() - it->second < 3000000000LL) continue;   /* wait until loading finishes */
                done.insert(key);
                if (mode == 2) { L("mode2: stable, no hooks\n"); continue; }
                reg_hooks(m.dev, m.inode);
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
            L("=== start v23 mode=4 bias=%d fpscap=%d scale=%d fz=%d th=%d core=%d near=%d blk=%d min=%d delay=%d skip=%d style=%d sedge=%d flat=%d testtex=%d builtin=%d ===\n", g_bias.load(), g_fps.load(), g_scale.load(), g_fz.load(), g_th.load(), g_core.load(), g_near.load(), g_blk.load(), g_minsz.load(), g_delay.load(), (int)g_skip.size(), g_style.load(), g_sedge.load(), g_flat.load(), g_testtex.load(), (int)(sizeof g_builtin / sizeof g_builtin[0]));
            hook_nativeloader();
            return;
        }
        if (mode >= 1) std::thread(run, mode).detach();
    }
};
REGISTER_ZYGISK_MODULE(M)
