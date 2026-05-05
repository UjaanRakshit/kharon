#include "refio.h"
#include "kharon.h"
#include <string.h>

// Format (little-endian): "CHRN", i32 version, 6*i32 config, 5*f32 opt,
// then records { i32 name_len, name, i32 ndim, ndim*i32 dims, i32 dtype, raw },
// terminated by a record with name_len == -1.

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

RefFile *ref_load(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) DIE("ref_load: cannot open %s", path);

  char magic[4];
  if (!rd(f, magic, 4) || memcmp(magic, "CHRN", 4) != 0) DIE("ref_load: bad magic");
  int version, cfg[6];
  float opt[5];
  if (!rd(f, &version, 4)) DIE("ref_load: short read (version)");
  if (!rd(f, cfg, sizeof(cfg))) DIE("ref_load: short read (cfg)");
  if (!rd(f, opt, sizeof(opt))) DIE("ref_load: short read (opt)");

  RefFile *r = (RefFile *)calloc(1, sizeof(RefFile));
  r->n_layer = cfg[0]; r->d_model = cfg[1]; r->n_head = cfg[2];
  r->vocab = cfg[3]; r->seq = cfg[4]; r->batch = cfg[5];
  r->lr = opt[0]; r->beta1 = opt[1]; r->beta2 = opt[2]; r->eps = opt[3]; r->wd = opt[4];

  // Two passes would need seeking; instead grow arrays as we go.
  int cap = 16;
  r->t = (RefTensor *)malloc(cap * sizeof(RefTensor));
  size_t blob_cap = 1 << 20, blob_off = 0;
  r->blob = malloc(blob_cap);

  for (;;) {
    int name_len;
    if (!rd(f, &name_len, 4)) DIE("ref_load: short read (name_len)");
    if (name_len < 0) break;
    if (r->nt == cap) { cap *= 2; r->t = (RefTensor *)realloc(r->t, cap * sizeof(RefTensor)); }
    RefTensor *t = &r->t[r->nt];
    memset(t, 0, sizeof(*t));
    if (name_len >= (int)sizeof(t->name)) DIE("ref_load: name too long");
    if (!rd(f, t->name, name_len)) DIE("ref_load: short read (name)");
    t->name[name_len] = 0;
    if (!rd(f, &t->ndim, 4)) DIE("ref_load: short read (ndim)");
    if (t->ndim < 0 || t->ndim > 4) DIE("ref_load: bad ndim %d", t->ndim);
    t->count = 1;
    for (int i = 0; i < t->ndim; i++) {
      if (!rd(f, &t->dims[i], 4)) DIE("ref_load: short read (dim)");
      t->count *= t->dims[i];
    }
    if (!rd(f, &t->dtype, 4)) DIE("ref_load: short read (dtype)");
    size_t bytes = (size_t)t->count * 4;  // both f32 and i32 are 4 bytes
    while (blob_off + bytes > blob_cap) {
      blob_cap *= 2;
      r->blob = realloc(r->blob, blob_cap);
    }
    t->data = (char *)r->blob + blob_off;
    if (!rd(f, t->data, bytes)) DIE("ref_load: short read (%s data)", t->name);
    blob_off += bytes;
    r->nt++;
  }
  fclose(f);

  // blob may have been realloc'd; repoint data offsets are stable only if no
  // realloc moved it after we stored pointers. Recompute pointers from offsets.
  size_t off = 0;
  for (int i = 0; i < r->nt; i++) {
    r->t[i].data = (char *)r->blob + off;
    off += (size_t)r->t[i].count * 4;
  }
  return r;
}

RefTensor *ref_get(RefFile *r, const char *name) {
  for (int i = 0; i < r->nt; i++)
    if (strcmp(r->t[i].name, name) == 0) return &r->t[i];
  return NULL;
}

float *ref_f32(RefFile *r, const char *name) {
  RefTensor *t = ref_get(r, name);
  if (!t) DIE("ref_f32: missing '%s'", name);
  if (t->dtype != 0) DIE("ref_f32: '%s' is not f32", name);
  return (float *)t->data;
}

int *ref_i32(RefFile *r, const char *name) {
  RefTensor *t = ref_get(r, name);
  if (!t) DIE("ref_i32: missing '%s'", name);
  if (t->dtype != 1) DIE("ref_i32: '%s' is not i32", name);
  return (int *)t->data;
}

void ref_free(RefFile *r) {
  if (!r) return;
  free(r->blob);
  free(r->t);
  free(r);
}
