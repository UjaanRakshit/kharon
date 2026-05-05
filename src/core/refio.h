#ifndef KHARON_REFIO_H
#define KHARON_REFIO_H

// Reader for the CHRN reference dump written by tests/gen_reference.py.
// Holds all records in host memory; the C stack loads weights/inputs from here
// and compares its outputs against the expected records.

typedef struct {
  char name[64];
  int ndim;
  int dims[4];
  int dtype;     // 0 = f32, 1 = i32
  long count;    // element count
  void *data;    // host pointer into the blob
} RefTensor;

typedef struct {
  int n_layer, d_model, n_head, vocab, seq, batch;
  float lr, beta1, beta2, eps, wd;
  RefTensor *t;
  int nt;
  void *blob;    // owns all record data
} RefFile;

#ifdef __cplusplus
extern "C" {
#endif

RefFile   *ref_load(const char *path);
RefTensor *ref_get(RefFile *r, const char *name);   // NULL if absent
float     *ref_f32(RefFile *r, const char *name);   // dies if absent/wrong dtype
int       *ref_i32(RefFile *r, const char *name);
void       ref_free(RefFile *r);

#ifdef __cplusplus
}
#endif

#endif
