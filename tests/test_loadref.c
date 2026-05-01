#include "refio.h"
#include <stdio.h>

// Sanity check: parse the reference dump and print config + a couple of values
// so we know the C side reads exactly what PyTorch wrote.

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  printf("config: L=%d d=%d h=%d vocab=%d seq=%d batch=%d\n",
         r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch);
  printf("opt: lr=%g b1=%g b2=%g eps=%g wd=%g\n", r->lr, r->beta1, r->beta2, r->eps, r->wd);
  printf("records: %d\n", r->nt);

  float loss = *ref_f32(r, "loss");
  printf("loss = %.6f\n", loss);

  RefTensor *wte = ref_get(r, "wte");
  double s = 0;
  float *w = (float *)wte->data;
  for (long i = 0; i < wte->count; i++) s += w[i];
  printf("wte: [%d,%d] sum=%.6f\n", wte->dims[0], wte->dims[1], s);

  int *idx = ref_i32(r, "input_ids");
  printf("input_ids[0..4] = %d %d %d %d %d\n", idx[0], idx[1], idx[2], idx[3], idx[4]);

  ref_free(r);
  return 0;
}
