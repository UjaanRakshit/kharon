#ifndef KHARON_CKPT_H
#define KHARON_CKPT_H

#include "model.h"
#include <stdint.h>

// Serialize weights + AdamW moments + step + RNG state so a job killed at the
// 16h walltime resumes bit-exact. Format is self-describing (magic + config).
#ifdef __cplusplus
extern "C" {
#endif

void ckpt_save(const Model *m, uint64_t rng, const char *path);
void ckpt_load(Model *m, uint64_t *rng, const char *path);

#ifdef __cplusplus
}
#endif

#endif
