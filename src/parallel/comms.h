#ifndef KHARON_COMMS_H
#define KHARON_COMMS_H

// Thin NCCL wrapper for tensor parallelism over PCIe (no NVLink). Process-per-GPU,
// bootstrapped with MPI. comm/stream are opaque (void*) to keep the C side clean.
typedef struct {
  int rank, nranks, device;
  void *comm;     // ncclComm_t
  void *stream;   // cudaStream_t for overlapped collectives
} Comms;

#ifdef __cplusplus
extern "C" {
#endif

void comms_init(Comms *c);
void comms_finalize(Comms *c);
// in-place sum-allreduce on the default stream (ordered with compute)
void comms_allreduce(Comms *c, float *buf, long n);
void comms_allreduce_bf16(Comms *c, void *buf, long n);
// async variant on c->stream (for overlap); pair with comms_wait
void comms_allreduce_bf16_async(Comms *c, void *buf, long n);
void comms_wait(Comms *c);

#ifdef __cplusplus
}
#endif

#endif
