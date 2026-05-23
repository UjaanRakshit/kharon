#ifndef KHARON_COMMS_H
#define KHARON_COMMS_H

// Thin NCCL wrapper for tensor parallelism over PCIe (no NVLink). Process-per-GPU,
// bootstrapped with MPI. comm/stream are opaque (void*) to keep the C side clean.
typedef struct {
  int rank, nranks, device;
  void *comm;     // ncclComm_t (global)
  void *stream;   // cudaStream_t for overlapped collectives
  // 3D (TP x PP x DP) grid. rank = dp_rank*(pp_size*tp_size) + pp_stage*tp_size + tp_rank,
  // so TP pairs sit on adjacent ranks (closest PCIe), then PP, then DP outermost.
  // tp_comm = TP group within a stage; dp_comm = replicas sharing (pp_stage,tp_rank);
  // PP point-to-point uses the global comm, peer = rank +- tp_size (stays in the dp block).
  int tp_size, tp_rank, pp_stage, pp_size, dp_size, dp_rank;
  void *tp_comm, *dp_comm;
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
// point-to-point for pipeline parallelism (bf16), on the default stream.
void comms_send_bf16(Comms *c, void *buf, long n, int peer);
void comms_recv_bf16(Comms *c, void *buf, long n, int peer);
// fused send+recv in one NCCL group (deadlock-free for adjacent-stage exchange)
void comms_sendrecv_bf16(Comms *c, void *sbuf, long sn, int speer,
                         void *rbuf, long rn, int rpeer);
void comms_sync_default(Comms *c);   // sync the default stream (0)
// 2D grid: split the global comm into TP groups (ranks sharing a pp_stage).
void comms_init_grid(Comms *c, int tp_size);
void comms_tp_allreduce_bf16(Comms *c, void *buf, long n);   // all-reduce within TP group
// 3D grid: also split a DP group (replicas sharing pp_stage+tp_rank). dp_size==1 -> 2D.
void comms_init_grid3(Comms *c, int tp_size, int dp_size);
// ZeRO-1 DP collectives over fp32 grads/params (averaging redop for the scatter).
void comms_dp_reducescatter_f32(Comms *c, const float *send, float *recv, long recvcount);
void comms_dp_allgather_f32(Comms *c, const float *send, float *recv, long sendcount);

#ifdef __cplusplus
}
#endif

#endif
