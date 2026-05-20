#include "comms.h"
#include "kharon.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define NCCL_CK(call) do { \
  ncclResult_t s_ = (call); \
  if (s_ != ncclSuccess) { \
    fprintf(stderr, "nccl error %s at %s:%d -> %s\n", #call, __FILE__, __LINE__, \
            ncclGetErrorString(s_)); \
    exit(1); \
  } \
} while (0)

void comms_init(Comms *c) {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &c->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &c->nranks);
  int ngpu = 0;
  CK(cudaGetDeviceCount(&ngpu));
  c->device = c->rank % ngpu;          // one rank per GPU on the node
  CK(cudaSetDevice(c->device));

  ncclUniqueId id;
  if (c->rank == 0) NCCL_CK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NCCL_CK(ncclCommInitRank(&comm, c->nranks, id, c->rank));
  c->comm = comm;
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  c->stream = s;
}

void comms_finalize(Comms *c) {
  ncclCommDestroy((ncclComm_t)c->comm);
  cudaStreamDestroy((cudaStream_t)c->stream);
  MPI_Finalize();
}

void comms_allreduce(Comms *c, float *buf, long n) {
  NCCL_CK(ncclAllReduce(buf, buf, n, ncclFloat, ncclSum, (ncclComm_t)c->comm, 0));
}
void comms_allreduce_bf16(Comms *c, void *buf, long n) {
  NCCL_CK(ncclAllReduce(buf, buf, n, ncclBfloat16, ncclSum, (ncclComm_t)c->comm, 0));
}
void comms_allreduce_bf16_async(Comms *c, void *buf, long n) {
  NCCL_CK(ncclAllReduce(buf, buf, n, ncclBfloat16, ncclSum, (ncclComm_t)c->comm,
                        (cudaStream_t)c->stream));
}
void comms_wait(Comms *c) { CK(cudaStreamSynchronize((cudaStream_t)c->stream)); }

void comms_send_bf16(Comms *c, void *buf, long n, int peer) {
  NCCL_CK(ncclGroupStart());
  NCCL_CK(ncclSend(buf, n, ncclBfloat16, peer, (ncclComm_t)c->comm, 0));
  NCCL_CK(ncclGroupEnd());
}
void comms_recv_bf16(Comms *c, void *buf, long n, int peer) {
  NCCL_CK(ncclGroupStart());
  NCCL_CK(ncclRecv(buf, n, ncclBfloat16, peer, (ncclComm_t)c->comm, 0));
  NCCL_CK(ncclGroupEnd());
}
void comms_sendrecv_bf16(Comms *c, void *sbuf, long sn, int speer,
                         void *rbuf, long rn, int rpeer) {
  NCCL_CK(ncclGroupStart());
  NCCL_CK(ncclSend(sbuf, sn, ncclBfloat16, speer, (ncclComm_t)c->comm, 0));
  NCCL_CK(ncclRecv(rbuf, rn, ncclBfloat16, rpeer, (ncclComm_t)c->comm, 0));
  NCCL_CK(ncclGroupEnd());
}
void comms_sync_default(Comms *c) { (void)c; CK(cudaStreamSynchronize(0)); }
