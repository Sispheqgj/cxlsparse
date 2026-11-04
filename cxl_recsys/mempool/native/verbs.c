/* RDMA libibverbs shim: Python owns allocation policy and all RPC logic.
 * No sockets or RPC payload fallback. One RC QP and registered region/session.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct endpoint {
    uint64_t addr;
    uint64_t size;
    uint32_t rkey;
    uint32_t qpn;
    uint32_t psn;
    uint32_t lid;
    uint32_t mtu;
    uint8_t gid[16];
    uint32_t max_rd_atomic;
    uint32_t max_dest_rd_atomic;
};

struct region {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    void *buf;
    size_t size;
    int port;
    int gid_index;
    uint8_t max_rd_atomic;
    uint8_t max_dest_rd_atomic;
};

static _Thread_local char error[256];

const char *cr_error(void) {
    return error;
}

static int fail(const char *where, int code) {
    snprintf(error, sizeof(error), "%s: %s (%d)", where, strerror(code), code);
    return -1;
}

void cr_close(struct region *region) {
    if (!region) {
        return;
    }
    if (region->qp) {
        ibv_destroy_qp(region->qp);
    }
    if (region->cq) {
        ibv_destroy_cq(region->cq);
    }
    if (region->mr) {
        ibv_dereg_mr(region->mr);
    }
    if (region->pd) {
        ibv_dealloc_pd(region->pd);
    }
    if (region->ctx) {
        ibv_close_device(region->ctx);
    }
    free(region->buf);
    free(region);
}

struct region *cr_open(
    const char *device,
    int port,
    int gid_index,
    size_t size,
    struct endpoint *local_endpoint
) {
    struct region *region = calloc(1, sizeof(*region));
    if (!region) {
        fail("calloc", ENOMEM);
        return NULL;
    }
    region->port = port;
    region->gid_index = gid_index;
    region->size = size;
    int device_count = 0;
    struct ibv_device **list = ibv_get_device_list(&device_count);
    if (!list) {
        fail("ibv_get_device_list", errno);
        goto cleanup;
    }
    for (int i = 0; i < device_count; i++) {
        if (!strcmp(ibv_get_device_name(list[i]), device)) {
            region->ctx = ibv_open_device(list[i]);
            break;
        }
    }
    ibv_free_device_list(list);
    if (!region->ctx) {
        fail("RDMA device unavailable", ENODEV);
        goto cleanup;
    }
    struct ibv_device_attr device_attr;
    if (ibv_query_device(region->ctx, &device_attr)) {
        fail("ibv_query_device", errno);
        goto cleanup;
    }
    if (device_attr.max_qp_init_rd_atom <= 0 || device_attr.max_qp_rd_atom <= 0) {
        fail("RDMA READ credits unavailable", ENOTSUP);
        goto cleanup;
    }
    /* QP attributes are uint8_t. Advertise the usable local capabilities so
     * each direction can consume the maximum supported by BOTH endpoints.
     */
    region->max_rd_atomic = device_attr.max_qp_init_rd_atom > UINT8_MAX
                                ? UINT8_MAX
                                : device_attr.max_qp_init_rd_atom;
    region->max_dest_rd_atomic =
        device_attr.max_qp_rd_atom > UINT8_MAX ? UINT8_MAX : device_attr.max_qp_rd_atom;
    struct ibv_port_attr port_attr;
    if (ibv_query_port(region->ctx, port, &port_attr)) {
        fail("ibv_query_port", errno);
        goto cleanup;
    }
    if (port_attr.state != IBV_PORT_ACTIVE) {
        fail("RDMA PORT_DOWN (no TCP fallback)", ENETDOWN);
        goto cleanup;
    }
    union ibv_gid gid;
    if (ibv_query_gid(region->ctx, port, gid_index, &gid)) {
        fail("ibv_query_gid", errno);
        goto cleanup;
    }
    // Register the region before creating the connection resources.
    region->pd = ibv_alloc_pd(region->ctx);
    if (!region->pd) {
        fail("ibv_alloc_pd", errno);
        goto cleanup;
    }
    int result;
    /* Pool servers own DRAM. GPU clients need only a PD/QP here and register
     * each caller-owned CUDA tensor separately; no CPU staging MR is allocated.
     */
    if (size) {
        result = posix_memalign(&region->buf, 4096, size);
        if (result) {
            fail("posix_memalign", result);
            goto cleanup;
        }
        memset(region->buf, 0, size);
        region->mr = ibv_reg_mr(
            region->pd,
            region->buf,
            size,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
        );
        if (!region->mr) {
            fail("ibv_reg_mr", errno);
            goto cleanup;
        }
    }
    region->cq = ibv_create_cq(region->ctx, 64, NULL, NULL, 0);
    if (!region->cq) {
        fail("ibv_create_cq", errno);
        goto cleanup;
    }
    // The queue pair transitions from INIT to RTR/RTS after endpoint exchange.
    struct ibv_qp_init_attr qp_init_attr = { 0 };
    qp_init_attr.send_cq = region->cq;
    qp_init_attr.recv_cq = region->cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 64;
    qp_init_attr.cap.max_recv_wr = 1;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    region->qp = ibv_create_qp(region->pd, &qp_init_attr);
    if (!region->qp) {
        fail("ibv_create_qp", errno);
        goto cleanup;
    }
    struct ibv_qp_attr qp_attr = { 0 };
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = port;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    result = ibv_modify_qp(
        region->qp,
        &qp_attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS
    );
    if (result) {
        fail("QP INIT", result);
        goto cleanup;
    }
    memset(local_endpoint, 0, sizeof(*local_endpoint));
    local_endpoint->addr = (uintptr_t)region->buf;
    local_endpoint->size = size;
    local_endpoint->rkey = region->mr ? region->mr->rkey : 0;
    local_endpoint->qpn = region->qp->qp_num;
    local_endpoint->psn = 0;
    local_endpoint->lid = port_attr.lid;
    local_endpoint->mtu = port_attr.active_mtu;
    memcpy(local_endpoint->gid, gid.raw, 16);
    local_endpoint->max_rd_atomic = region->max_rd_atomic;
    local_endpoint->max_dest_rd_atomic = region->max_dest_rd_atomic;
    return region;

cleanup:
    cr_close(region);
    return NULL;
}

int cr_connect(struct region *region, const struct endpoint *peer) {
    if (!peer->max_rd_atomic || !peer->max_dest_rd_atomic) {
        return fail("Peer must advertise RDMA READ capabilities", EINVAL);
    }
    const uint8_t initiator = region->max_rd_atomic < peer->max_dest_rd_atomic
                                  ? region->max_rd_atomic
                                  : peer->max_dest_rd_atomic;
    const uint8_t responder = region->max_dest_rd_atomic < peer->max_rd_atomic
                                  ? region->max_dest_rd_atomic
                                  : peer->max_rd_atomic;
    struct ibv_port_attr port_attr;
    if (ibv_query_port(region->ctx, region->port, &port_attr)) {
        return fail("query_port", errno);
    }
    struct ibv_qp_attr qp_attr = { 0 };
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu =
        peer->mtu < port_attr.active_mtu ? peer->mtu : port_attr.active_mtu;
    qp_attr.dest_qp_num = peer->qpn;
    qp_attr.rq_psn = peer->psn;
    qp_attr.max_dest_rd_atomic = responder;
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.dlid = peer->lid;
    qp_attr.ah_attr.port_num = region->port;
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.sgid_index = region->gid_index;
    qp_attr.ah_attr.grh.hop_limit = 64;
    memcpy(qp_attr.ah_attr.grh.dgid.raw, peer->gid, 16);
    int result = ibv_modify_qp(
        region->qp,
        &qp_attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER
    );
    if (result) {
        return fail("QP RTR", result);
    }
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 3;
    qp_attr.rnr_retry = 3;
    qp_attr.sq_psn = 0;
    qp_attr.max_rd_atomic = initiator;
    result = ibv_modify_qp(
        region->qp,
        &qp_attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC
    );
    if (result) {
        return fail("QP RTS", result);
    }
    /* Verify the actual QP, not just the requested values used in logs. */
    struct ibv_qp_init_attr actual_init = { 0 };
    result = ibv_query_qp(
        region->qp,
        &qp_attr,
        IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_MAX_DEST_RD_ATOMIC,
        &actual_init
    );
    if (result) {
        return fail("query QP READ credits", result);
    }
    if (qp_attr.max_rd_atomic != initiator || qp_attr.max_dest_rd_atomic != responder) {
        return fail("QP READ credits differ from negotiated maximum", EIO);
    }
    return 0;
}

static uint64_t now_ms(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (uint64_t)timestamp.tv_sec * 1000 + timestamp.tv_nsec / 1000000;
}

int cr_io(
    struct region *region,
    const struct endpoint *peer,
    uint64_t offset,
    size_t bytes,
    int write,
    int timeout_ms
) {
    if (!region->mr || !region->qp || bytes > region->size || offset > peer->size ||
        bytes > peer->size - offset || bytes > UINT32_MAX) {
        return fail("RDMA bounds", EINVAL);
    }
    struct ibv_sge scatter_entry = { 0 };
    scatter_entry.addr = (uintptr_t)region->buf;
    scatter_entry.length = bytes;
    scatter_entry.lkey = region->mr->lkey;

    struct ibv_send_wr request = { 0 };
    request.wr_id = 1;
    request.sg_list = &scatter_entry;
    request.num_sge = 1;
    request.opcode = write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
    request.send_flags = IBV_SEND_SIGNALED;
    request.wr.rdma.remote_addr = peer->addr + offset;
    request.wr.rdma.rkey = peer->rkey;
    struct ibv_send_wr *bad_request;
    int result = ibv_post_send(region->qp, &request, &bad_request);
    if (result) {
        return fail("ibv_post_send", result);
    }
    // One operation is outstanding at a time; do not reuse a timed-out QP.
    uint64_t deadline = now_ms() + timeout_ms;
    for (;;) {
        struct ibv_wc completion;
        result = ibv_poll_cq(region->cq, 1, &completion);
        if (result < 0) {
            return fail("ibv_poll_cq", EIO);
        }
        if (result > 0) {
            if (completion.status != IBV_WC_SUCCESS) {
                snprintf(
                    error,
                    sizeof(error),
                    "RDMA completion: %s vendor=%u",
                    ibv_wc_status_str(completion.status),
                    completion.vendor_err
                );
                return -1;
            }
            return 0;
        }
        if (now_ms() >= deadline) {
            return fail("RDMA completion timeout (QP must be discarded)", ETIMEDOUT);
        }
        struct timespec pause = { .tv_nsec = 10000 };
        nanosleep(&pause, NULL);
    }
}

/* Register an exported CUDA allocation. The caller retains both the tensor and
 * DMA-BUF fd until all work completes and this MR has been deregistered.
 * The NIC uses this MR directly: no CPU-addressable payload buffer is involved.
 */
struct ibv_mr *
cr_register_gpu(struct region *region, uint64_t address, size_t bytes, int dmabuf_fd) {
    struct ibv_mr *mr = ibv_reg_dmabuf_mr(
        region->pd, 0, bytes, address, dmabuf_fd, IBV_ACCESS_LOCAL_WRITE
    );
    if (!mr) {
        fail("ibv_reg_dmabuf_mr (GPUDirect required; no staging fallback)", errno);
    }
    return mr;
}

int cr_deregister_gpu(struct ibv_mr *mr) {
    int result = ibv_dereg_mr(mr);
    return result ? fail("ibv_dereg_mr GPU", result) : 0;
}

static void abort_qp(struct region *region) {
    /* Destroy before releasing GPU memory, including timeout/error paths. */
    if (region->qp) {
        ibv_destroy_qp(region->qp);
        region->qp = NULL;
    }
}

static uint64_t now_ns(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (uint64_t)timestamp.tv_sec * 1000000000ULL + timestamp.tv_nsec;
}

static int gpu_io(
    struct region *region,
    const struct endpoint *peer,
    struct ibv_mr *mr,
    uint64_t gpu_iova,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    const uint64_t *remote_slots,
    const uint64_t *local_rows,
    size_t count,
    size_t token_bytes,
    int write,
    int timeout_ms,
    uint64_t *operations,
    uint64_t *transport_ns
) {
    uint64_t started_ns = 0;
    if (transport_ns) {
        *transport_ns = 0;
    }
    if (!region->qp || !token_bytes || token_bytes > UINT32_MAX ||
        tensor_offset > mr->length || tensor_bytes > mr->length - tensor_offset) {
        return fail("GPUDirect tensor bounds or closed QP", EINVAL);
    }
    size_t next = 0;
    size_t outstanding = 0;
    uint64_t deadline = now_ms() + timeout_ms;
    *operations = 0;
    while (next < count || outstanding) {
        while (next < count && outstanding < 32) {
            size_t end = next + 1;
            /* Merge only spans contiguous in BOTH remote KV and compact HBM.
             * Never read holes or overwrite HBM hits in the compact buffer.
             */
            while (end < count && end - next < UINT32_MAX / token_bytes &&
                   remote_slots[end] == remote_slots[end - 1] + 1 &&
                   local_rows[end] == local_rows[end - 1] + 1) {
                end++;
            }
            size_t tokens = end - next;
            uint64_t remote_capacity = peer->size / token_bytes;
            uint64_t local_capacity = tensor_bytes / token_bytes;
            if (remote_slots[next] > remote_capacity ||
                tokens > remote_capacity - remote_slots[next] ||
                local_rows[next] > local_capacity ||
                tokens > local_capacity - local_rows[next]) {
                fail("GPUDirect span bounds", EINVAL);
                goto failed;
            }
            struct ibv_sge sge = { 0 };
            /* A DMA-BUF MR's addr is its fd offset, NOT the registered IOVA. */
            sge.addr = gpu_iova + tensor_offset + local_rows[next] * token_bytes;
            sge.length = tokens * token_bytes;
            sge.lkey = mr->lkey;
            struct ibv_send_wr wr = { 0 };
            wr.wr_id = *operations + 1;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = peer->addr + remote_slots[next] * token_bytes;
            wr.wr.rdma.rkey = peer->rkey;
            struct ibv_send_wr *bad_wr;
            if (transport_ns && !*operations) {
                started_ns = now_ns();
            }
            int result = ibv_post_send(region->qp, &wr, &bad_wr);
            if (result) {
                fail("ibv_post_send GPU", result);
                goto failed;
            }
            outstanding++;
            (*operations)++;
            next = end;
        }
        struct ibv_wc completions[32];
        int completed = ibv_poll_cq(region->cq, 32, completions);
        if (completed < 0) {
            fail("ibv_poll_cq GPU", EIO);
            goto failed;
        }
        for (int i = 0; i < completed; i++) {
            if (completions[i].status != IBV_WC_SUCCESS) {
                snprintf(
                    error,
                    sizeof(error),
                    "GPUDirect completion: %s vendor=%u",
                    ibv_wc_status_str(completions[i].status),
                    completions[i].vendor_err
                );
                goto failed;
            }
        }
        outstanding -= completed;
        if (completed) {
            deadline = now_ms() + timeout_ms;
        } else if (now_ms() >= deadline) {
            fail("GPUDirect completion timeout", ETIMEDOUT);
            goto failed;
        }
    }
    if (transport_ns && *operations) {
        *transport_ns = now_ns() - started_ns;
    }
    return 0;

failed:
    abort_qp(region);
    return -1;
}

/* Production and profiling use exactly the same coalescing/posting loop. */
int cr_gpu_io(
    struct region *region,
    const struct endpoint *peer,
    struct ibv_mr *mr,
    uint64_t gpu_iova,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    const uint64_t *remote_slots,
    const uint64_t *local_rows,
    size_t count,
    size_t token_bytes,
    int write,
    int timeout_ms,
    uint64_t *operations
) {
    return gpu_io(
        region,
        peer,
        mr,
        gpu_iova,
        tensor_offset,
        tensor_bytes,
        remote_slots,
        local_rows,
        count,
        token_bytes,
        write,
        timeout_ms,
        operations,
        NULL
    );
}

int cr_gpu_io_timed(
    struct region *region,
    const struct endpoint *peer,
    struct ibv_mr *mr,
    uint64_t gpu_iova,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    const uint64_t *remote_slots,
    const uint64_t *local_rows,
    size_t count,
    size_t token_bytes,
    int write,
    int timeout_ms,
    uint64_t *operations,
    uint64_t *transport_ns
) {
    return gpu_io(
        region,
        peer,
        mr,
        gpu_iova,
        tensor_offset,
        tensor_bytes,
        remote_slots,
        local_rows,
        count,
        token_bytes,
        write,
        timeout_ms,
        operations,
        transport_ns
    );
}
