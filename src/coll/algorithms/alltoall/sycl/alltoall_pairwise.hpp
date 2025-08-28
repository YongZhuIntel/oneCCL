
// assume GPU RDMA is enabled
// does not work for in place
ccl::event alltoall_sycl_pairwise_rdma(sycl::queue& q,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       ccl_comm* comm,
                                       ccl_stream* global_stream,
                                       const ccl::vector_class<ccl::event>& deps,
                                       bool& done) {
    ccl_comm* node_comm = comm->get_node_comm().get();
    uint32_t world = comm->size();
    uint32_t rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();
    int ep_idx = 0;
    int pof2 = is_pof2(world);

    done = true;

#if 0
    sycl::queue scaleout_q =
        sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
#else
    sycl::queue scaleout_q = q;
#endif

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    std::vector<int> local_ar(world);
    int first_node_rank = world;
    auto rank_info = topo_manager.get_filtered_rank_info_vec(topo_manager.get_host_idx());
    for (int rank_idx = 0; rank_idx < world; rank_idx++) {
        local_ar[rank_idx] = 0;
        for (auto& local_info : rank_info) {
            if (rank_idx == local_info.rank) {
                local_ar[rank_idx] = 1;
                if (rank_idx < first_node_rank)
                    first_node_rank = rank_idx;
            }
        }
    }

    /* scale up */
    if (node_comm->size() > 1) {
        sycl::queue scaleup_q = q;
        ccl::event up_e = alltoall_sycl_single_node(scaleup_q,
                                                    (char*)send_buf + first_node_rank * size,
                                                    (char*)recv_buf + first_node_rank * size,
                                                    count,
                                                    dtype,
                                                    node_comm,
                                                    global_stream,
                                                    deps,
                                                    done);
    }
    else {
        // copy self
        sycl::event copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy((char*)recv_buf + rank * size, (char*)send_buf + rank * size, size);
        });
    }

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag =
        atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    /* Do the pairwise exchanges */
    sycl::event out_e = scaleout_q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.host_task([=]() {
            int src, dst;
            for (int i = 1; i < world; i++) {
                if (pof2) {
                    src = dst = rank ^ i;
                }
                else {
                    src = (rank - i + world) % world;
                    dst = (rank + i) % world;
                }

                atl_req_t send_req, recv_req;
                if (local_ar[dst] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->send(
                        ep_idx, (char*)send_buf + dst * size, size, dst, tag + i, send_req));
                }
                if (local_ar[src] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->recv(
                        ep_idx, (char*)recv_buf + src * size, size, src, tag + i, recv_req));
                }
                if (local_ar[dst] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
                }
                if (local_ar[src] == 0) {
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
                }
            }
        });
    });

    dep_events.clear();
    dep_events.push_back(std::move(out_e));
    sycl::event sycl_e = submit_wait_on_events(q, dep_events);

    return ccl::event::create_from_native(sycl_e);
}
