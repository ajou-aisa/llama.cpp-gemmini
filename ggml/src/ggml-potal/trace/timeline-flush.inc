// Shared W4/W8 shutdown serialization; included inside potal::timeline.
bool flush() noexcept {
    if (!detail::g_on) {
        return false;
    }
    const char * format = std::getenv("POTAL_TIMELINE_FORMAT");
    const bool binary = format != nullptr && std::strcmp(format, "binary") == 0;
    if (format != nullptr && std::strcmp(format, "csv") != 0 && !binary) {
        std::fprintf(stderr, "potal timeline: POTAL_TIMELINE_FORMAT must be csv or binary\n");
        return false;
    }
    if (binary && !binary_trace::supported()) return false;
    state &                     s = st();
    std::lock_guard<std::mutex> lk(s.flush_mu);
    if (t_info != nullptr) {
        take_anchor(t_info, a_flush, 0);
    }
    const uint32_t                                         n = s.nthreads.load(std::memory_order_acquire);
    std::unordered_map<uint32_t, std::pair<uint8_t, uint8_t>> run_kind;
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      rn = ti->run_head.load(std::memory_order_relaxed);
        for (uint32_t j = 0; j < rn; ++j) {
            if (ti->runs[j].layer_id != kMissing || ti->runs[j].begin_anchor != kMissing) {
                run_kind[ti->runs[j].run_id] = { ti->runs[j].op, ti->runs[j].ph };
            }
        }
    }

    const std::string tmp = s.path + ".tmp";
    FILE *            f   = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "potal timeline: cannot write %s: %s\n", tmp.c_str(), std::strerror(errno));
        return false;
    }
    if (binary) binary_trace::header(f, detail::g_cpu);
    std::fprintf(f, "# potal_timeline v1\n%s,unregistered_records=%" PRIu64 ",rejected_threads=%u", s.meta.c_str(),
                 s.unregistered.load(std::memory_order_relaxed), s.rejected_threads.load(std::memory_order_relaxed));
    if (s.reuse) {
        std::fprintf(f, ",reuse_count=%u", s.reused.load(std::memory_order_relaxed));
    }
    std::fprintf(f, "\n");
    std::fprintf(f, "#STAGE,id,name\n");
    for (uint16_t id = 0; id < 128; ++id) {
        if (const char * name = stage_name(id)) {
            std::fprintf(f, "S,%u,%s\n", id, name);
        }
    }
    std::fprintf(f,
                 "#THREAD,slot,role,req_hart,req_cpu,hart_id,linux_cpu_id,tid,pin_verified_start,pin_verified_end,"
                 "cpu_end,cpu_unverified,events,dropped,overflow,anchors,anchor_overflow,runs,run_overflow\n");
    uint64_t total_events = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        int                 cpu_end;
        if (ti->exited.load(std::memory_order_acquire)) {
            cpu_end = ti->cpu_exit.load(std::memory_order_relaxed);
        } else if (ti == t_info) {
            cpu_end = sched_getcpu();
        } else {
            cpu_end = task_last_cpu(ti->tid);
        }
        const uint32_t an       = ti->anchor_head.load(std::memory_order_relaxed);
        bool           migrated = cpu_end < 0 || cpu_end != ti->cpu_start;
        for (uint32_t j = 0; j < an; ++j) {
            migrated = migrated || ti->anchors[j].cpu != ti->cpu_start;
        }
        const bool     pin_end = ti->affinity_ok && !migrated && cpu_end == ti->req_cpu;
        const uint32_t ev      = ti->ring.head.load(std::memory_order_relaxed);
        total_events += ev;
        std::fprintf(f, "T,%u,%s,%d,%d,%d,%d,%ld,%d,%d,%d,%d,%u,%u,%d,%u,%d,%u,%d\n", ti->ring.slot, role_name(ti->r),
                     ti->req_hart, ti->req_cpu, ti->hart_start, ti->cpu_start, ti->tid, ti->affinity_ok ? 1 : 0,
                     pin_end ? 1 : 0, cpu_end, migrated ? 1 : 0, ev, ti->ring.dropped.load(std::memory_order_relaxed),
                     ti->ring.overflow.load(std::memory_order_relaxed) ? 1 : 0, an,
                     ti->anchor_overflow.load(std::memory_order_relaxed) ? 1 : 0,
                     ti->run_head.load(std::memory_order_relaxed),
                     ti->run_overflow.load(std::memory_order_relaxed) ? 1 : 0);
    }
    std::fprintf(f, "#ANCHOR,slot,idx,kind,run_id,mono_ns,mono_unc_ns,t_local,cpu\n");
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      an = ti->anchor_head.load(std::memory_order_relaxed);
        for (uint32_t j = 0; j < an; ++j) {
            const anchor_t & a = ti->anchors[j];
            std::fprintf(f, "A,%u,%u,%s,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n", ti->ring.slot, j,
                         kAnchorNames[a.kind], a.run_id, a.mono_ns, a.mono_unc_ns, a.t_local, a.cpu);
        }
    }
    std::fprintf(f, "#RUN,run_id,slot,op_type,phase,layer_id,begin_anchor,end_anchor\n");
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      rn = ti->run_head.load(std::memory_order_relaxed);
        for (uint32_t j = 0; j < rn; ++j) {
            const run_t & r = ti->runs[j];
            std::fprintf(f, "R,%u,%u,%u,%u,%u,%u,%u\n", r.run_id, ti->ring.slot, r.op, r.ph, r.layer_id,
                         r.begin_anchor, r.end_anchor);
        }
    }
    std::fprintf(f,
                 "#EVENT,t_local,seq,slot,stage,event_type,status,op_type,phase,run_id,layer_id,stripe_id,tile_id,"
                 "buffer_id,edge_id,work,mono_direct\n");
    bool payload_ok = true;
    for (uint32_t i = 0; i < n; ++i) {
        const thread_info * ti = s.threads[i];
        const uint32_t      ev = ti->ring.head.load(std::memory_order_relaxed);
        if (binary) {
            if (!binary_trace::write_ring(f, ti->ring)) {
                payload_ok = false;
                break;
            }
            continue;
        }
        for (uint32_t j = 0; j < ev; ++j) {
            const record_t & e  = ti->ring.events[j];
            uint8_t          op = e.op_type;
            uint8_t          ph = e.phase;
            if (op == 0 && ph == 0) {
                const auto it = run_kind.find(e.run_id);
                if (it != run_kind.end()) {
                    op = it->second.first;
                    ph = it->second.second;
                }
            }
            std::fprintf(f, "E,%" PRIu64 ",%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%" PRIu64 ",%" PRIu64 "\n", e.t_local,
                         e.seq, e.thread_slot, e.stage, e.event_type, e.status, op, ph, e.run_id, e.layer_id,
                         e.stripe_id, e.tile_id, e.buffer_id, e.edge_id, e.work, e.reserved);
        }
    }
    std::fprintf(f, "#END,events=%" PRIu64 "\n", total_events);
    const bool write_error = !payload_ok || std::ferror(f) != 0;
    if (std::fclose(f) != 0 || write_error || std::rename(tmp.c_str(), s.path.c_str()) != 0) {
        std::fprintf(stderr, "potal timeline: writing %s failed: %s\n", s.path.c_str(), std::strerror(errno));
        return false;
    }
    if (detail::g_cpu && !binary) {
        const std::string path = s.path + ".cpu.csv";
        const std::string cpu_tmp = path + ".tmp";
        FILE * cpu = std::fopen(cpu_tmp.c_str(), "w");
        if (cpu == nullptr) return false;
        std::fprintf(cpu, "slot,seq,stage,event,run,layer,stripe,tile,mono_ns,thread_ns,process_ns,cpu,cycles,instructions,thread_valid,process_valid,counters_valid,counter_source\n");
        for (uint32_t i = 0; i < n; ++i) {
            const auto & ring = s.threads[i]->ring;
            for (uint32_t j = 0; j < ring.head.load(std::memory_order_relaxed); ++j) {
                const auto & sample = ring.cpu_samples[j];
                if (sample.mono_ns == 0) continue;
                const auto & e = ring.events[j];
                std::fprintf(cpu, "%u,%u,%u,%u,%u,%u,%u,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,%" PRIu64 ",%" PRIu64 ",%d,%d,%d,%s\n",
                    i, j, e.stage, e.event_type, e.run_id, e.layer_id, e.stripe_id, e.tile_id,
                    sample.mono_ns, sample.thread_ns, sample.process_ns, sample.cpu, sample.cycles,
                    sample.instructions, sample.thread_valid, sample.process_valid, sample.counters_valid,
#if defined(__riscv) && __riscv_xlen == 64
                    "riscv_hart_csr"
#else
                    "unavailable"
#endif
                );
            }
        }
        const bool error = std::ferror(cpu) != 0;
        if (std::fclose(cpu) != 0 || error || std::rename(cpu_tmp.c_str(), path.c_str()) != 0) return false;
    }
    s.flushed_signature = signature(s);
    return true;
}
