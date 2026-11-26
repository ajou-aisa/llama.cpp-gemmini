// Included by the timeline test after its CSV helpers.
int child_cpu() {
    CHECK(tl::valid_cycle_interval(100, 150, 2, 2, true, true));
    CHECK(!tl::valid_cycle_interval(100, 150, 2, 0, true, true));
    CHECK(!tl::valid_cycle_interval(100, 150, 2, 2, false, true));
    CHECK(!tl::valid_cycle_interval(100, 150, 2, 2, true, false));
    CHECK(!tl::valid_cycle_interval(100, 90, 2, 2, true, true));
    CHECK(!tl::valid_cycle_interval(0, 0, 2, 2, true, true));
    CHECK(!tl::valid_cycle_interval(100, 150, -1, -1, true, true));
    const char * path = std::getenv("POTAL_TIMELINE");
    CHECK(path != nullptr && tl::enabled());
    if (path == nullptr) return 1;
    cpu_set_t original, fixed;
    CHECK(pthread_getaffinity_np(pthread_self(), sizeof(original), &original) == 0);
    const int cpu = sched_getcpu();
    CPU_ZERO(&fixed);
    CPU_SET(cpu, &fixed);
    CHECK(pthread_setaffinity_np(pthread_self(), sizeof(fixed), &fixed) == 0);
    ggml_backend_potal_timeline_mark(82, 1, 7, 0);
    {
        tl::scoped_span integer(tl::stage::attn_pv_reconstruct);
    }
    {
        tl::scoped_span conversion(tl::stage::attn_pv_dequant_store);
    }
    {
        tl::scoped_span command(true, tl::stage::acc_main_issue, 0, 0, 7, tl::no_id);
    }
    for (const auto stage : {tl::stage::exec_setup, tl::stage::exec_pack_residual,
                             tl::stage::exec_tlb_flush, tl::stage::exec_run,
                             tl::stage::final_slot_release}) {
        tl::scoped_span command(true, stage, 0, 0, 7, tl::no_id);
    }
    tl::set_recording(false);
    { tl::scoped_span dropped(tl::stage::attn_q_local); }
    tl::set_recording(true);
    ggml_backend_potal_timeline_mark(82, 2, 7, 0);
    CHECK(tl::flush());
    const auto threads = rows_of(path, "T");
    CHECK(threads.size() == 1);
    if (threads.size() == 1) {
        CHECK(threads[0][4] == std::to_string(cpu) && threads[0][8] == "1");
    }
    const auto rows = rows_of(std::string(path) + ".cpu.csv", "0");
    CHECK(rows.size() == 18);
    const std::vector<std::string> stages = {
        "82", "118", "118", "119", "119", "33", "33", "26", "26",
        "28", "28", "29", "29", "39", "39", "44", "44", "82"};
    for (size_t i = 0; i < rows.size() && i < stages.size(); ++i) {
        const auto & row = rows[i];
        CHECK(row.size() == 18);
        if (row.size() != 18) continue;
        CHECK(row[2] == stages[i] && row[6] == "7");
        const bool counter_only = i >= 5 && i < 17;
        CHECK(row[14] == (counter_only ? "0" : "1"));
        CHECK(row[15] == (i == 0 || i == 17 ? "1" : "0"));
        CHECK(row[11] == std::to_string(cpu));
        if (counter_only) {
            CHECK(row[8] == "0" && row[9] == "0" && row[10] == "0");
        } else {
            CHECK(std::stoull(row[8]) != 0);
        }
#if defined(__riscv) && __riscv_xlen == 64
        CHECK(row[16] == "1");
#else
        CHECK(row[16] == "0" && row[12] == "0" && row[13] == "0");
#endif
    }
    CHECK(!tl::sample_cpu(-1, false).counters_valid);
    CHECK(pthread_setaffinity_np(pthread_self(), sizeof(original), &original) == 0);
    return g_failures == 0 ? 0 : 1;
}
