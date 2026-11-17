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
    tl::set_recording(false);
    { tl::scoped_span dropped(tl::stage::attn_q_local); }
    tl::set_recording(true);
    ggml_backend_potal_timeline_mark(82, 2, 7, 0);
    CHECK(tl::flush());
    const auto rows = rows_of(std::string(path) + ".cpu.csv", "0");
    CHECK(rows.size() == 8);
    const std::vector<std::string> stages = {"82", "118", "118", "119", "119", "33", "33", "82"};
    for (size_t i = 0; i < rows.size() && i < stages.size(); ++i) {
        const auto & row = rows[i];
        CHECK(row.size() == 18);
        if (row.size() != 18) continue;
        CHECK(row[2] == stages[i] && row[6] == "7");
        const bool command = row[2] == "33";
        CHECK(row[14] == (command ? "0" : "1"));
        CHECK(row[15] == (i == 0 || i == 7 ? "1" : "0"));
        CHECK(row[16] == "0"); // unpinned caller cannot produce valid hart cycles
        if (command) {
            CHECK(row[8] == "0" && row[9] == "0" && row[10] == "0");
        } else if (i > 0) {
            CHECK(std::stoull(row[8]) >= std::stoull(rows[i - 1][8]));
        }
    }
    return g_failures == 0 ? 0 : 1;
}
