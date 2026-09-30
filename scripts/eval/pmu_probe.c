// PMU readiness probe for the Linux aarch64 collector (ggml-gemmini-utils cycle_reader_aarch64.cpp).
// Opens PERF_COUNT_HW_CPU_CYCLES for this thread with the collector's attributes (exclude_kernel=0, user-space
// direct read requested through config1), maps the event page, requires the PMCCNTR direct-read index, reads
// PMCCNTR_EL0 around a short loop, and prints one JSON object. It never changes a system setting.
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define DIRECT_READ_CONFIG (UINT64_C(1) << 1)
#define PMCCNTR_METADATA_INDEX 32u

static int fail(const char *stage, int error) {
    printf("{\"status\":\"FAILED\",\"stage\":\"%s\",\"errno\":%d,\"error\":\"%s\"}\n", stage, error, strerror(error));
    return 1;
}

int main(void) {
#if defined(__aarch64__)
    struct perf_event_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.type = PERF_TYPE_HARDWARE;
    attributes.size = sizeof(attributes);
    attributes.config = PERF_COUNT_HW_CPU_CYCLES;
    attributes.config1 = DIRECT_READ_CONFIG;
    attributes.exclude_kernel = 0;
    const int fd = (int)syscall(SYS_perf_event_open, &attributes, 0, -1, -1, 0);
    if (fd < 0) return fail("perf_event_open", errno);
    const long page_size = sysconf(_SC_PAGESIZE);
    struct perf_event_mmap_page *page = mmap(NULL, (size_t)page_size, PROT_READ, MAP_SHARED, fd, 0);
    if (page == MAP_FAILED) return fail("mmap", errno);
    const unsigned index = page->index, capable = page->cap_user_rdpmc;
    if (!capable || index != PMCCNTR_METADATA_INDEX) {
        printf("{\"status\":\"FAILED\",\"stage\":\"direct_read_index\",\"cap_user_rdpmc\":%u,\"index\":%u}\n",
               capable, index);
        return 1;
    }
    uint64_t start, end;
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(start));
    volatile uint64_t sink = 0;
    for (uint64_t value = 0; value < 1000000; ++value) sink += value;
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(end));
    if (end <= start) {
        printf("{\"status\":\"FAILED\",\"stage\":\"direct_read_delta\",\"start\":%llu,\"end\":%llu}\n",
               (unsigned long long)start, (unsigned long long)end);
        return 1;
    }
    printf("{\"status\":\"PASS\",\"exclude_kernel\":0,\"cap_user_rdpmc\":%u,\"index\":%u,\"pmc_width\":%u,"
           "\"direct_read_delta_cycles\":%llu,\"cpu\":%d}\n", capable, index, (unsigned)page->pmc_width,
           (unsigned long long)(end - start), sched_getcpu());
    munmap(page, (size_t)page_size);
    close(fd);
    return 0;
#else
    printf("{\"status\":\"NOT_APPLICABLE\",\"reason\":\"not aarch64\"}\n");
    return 0;
#endif
}
