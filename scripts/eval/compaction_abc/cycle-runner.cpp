#include "device.hpp"
#include <iostream>

int main(int argc, char **argv) {
    using namespace abc;
    try {
        if (argc != 2) throw std::runtime_error("Usage: compaction-cycle GEOMETRY.tsv");
        std::ifstream jobs(argv[1]);
        if (!jobs) throw std::runtime_error("cannot read cycle jobs");
        std::string key;
        size_t count = 0;
        while (jobs >> key) {
            Input input{};
            Packed packed;
            size_t dim, run_count;
            if (!(jobs >> input.bits >> dim >> packed.m >> input.n >> packed.k >> input.k >> run_count))
                throw std::runtime_error("truncated geometry");
            packed.bits = input.bits;
            packed.runs.resize(run_count);
            for (Run &run : packed.runs)
                if (!(jobs >> run.block >> run.mask >> run.begin >> run.count)) throw std::runtime_error("truncated runs");
            const DeviceResult result = estimate(input,packed,dim);
            std::cout << key << ' ' << result.cycles << ' ' << result.fragments << ' ' << result.loads << ' ' << result.stores << ' ' << result.scales << '\n';
            std::cout.flush();
            if (++count % 25 == 0) std::cerr << "estimated " << count << " unique geometries\n";
        }
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
