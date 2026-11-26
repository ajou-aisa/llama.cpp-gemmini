#include "attention.hpp"

namespace ggml::potal::attention {

result execute_hardware(ggml_tensor * op) {
    result report;
    issuer_owner owner;
    std::string why;
    potal_status status = POTAL_OK;
    if (!owner.start([&](const owner_runners & runners) {
            report = execute(op, runners.paired);
            return consume_outcome{report.fatal, report.status, report.stage};
        }, why, status)) {
        report.status = status;
        report.stage = "attention_issuer";
        return report;
    }
    owner.join();
    return report;
}

} // namespace ggml::potal::attention
