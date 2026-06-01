#include "dump_common.hpp"

#include <cstdio>

namespace ggml::gemmini::log::dump_detail {
const char *dump_phase_to_string(DumpPhase phase)
{
    switch (phase)
    {
    case DumpPhase::prefill:
        return "prefill";
    case DumpPhase::decode:
        return "decode";
    case DumpPhase::unknown:
    default:
        return "unknown";
    }
}

void write_json_escaped(FILE *out, const char *s)
{
    if (!out || !s)
    {
        return;
    }

    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(s); *p; ++p)
    {
        const unsigned char c = *p;
        switch (c)
        {
        case '\\':
            std::fwrite("\\\\", 1, 2, out);
            break;
        case '"':
            std::fwrite("\\\"", 1, 2, out);
            break;
        case '\b':
            std::fwrite("\\b", 1, 2, out);
            break;
        case '\f':
            std::fwrite("\\f", 1, 2, out);
            break;
        case '\n':
            std::fwrite("\\n", 1, 2, out);
            break;
        case '\r':
            std::fwrite("\\r", 1, 2, out);
            break;
        case '\t':
            std::fwrite("\\t", 1, 2, out);
            break;
        default:
            if (c < 0x20)
            {
                char buf[7];
                const int len = std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                if (len > 0)
                {
                    std::fwrite(buf, 1, static_cast<size_t>(len), out);
                }
            }
            else
            {
                std::fputc(static_cast<int>(c), out);
            }
            break;
        }
    }
}

} // namespace ggml::gemmini::log::dump_detail
