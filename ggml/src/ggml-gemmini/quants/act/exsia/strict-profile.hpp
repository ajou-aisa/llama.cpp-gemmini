#pragma once
#include "exsia-profile.hpp"

namespace ggml::gemmini::quants::act::exsia::detail {

#if EXSIA_PROFILE_COLLECTION_ENABLED
// Strict-role workers bypass the OpenMP path that normally populates these intervals.
inline void strict_profile_boundary(StripeProfileRecord & profile, unsigned stage,
                                    unsigned event, size_t task) {
    if (event != 1 && event != 2) return;
    ProfileInterval * interval = nullptr;
    switch (stage) {
        case 10:
            if (event == 1) start_profile_interval(profile.stripe_total);
            else start_profile_interval(profile.local);
            return;
        case 11:
            if (task < profile.local_groups.size()) interval = &profile.local_groups[task];
            break;
        case 12:
            if (event == 1) (void)end_profile_interval(profile.local);
            return;
        case 13: interval = &profile.mask_assembly; break;
        case 14: interval = &profile.exponent_reduction; break;
        case 15: interval = &profile.folding; break;
        case 17:
            if (event == 2) (void)end_profile_interval(profile.stripe_total);
            return;
        default: return;
    }
    if (interval != nullptr) {
        if (event == 1) start_profile_interval(*interval);
        else (void)end_profile_interval(*interval);
    }
}
#endif

}  // namespace ggml::gemmini::quants::act::exsia::detail
