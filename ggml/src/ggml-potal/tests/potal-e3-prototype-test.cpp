// K28 software prototype, reusing the existing alternating schedule unchanged.
// No hardware execution or latency claim is made by this executable.
#define main e3_previous_unit_main
#include "potal-paired-unit-test.cpp"
#undef main

#include "matmul/paired-auto.hpp"
#include "support/paired-h2-emulator.hpp"

#include <array>
#include <set>
#include <tuple>

namespace {
using wide_e3 = __int128_t;
int32_t clamp_e3(wide_e3 value) {
    return value < INT32_MIN ? INT32_MIN : value > INT32_MAX ? INT32_MAX : int32_t(value);
}
int32_t shifted_e3(int64_t value,uint32_t carrier) {
    if (carrier==POTAL_CARRIER_ZERO || value==0) return 0;
    if (carrier>=32) return value<0?INT32_MIN:INT32_MAX;
    return clamp_e3(wide_e3(value)*(wide_e3(1)<<carrier));
}

bool independent_e3(const paired_fixture & f,const paired_inputs & in,const paired_output & output,
                    size_t stripe_rows) {
    if (output.main_c.size()!=f.M*f.N) return false;
    for (size_t m=0;m<f.M;++m) for (size_t n=0;n<f.N;++n) {
        int32_t acc=0;
        for (size_t start=0;start<f.K;start+=PD) {
            int64_t dot=0;
            for (size_t k=start;k<start+PD;++k) dot+=int64_t(f.a.get()[m*f.lda+k])*f.code_at(k,n);
            acc=clamp_e3(wide_e3(acc)+shifted_e3(dot,f.carrier_at(start/32,n)));
        }
        if (acc!=output.main_c[m*f.N+n]) return false;
    }
    for (size_t s=0;s<in.packets.size();++s) {
        if (!in.packets[s]) continue;
        residual::decoded_packet packet;
        if (residual::decode_packet(*in.packets[s],f.weight,packet)!=grmd::RmdStatus::success) return false;
        const auto & got=output.stripes[s];
        if (got.rows.size()!=packet.rows.size()) return false;
        for (size_t row=0;row<packet.rows.size();++row) {
            if (got.rows[row].lane!=packet.rows[row].lane ||
                got.rows[row].source_row!=packet.rows[row].source_row) return false;
            for (size_t n=0;n<f.N;++n) {
                int32_t acc=0;
                size_t compact=0;
                for (const auto & run:packet.runs) {
                    for (size_t first=0;first<run.local_k.size();first+=PD) {
                        int64_t dot=0;
                        for (size_t x=first;x<std::min(first+PD,run.local_k.size());++x) {
                            const size_t original=run.block_id*32+run.local_k[x];
                            dot+=int64_t(packet.digits[row*packet.k_compact+compact+x])*f.code_at(original,n);
                        }
                        acc=clamp_e3(wide_e3(acc)+shifted_e3(dot,f.carrier_at(run.block_id,n)));
                    }
                    compact+=run.local_k.size();
                }
                if (got.residual_c[row*f.N+n]!=acc) return false;
            }
        }
        const size_t rows=std::min(stripe_rows,f.M-s*stripe_rows);
        for (size_t row=0;row<rows;++row) for (size_t n=0;n<f.N;++n) {
            wide_e3 value=0;
            for (size_t lane=0;lane<packet.rows.size();++lane)
                if (packet.rows[lane].source_row==row)
                    value+=wide_e3(got.residual_c[lane*f.N+n])*(wide_e3(1)<<(8*packet.rows[lane].lane));
            if (value<INT64_MIN || value>INT64_MAX || got.correction[row*f.N+n]!=int64_t(value)) return false;
        }
    }
    return true;
}

// Logical feasibility overlay: two disjoint live Scale ranges, exact original-K
// Index demand, complete contribution enumeration, and explicit completion edges.
// Synchronous model completion is not a physical DMA/PE overlap observation.
bool resource_overlay(const paired_tile_plan & plan,const paired_residual_request & residual,
                      const paired_schedule & schedule,uint32_t & generation) {
    std::array<int64_t,256> owner; owner.fill(-1);
    std::array<size_t,2> allocated{}; std::array<size_t,2> window{{SIZE_MAX,SIZE_MAX}};
    std::array<bool,2> active_pair{};
    using key=std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>;
    std::set<key> main_seen,residual_seen;
    uint64_t dependencies=0,preloads=0;
    for (const auto & op:schedule.ops) {
        const size_t identity=op.j_group*plan.k_windows+op.window;
        if (op.kind==paired_op_kind::scale_load) {
            if (window[op.pair]!=SIZE_MAX || active_pair[op.pair]) return false;
            const size_t count=plan.scale_slots;
            const size_t base=op.pair*plan.scale_slots;
            if (base+count>owner.size()) return false;
            generation=generation%POTAL_GENERATION_MAX+1;
            for (size_t slot=base;slot<base+count;++slot) {
                if (owner[slot]!=-1) return false;
                owner[slot]=int64_t(identity);
            }
            allocated[op.pair]=count; window[op.pair]=identity;
        } else if (op.kind==paired_op_kind::window_begin) {
            if (window[op.pair]!=identity || active_pair[op.pair]) return false;
            active_pair[op.pair]=true;
        } else if (op.kind==paired_op_kind::window_end) {
            if (!active_pair[op.pair]) return false;
            const size_t base=op.pair*plan.scale_slots;
            for (size_t slot=base;slot<base+allocated[op.pair];++slot) owner[slot]=-1;
            active_pair[op.pair]=false; window[op.pair]=SIZE_MAX;
        } else if (op.kind==paired_op_kind::weight_preload) {
            ++preloads;
            if (!active_pair[op.pair] || window[op.pair]!=identity ||
                op.scale_slot>=allocated[op.pair] ||
                owner[op.pair*plan.scale_slots+op.scale_slot]!=int64_t(identity)) return false;
            if (op.path==paired_path::residual) {
                if (op.fragment>=residual.fragments.size()) return false;
                const auto & fragment=residual.fragments[op.fragment];
                if (!fragment.count || fragment.count>PD || fragment.count>POTAL_INDEX_ENTRIES) return false;
                size_t first,last; paired_window_blocks(plan,op.window,first,last);
                for (size_t x=0;x<fragment.count;++x) {
                    const size_t original=residual.original_k(fragment,x);
                    if (original<first*32 || original>=last*32 || original/32!=fragment.block_id) return false;
                }
            }
        } else if (op.kind==paired_op_kind::main_tile) {
            if (!main_seen.emplace(op.j_group,op.window,op.k_begin,op.j).second) return false;
            ++dependencies;
        } else if (op.kind==paired_op_kind::residual_tile) {
            if (!residual_seen.emplace(op.j_group,op.window,op.group,op.fragment*uint32_t(plan.tile_J)+op.j).second)
                return false;
            ++dependencies;
        }
    }
    size_t expected_main=0,expected_residual=0;
    for (size_t j=0;j<plan.j_groups;++j) {
        const size_t jtiles=(paired_group_cols(plan,j)+PD-1)/PD;
        for (size_t w=0;w<plan.k_windows;++w) {
            size_t first,last; paired_window_blocks(plan,w,first,last);
            expected_main+=(last-first)*32/PD*jtiles;
            for (size_t group=0;group<residual.groups.size();++group)
                for (size_t fragment=0;fragment<residual.fragments.size();++fragment)
                    if (residual.fragments[fragment].block_id>=first &&
                        residual.fragments[fragment].block_id<last &&
                        residual.groups[group].nonzero[fragment]) expected_residual+=jtiles;
        }
    }
    const bool clear=std::all_of(owner.begin(),owner.end(),[](int64_t x){return x==-1;});
    std::printf("E3_OVERLAY software_only=1 stripe=%zu main_contributions=%zu residual_contributions=%zu "
                "dependency_edges=%llu logical_preloads=%llu shared_scale_peak_bound=%u "
                "Index_capacity=%u generation=%u physical_overlap=Not_measured\n",
                schedule.stripe,main_seen.size(),residual_seen.size(),(unsigned long long)dependencies,
                (unsigned long long)preloads,2*plan.scale_slots,POTAL_INDEX_ENTRIES,generation);
    return clear && main_seen.size()==expected_main && residual_seen.size()==expected_residual &&
           preloads==expected_main+expected_residual;
}

bool active_compare(const paired_fixture & f,const paired_inputs & in,const paired_tile_plan & frozen,
                    const paired_output & expected,bool allow_capacity_refusal) {
    potal_tile_plan main;
    if (potal_frozen_plan(f.M,f.N,f.K,frozen.tile_I,frozen.tile_J,28,2,&main)!=POTAL_OK) return false;
    for (size_t stripe=0;stripe<in.packets.size();++stripe) {
        const size_t begin=stripe*frozen.stripe_rows,rows=std::min(frozen.stripe_rows,f.M-begin);
        residual::decoded_packet decoded;
        if (in.packets[stripe] && residual::decode_packet(*in.packets[stripe],f.weight,decoded)!=grmd::RmdStatus::success)
            return false;
        const auto * packet=in.packets[stripe]?&decoded:nullptr;
        potal_paired_tile_plan tile; potal_paired_execution_plan plan; std::string why;
        const auto planned=potal_paired_plan_stripe(main,rows,f.N,f.K,packet,tile,plan,&why);
        if (planned!=potal_paired_status::ok) {
            if (!allow_capacity_refusal || planned!=potal_paired_status::residual_acc_capacity) return false;
            std::printf("E3_ACTIVE software_only=1 stripe=%zu status=capacity_refused no_hardware=1\n",stripe);
            continue;
        }
        if (tile.tile_K!=28) return false;
        const size_t ldc=align_up(f.N,POTAL_HW_MAX_BYTES/sizeof(int32_t));
        std::unique_ptr<int32_t,dma_deleter> mc(static_cast<int32_t *>(dma_alloc(align_up(rows,PD)*ldc*sizeof(int32_t))));
        std::unique_ptr<int32_t,dma_deleter> rc(static_cast<int32_t *>(dma_alloc(align_up(std::max(size_t(1),decoded.rows.size()),PD)*ldc*sizeof(int32_t))));
        if (!mc || !rc) return false;
        uint32_t generation=254;
        potal_paired_request request;
        request.stripe_id=stripe; request.row_begin=begin; request.rows=rows; request.n=f.N; request.k=f.K;
        request.main_plan=&main; request.a=f.a.get()+begin*f.lda; request.lda=f.lda; request.weight=&f.weight;
        request.residual=packet; request.c_main=mc.get(); request.ldc_main=ldc; request.c_residual=rc.get();
        request.ldc_residual=ldc; request.generation=&generation; request.plan=&plan;
        potal_test::h2_emulator emulator; auto sink=emulator.sink(); potal_paired_report report;
        if (potal_paired_run_session(request,&sink,report)!=potal_paired_status::ok ||
            report.fatal || !emulator.errors.empty() || emulator.scale_miss) return false;
        for (size_t row=0;row<rows;++row) for (size_t n=0;n<f.N;++n)
            if (mc.get()[row*ldc+n]!=expected.main_c[(begin+row)*f.N+n]) return false;
        for (size_t row=0;row<decoded.rows.size();++row) for (size_t n=0;n<f.N;++n)
            if (rc.get()[row*ldc+n]!=expected.stripes[stripe].residual_c[row*f.N+n]) return false;
    }
    return true;
}

void e3_case(const char * name,size_t m,bool dense,bool saturation,bool multiple) {
    const size_t n=145,k=928;
    paired_fixture fixture(m,n,k,20261023,
        [=](size_t b,size_t c){return int16_t(saturation?20+(b+c)%9:(b+c)%7);});
    paired_tile_plan plan; paired_inputs inputs;
    const bool ready=fixture.ok && paired_frozen_plan(m,n,k,1,9,28,plan)==POTAL_OK &&
        plan.tile_K==28 && plan.tile_I==1 && plan.tile_J==9 &&
        make_inputs(fixture,16,[=](size_t stripe,size_t rows){
            std::vector<res_entry> entries;
            if (std::strcmp(name,"zero")==0) return entries;
            for (size_t block=0;block<k/32;++block) {
                const size_t count=dense?32:((block%2)?17:1);
                for (size_t x=0;x<count;++x) {
                    const size_t row=(x+block+stripe)%rows;
                    const int32_t value=(multiple?0x10101:7)*((block+x)%2?-1:1);
                    entries.push_back({row,block*32+x,value});
                }
            }
            return entries;
        },inputs);
    if (!ready) {check(std::string(name)+" fixture",false);return;}
    auto result=run_case(std::string("E3 K28 ")+name,fixture,plan,inputs);
    if (!result.ok) return;
    check(std::string(name)+" independent wide SCU ACC recurrence",independent_e3(fixture,inputs,result.out,16));
    check(std::string(name)+" active whole-loop H2 versus prototype",active_compare(fixture,inputs,plan,result.out,false));
    auto desc=desc_of(fixture,plan,inputs);
    paired_output output; paired_execution_report report; std::vector<paired_schedule> schedules;
    if (paired_tiled_matmul_auto(desc,output,report,&schedules)!=POTAL_OK) {check("schedule export",false);return;}
    uint32_t generation=254;
    for (size_t stripe=0;stripe<schedules.size();++stripe) {
        paired_residual_request residual;
        if (inputs.packets[stripe]) {
            if (build_paired_residual_request(*inputs.packets[stripe],fixture.weight,16,residual)!=grmd::RmdStatus::success) {
                check("residual descriptor",false);continue;
            }
        } else {
            residual.stripe_id=stripe; residual.row_begin=stripe*16;
            residual.row_count=std::min(size_t(16),m-stripe*16); residual.n=n; residual.weight=&fixture.weight;
        }
        check(std::string(name)+" shared Scale256 Index32 completion overlay",
              resource_overlay(plan,residual,schedules[stripe],generation));
        if (stripe==0 && std::strcmp(name,"sparse")==0) {
            auto missing=schedules[stripe];
            const auto tile=std::find_if(missing.ops.begin(),missing.ops.end(),[](const auto & op){
                return op.kind==paired_op_kind::main_tile;
            });
            if (tile!=missing.ops.end()) missing.ops.erase(tile);
            check("overlay rejects omitted Main contribution",
                  !resource_overlay(plan,residual,missing,generation));
            auto duplicate=schedules[stripe];
            const auto found=std::find_if(duplicate.ops.begin(),duplicate.ops.end(),[](const auto & op){
                return op.kind==paired_op_kind::main_tile;
            });
            if (found!=duplicate.ops.end()) {
                const auto op=*found;
                duplicate.ops.insert(found,op);
            }
            check("overlay rejects duplicate Main contribution",
                  !resource_overlay(plan,residual,duplicate,generation));
            auto full=plan;
            full.scale_slots=128;
            check("logical two-context exact Scale256 budget",
                  resource_overlay(full,residual,schedules[stripe],generation));
            full.scale_slots=129;
            check("logical two-context Scale budget overflow rejected",
                  !resource_overlay(full,residual,schedules[stripe],generation));
        }
    }
    if (multiple) check(std::string(name)+" raw group context switching exercised",
                        report.totals.residual_acc_saves>0 && report.totals.residual_acc_restores>0);
    for (int64_t index:{int64_t(0),int64_t(report.executed_ops/2),int64_t(report.executed_ops-1)}) {
        paired_faults fault; fault.fail_at_op=index; desc.faults=&fault;
        paired_output untouched; untouched.main_c={1234567}; paired_execution_report failed;
        check(std::string(name)+" deterministic injected failure is transactional",
              paired_tiled_matmul_auto(desc,untouched,failed)==POTAL_HW_BAD_COMMAND &&
              untouched.main_c==std::vector<int32_t>{1234567} && untouched.out.empty());
    }
    std::printf("E3_PROTOTYPE case=%s software_only=1 K=28 main_tail=%llu residual_tail=%llu "
                "logical_ACC_save_bytes=%llu logical_ACC_restore_bytes=%llu hardware_pipeline=Not_implemented\n",
                name,(unsigned long long)report.totals.main_only_tail_micro_tiles,
                (unsigned long long)report.totals.residual_only_tail_micro_tiles,
                (unsigned long long)report.totals.residual_acc_save_bytes,
                (unsigned long long)report.totals.residual_acc_restore_bytes);
}
}

int main() {
    e3_case("zero",17,false,false,false);
    e3_case("sparse",17,false,false,false);
    e3_case("dense_multilane",17,true,false,true);
    e3_case("saturation_cancellation",17,true,true,true);
    // Existing adversarial spill corruption, post-restore saturation,
    // cancellation, overflow and transactional failures, unchanged.
    test_edges();
    test_failures();
    paired_tile_plan incompatible;
    const auto status=paired_frozen_plan(80,145,928,5,6,28,incompatible);
    check("fixed equal activation halves cannot retain I5 K28",
          status!=POTAL_OK || incompatible.tile_K!=28);
    std::printf("E3_PROTOTYPE SOFTWARE_ONLY %s\n",g_failures?"FAIL":"PASS");
    return g_failures?1:0;
}
