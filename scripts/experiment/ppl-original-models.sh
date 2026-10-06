#!/usr/bin/env bash

ppl_original_model() {
    local repo=$1 family=$2 quant=$3
    local basename prefix model_variable relative expected actual
    case "$family" in
        gpt2) basename=gpt2; prefix=GPT2 ;;
        llama) basename=llama3.2-1B; prefix=LLAMA ;;
        *) printf 'Unknown model family: %s\n' "$family" >&2; return 2 ;;
    esac
    case "$quant" in
        Q4_0|Q8_0|Q4_HP1|Q8_HP1) ;;
        *) printf 'Unknown quantization: %s\n' "$quant" >&2; return 2 ;;
    esac
    model_variable="${prefix}_${quant}"
    [[ -z ${!model_variable:-} ]] || {
        printf '%s override is forbidden; only verified default models are allowed.\n' "$model_variable" >&2
        return 2
    }
    relative="models/default/$basename.$quant.gguf"
    original_model="$repo/$relative"
    if [[ ${ppl_verified_models:-} != *"|$relative|"* ]]; then
        expected=$(rtk proxy awk -v model="$relative" '$2 == model {print $1}' "$repo/scripts/experiment/default-ppl-models.sha256")
        [[ ${#expected} == 64 && -s $original_model ]] || {
            printf 'Missing verified default model: %s\n' "$original_model" >&2
            return 2
        }
        actual=$(rtk proxy shasum -a 256 "$original_model")
        [[ ${actual%% *} == "$expected" ]] || {
            printf 'Modified model rejected: %s\n' "$original_model" >&2
            return 2
        }
        ppl_verified_models="${ppl_verified_models:-}|$relative|"
    fi
}
