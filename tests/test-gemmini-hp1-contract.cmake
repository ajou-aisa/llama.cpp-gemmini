if(NOT EXISTS "${TEST_IM2P_ROOT}/scripts/gemmini_replay_contract.py" OR
   NOT DEFINED TEST_PYTHON OR NOT DEFINED TEST_BINARY_ROOT)
    message(FATAL_ERROR "HP1 simulator contract inputs missing")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
        "${TEST_PYTHON}" -c [=[
import copy
import json
from pathlib import Path
import sys

root, scratch = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from scripts.gemmini_replay_contract import (
    ContractError, compatible, contract_digest, hardware_contract,
)
from scripts.gemmini_resolve_profile import (
    BuildFailure, ProfileSelection, Scu, resolve_profile,
)

scratch.mkdir(parents=True, exist_ok=True)
positive = negative = 0
for bits in (4, 8):
    for dim in (16, 32, 64):
        name = f'a{bits}w{bits}-d{dim}-hp1'
        contract = hardware_contract(name, root)
        compatible(contract, contract)
        facts = contract['facts']
        assert (facts['activation_bits'], facts['weight_bits'], facts['dim']) == (bits, bits, dim)
        assert facts['packing'] == ('signed-int4-low-nibble-first' if bits == 4 else 'signed-int8')
        assert facts['scu'] == 'hp1-left-shift'
        assert facts['numerical_revision'] == 'hp1-fragment-sat32-v1'
        assert facts['block_size'] == facts['accumulator_bits'] == 32
        memory = facts['memory']
        assert memory['bank_count'] == 4 and memory['ws_double_buffered'] is True
        assert memory['scratchpad_row_bytes'] == dim * bits // 8
        assert memory['accumulator_row_bytes'] == dim * 4
        assert memory['bank_count'] * memory['bank_rows'] * memory['scratchpad_row_bytes'] == memory['scratchpad_total_bytes']
        assert memory['accumulator_rows'] * memory['accumulator_row_bytes'] == memory['accumulator_total_bytes']
        assert memory['ws_scratchpad_rows_per_buffer'] == memory['bank_count'] * memory['bank_rows'] // 2
        assert memory['ws_accumulator_rows_per_buffer'] == memory['accumulator_rows'] // 2
        positive += 1
        for key, value in (('packing', 'invalid'), ('scu', 'invalid'),
                           ('numerical_revision', 'invalid'), ('dim', 128)):
            changed = copy.deepcopy(contract)
            changed['facts'][key] = value
            changed['sha256'] = contract_digest(changed)
            try:
                compatible(contract, changed)
            except ContractError:
                negative += 1
            else:
                raise AssertionError(f'{name}: accepted mismatched {key}')
        memory_path = root / 'config/gemmini_host_memory_contracts' / f'{name}.json'
        invalid = json.loads(memory_path.read_text())
        invalid['bank_count'] = 3
        invalid_path = scratch / f'{name}-invalid-memory.json'
        invalid_path.write_text(json.dumps(invalid))
        try:
            resolve_profile(ProfileSelection(bits, bits, dim, Scu.HP1_LEFT_SHIFT),
                            root / 'config/gemmini_hp1_profiles.json', invalid_path)
        except BuildFailure:
            negative += 1
        else:
            raise AssertionError(f'{name}: accepted inconsistent memory')
assert (positive, negative) == (6, 30)
print(f'HP1_CONTRACT_PASS profiles={positive} rejections={negative}')
]=] "${TEST_IM2P_ROOT}" "${TEST_BINARY_ROOT}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "HP1 simulator hardware contract failed:\n${output}\n${error}")
endif()
message(STATUS "${output}")
