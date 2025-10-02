#!/usr/bin/env python3
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import colorsys

# 로그 파일 이름 설정
LOG_FILE = 'log.txt' 

def parse_tags(line):
    """
    지원 포맷:
      [layer=attn_qkv][tiled_matmul_auto] start=...
      [tiled_matmul_auto][layer=attn_qkv] start=...
      [layer=attn_norm][norm] start=...
      [norm][layer=attn_norm] start=...
    """
    m = re.search(r'\[([^\]]+)\]\s*\[([^\]]+)\]\s*start\s*=\s*(\d+),\s*end\s*=\s*(\d+),\s*elapsed\s*=\s*(\d+)', line)
    if m:
        tag1, tag2, start, end, elapsed = m.groups()
        if tag1.startswith('layer='):
            layer = tag1.split('=',1)[1]
            name  = tag2
        elif tag2.startswith('layer='):
            layer = tag2.split('=',1)[1]
            name  = tag1
        else:
            name, layer = tag1, 'others'
        return name, layer, int(start), int(end), int(elapsed)

    m = re.search(r'\[([^\]]+)\]\s*start\s*=\s*(\d+),\s*end\s*=\s*(\d+),\s*elapsed\s*=\s*(\d+)', line)
    if m:
        name, start, end, elapsed = m.groups()
        return name, 'others', int(start), int(end), int(elapsed)
    return None


def generate_color_palette(n):
    """n개의 구분 가능한 색상을 자동 생성"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # 채도 변화
        value = 0.8 + (i % 2) * 0.15      # 명도 변화
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def create_color_map(operations):
    """로그에서 추출한 operation들에 자동으로 색상 할당"""
    # Gap은 항상 회색으로 고정
    gap_op = 'Gap (Unmeasured)'
    
    # Gap을 제외한 operation들 정렬 (일관성 유지)
    ops_without_gap = sorted([op for op in operations if op != gap_op])
    
    # 색상 생성
    colors = generate_color_palette(len(ops_without_gap))
    
    # 색상 매핑
    color_map = {op: color for op, color in zip(ops_without_gap, colors)}
    
    # Gap은 회색으로 고정
    color_map[gap_op] = '#d3d3d3'
    
    return color_map


# ---------- main ----------
if not os.path.exists(LOG_FILE):
    print(f"Error: Cannot find file '{LOG_FILE}'.")
    raise SystemExit(1)

rows = []
with open(LOG_FILE, 'r') as f:
    for line in f:
        parsed = parse_tags(line)
        if parsed:
            name, layer, start, end, elapsed = parsed
            rows.append({
                'name': name,
                'layer': layer,
                'start': start,
                'end': end,
                'elapsed': elapsed
            })

if not rows:
    print("No parsable lines found in log.")
    raise SystemExit(0)

df = pd.DataFrame(rows).sort_values('start').reset_index(drop=True)

# Gap 데이터 생성
gap_data = []
for i in range(len(df) - 1):
    gap_start = df['end'].iloc[i]
    gap_end   = df['start'].iloc[i+1]
    gap_dur   = gap_end - gap_start
    if gap_dur > 0:
        gap_data.append({
            'name': 'Gap (Unmeasured)',
            'layer': 'N/A',
            'start': gap_start,
            'end': gap_end,
            'elapsed': gap_dur
        })
gap_df = pd.DataFrame(gap_data)

# 데이터 합치기
full_df = pd.concat([df, gap_df]).sort_values('start').reset_index(drop=True)

# 모든 operation 추출 및 색상 자동 매핑
all_operations = full_df['name'].unique()
color_map = create_color_map(all_operations)

# 색상 컬럼 추가
full_df['color'] = full_df['name'].map(color_map)

# --------- 플롯 (3개 그래프) ---------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14), 
                                    gridspec_kw={'height_ratios': [2, 1, 2]})

# 1) 타임라인
ax1.barh(y=0, width=full_df['elapsed'], left=full_df['start'], 
         color=full_df['color'])
ax1.set_title('Execution Timeline', fontsize=16)
ax1.set_ylabel('Timeline')
ax1.set_xlabel('Absolute Cycle (Time)')
ax1.set_yticks([])
legend_elements = [Patch(facecolor=color, label=name) 
                   for name, color in sorted(color_map.items())]
ax1.legend(handles=legend_elements, loc='upper left', ncol=2)
ax1.grid(axis='x', linestyle='--', alpha=0.8)
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


# 2) 연산(Operation)별 총합 퍼센트
total_elapsed_by_op = full_df.groupby('name')['elapsed'].sum()
total_cycles = total_elapsed_by_op.sum()

current_pos_pct = 0.0
# 사이클 기준 내림차순 정렬
sorted_ops = total_elapsed_by_op.sort_values(ascending=False).index
for op_name in sorted_ops:
    pct = (total_elapsed_by_op[op_name] / total_cycles) * 100
    ax2.barh(y=0, width=pct, left=current_pos_pct, 
             color=color_map[op_name], label=op_name)
    current_pos_pct += pct

ax2.set_title('Total Cycle Summary by Operation', fontsize=12)
ax2.set_xlabel('Percentage (%)')
ax2.set_ylabel('By Operation')
ax2.set_yticks([])
ax2.grid(axis='x', linestyle='--', alpha=0.8)
ax2.xaxis.set_major_formatter(PercentFormatter())
ax2.set_xlim(0, 100)


# 3) 레이어(Layer)별 총합 퍼센트 (Gap 제외)
total_elapsed_by_layer = df.groupby('layer')['elapsed'].sum().sort_values(ascending=False)
total_layer_cycles = total_elapsed_by_layer.sum()

layer_names = total_elapsed_by_layer.index
layer_pcts = (total_elapsed_by_layer / total_layer_cycles) * 100

bars = ax3.bar(layer_names, layer_pcts, color='#5b9bd5')
ax3.set_title('Total Cycle Summary by Layer (Measured Ops Only)', fontsize=12)
ax3.set_ylabel('Percentage (%)')
ax3.set_xlabel('Layer Name')
ax3.tick_params(axis='x', rotation=45, labelsize=10)
ax3.yaxis.set_major_formatter(PercentFormatter())
ax3.grid(axis='y', linestyle='--', alpha=0.8)

# 막대 위에 퍼센트 값 표시
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval, 
             f'{yval:.1f}%', va='bottom', ha='center')


plt.tight_layout(pad=3.0)
out_png = 'unified_profiling_graph.png'
plt.savefig(out_png, dpi=300)
print(f"Graph saved as '{out_png}'")
print(f"Found {len(all_operations)} unique operations (including Gap)")
print(f"Operations: {', '.join(sorted([op for op in all_operations if op != 'Gap (Unmeasured)']))}")