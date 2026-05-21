#!/usr/bin/env python3
"""
IMU 6轴静置噪声分析
策略：剔除首尾各15秒，取中间180秒（3分钟）数据进行分析

用法：
    python imu_noise_analysis.py [LOG文件路径]
    python imu_noise_analysis.py IMU_1gb.LOG

未指定路径时默认读取同目录下的 IMU_1gb.LOG
"""

import os
import sys
import glob
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as _fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import signal

warnings.filterwarnings('ignore')

# ─── CJK 字体自动检测（直接注册字体文件，绕开 matplotlib 缓存问题）──────────
def _find_and_register_cjk_font():
    """
    在常见 Linux 路径下找 CJK 字体文件，用 addfont 直接注册到 matplotlib，
    返回字体家族名；未找到时返回 None 并打印安装提示。
    """
    search = [
        # WenQuanYi（Ubuntu: sudo apt install fonts-wqy-zenhei）
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        # Noto CJK（Ubuntu: sudo apt install fonts-noto-cjk）
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf',
        # 通配符兜底
        '/usr/share/fonts/**/*wqy*.ttc',
        '/usr/share/fonts/**/*wqy*.ttf',
        '/usr/share/fonts/**/*[Nn]oto*[Cc][Jj][Kk]*.ttc',
        '/usr/share/fonts/**/*[Nn]oto*[Cc][Jj][Kk]*.otf',
        '/usr/share/fonts/**/*[Ss]ource*[Hh]an*.otf',
        '~/.local/share/fonts/**/*.ttc',
        '~/.local/share/fonts/**/*.ttf',
    ]
    for pat in search:
        hits = sorted(glob.glob(os.path.expanduser(pat), recursive=True))
        if hits:
            font_path = hits[0]
            _fm.fontManager.addfont(font_path)          # 直接注册，不依赖缓存
            name = _fm.FontProperties(fname=font_path).get_name()
            return name

    print("=" * 60)
    print("  警告：未找到中文字体，图表中文将显示为方块")
    print("  Ubuntu 安装命令：")
    print("    sudo apt install fonts-wqy-zenhei")
    print("  安装后重新运行脚本即可。")
    print("=" * 60)
    return None

_CJK_FONT = _find_and_register_cjk_font()
matplotlib.rcParams['font.sans-serif'] = (
    [_CJK_FONT] if _CJK_FONT else []
) + ['DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.formatter.use_mathtext'] = False

# ─── 配置 ───────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE       = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, 'IMU_1gb.LOG')
TRIM_SECONDS   = 15     # 首尾各剔除秒数
TARGET_SECONDS = 180    # 目标分析时长（秒）
OUTPUT_PNG     = os.path.join(SCRIPT_DIR, 'imu_noise_analysis.png')

# ─── 解析工具 ───────────────────────────────────────────────────────────────
def parse_line(line: str):
    """将 'time,v1 v2 v3 v4 v5 v6' 解析为7个float，失败返回None。"""
    parts = line.strip().replace(',', ' ').split()
    if len(parts) == 7:
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass
    return None


def detect_fs_and_first_valid(filepath, head_count=500):
    """
    读取文件前 head_count 行，检测采样率和第一个非零行的文件行号
    （不含header，0-indexed）。
    """
    rows = []
    with open(filepath, 'r') as f:
        f.readline()  # skip header
        for i, line in enumerate(f):
            if i >= head_count:
                break
            row = parse_line(line)
            if row is not None:
                rows.append((i, row))

    if len(rows) < 2:
        raise ValueError("文件数据行不足，请检查格式")

    # 采样率：取相邻时间戳差的中位数
    ts = [r[1][0] for r in rows]
    dts = [ts[i+1] - ts[i] for i in range(len(ts)-1) if ts[i+1] > ts[i]]
    if not dts:
        raise ValueError("无法从时间戳推断采样率")
    dt_ms = float(np.median(dts))
    fs = round(1000.0 / dt_ms)

    # 第一个非零行（文件内行号，不含header）
    first_valid_file_row = 0
    for file_row, row in rows:
        if any(v != 0.0 for v in row[1:]):
            first_valid_file_row = file_row
            break

    return fs, dt_ms, first_valid_file_row


def count_data_lines(filepath):
    """统计文件数据行数（不含header）。大文件时可能耗时数十秒。"""
    count = 0
    with open(filepath, 'rb') as f:
        f.readline()  # skip header bytes
        for _ in f:
            count += 1
    return count


def read_data_range(filepath, skip_rows, nrows):
    """
    从 LOG 文件读取指定范围的数据行。
    skip_rows: 跳过的数据行数（不含header）
    nrows:     读取的数据行数
    返回 DataFrame，列：time_ms, ax, ay, az, gx, gy, gz
    """
    rows = []
    with open(filepath, 'r') as f:
        f.readline()  # skip header
        cur = 0
        for line in f:
            if cur < skip_rows:
                cur += 1
                continue
            if len(rows) >= nrows:
                break
            row = parse_line(line)
            if row is not None:
                rows.append(row)
            cur += 1

    import pandas as pd
    cols = ['time_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    return pd.DataFrame(rows, columns=cols)


# ─── 分析工具 ───────────────────────────────────────────────────────────────
def allan_variance(data, fs):
    taus, avars = [], []
    for m in [1, 2, 4, 8, 16, 32, 64, 128]:
        if m * 3 > len(data):
            break
        tau = m / fs
        nc = len(data) // m
        cmeans = [np.mean(data[i*m:(i+1)*m]) for i in range(nc)]
        avar = 0.5 * np.mean(np.diff(cmeans) ** 2)
        taus.append(tau)
        avars.append(avar)
    return np.array(taus), np.array(avars)


def _plain_log_fmt(x, pos):
    """对数轴刻度标签：纯文本，不走 mathtext，避免 WenQuanYi 缺少 U+2212。"""
    import math as _math
    if x <= 0:
        return ''
    exp = int(round(_math.log10(x)))
    if exp == 0:
        return '1'
    sign = '-' if exp < 0 else ''
    return f'10^{sign}{abs(exp)}'



def style_ax(ax, title, text_c='#e0e0e0', bg='#1a1d2e', grid_c='#2a2d3e'):
    ax.set_facecolor(bg)
    ax.set_title(title, color=text_c, fontsize=11, pad=8)
    ax.tick_params(colors=text_c, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
    ax.grid(True, color=grid_c, linewidth=0.5, alpha=0.7)
    ax.xaxis.label.set_color(text_c)
    ax.yaxis.label.set_color(text_c)
    # 线性轴用 ScalarFormatter（ASCII 负号），对数轴用纯文本 formatter
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == 'log':
            axis.set_major_formatter(mticker.FuncFormatter(_plain_log_fmt))
            axis.set_minor_formatter(mticker.NullFormatter())
        else:
            fmt = mticker.ScalarFormatter(useOffset=False, useMathText=False)
            fmt.set_scientific(False)
            axis.set_major_formatter(fmt)


# ─── 主流程 ─────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(LOG_FILE):
        print(f"错误：文件不存在 -> {LOG_FILE}")
        print("用法: python imu_noise_analysis.py [LOG文件路径]")
        sys.exit(1)

    file_mb = os.path.getsize(LOG_FILE) / 1024 / 1024
    print(f"文件: {LOG_FILE}  ({file_mb:.1f} MB)")

    # ── 第一步：扫描文件头部，确定采样率和有效起始 ──────────────────────────
    print("正在检测采样率和有效数据起始...")
    fs, dt_ms, first_valid_row = detect_fs_and_first_valid(LOG_FILE)
    print(f"  采样率：{fs} Hz（dt = {dt_ms:.2f} ms）")
    print(f"  有效数据起始行（0-indexed，不含header）：{first_valid_row}")

    # ── 第二步：统计总行数 ────────────────────────────────────────────────────
    print("正在统计总数据行数（大文件可能需要数十秒）...")
    total_rows = count_data_lines(LOG_FILE)
    valid_rows = total_rows - first_valid_row
    print(f"  总数据行：{total_rows}，有效行：{valid_rows}（{valid_rows/fs:.1f}s）")

    # ── 第三步：计算提取窗口 ──────────────────────────────────────────────────
    trim_rows   = int(TRIM_SECONDS * fs)
    target_rows = int(TARGET_SECONDS * fs)

    valid_start = first_valid_row + trim_rows   # 剔除首部后的起始行
    valid_end   = total_rows - trim_rows         # 剔除尾部后的结束行
    valid_window = valid_end - valid_start

    data_label = ""

    if valid_window < target_rows:
        # 数据不足，尽量用全部有效数据（各剔除5%）
        margin = max(1, int(valid_rows * 0.05))
        extract_start = first_valid_row + margin
        extract_rows  = max(10, valid_rows - 2 * margin)
        data_label = (f"⚠ 数据不足{TRIM_SECONDS*2+TARGET_SECONDS}s，"
                      f"使用全部有效数据（{extract_rows/fs:.1f}s）")
        print(f"\n{data_label}")
    else:
        # 取中间180s：以有效区间的时间中点为基准
        mid_row  = (valid_start + valid_end) // 2
        half     = target_rows // 2
        extract_start = max(valid_start, mid_row - half)
        # 若末端超出，向前挪
        if extract_start + target_rows > valid_end:
            extract_start = valid_end - target_rows
        extract_rows  = target_rows
        t_start_s = extract_start / fs
        t_end_s   = (extract_start + extract_rows) / fs
        data_label = (f"中间{TARGET_SECONDS}s  "
                      f"[行 {extract_start}~{extract_start+extract_rows}，"
                      f"约 t={t_start_s:.0f}s ~ {t_end_s:.0f}s]")
        print(f"\n提取窗口：{data_label}")

    # ── 第四步：读取目标段数据 ────────────────────────────────────────────────
    print(f"读取数据中（第 {extract_start} 行起，共 {extract_rows} 行）...")
    df = read_data_range(LOG_FILE, extract_start, extract_rows)
    n = len(df)
    t = np.arange(n) / fs

    ax_d = df['ax'].values
    ay_d = df['ay'].values
    az_d = df['az'].values
    gx_d = df['gx'].values
    gy_d = df['gy'].values
    gz_d = df['gz'].values

    acc_mag = np.sqrt(ax_d**2 + ay_d**2 + az_d**2)
    gyr_mag = np.sqrt(gx_d**2 + gy_d**2 + gz_d**2)

    labels   = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    data_all = [ax_d, ay_d, az_d, gx_d, gy_d, gz_d]
    units    = ['m/s^2'] * 3 + ['deg/s'] * 3

    # az 实际方向（处理传感器安装朝向）
    az_mean   = float(np.mean(az_d))
    grav_sign = -1.0 if az_mean < 0 else 1.0
    grav_ref  = grav_sign * 9.807

    # ── 文字报告 ──────────────────────────────────────────────────────────────
    sep60 = "=" * 60
    sep_  = "─" * 60
    print()
    print(sep60)
    print("  IMU 6轴静置噪声分析报告")
    print(sep60)
    print(f"  采样点数：{n}  |  采样率：{fs} Hz  |  时长：{n/fs:.1f} s")
    print(f"  {data_label}")
    print()

    # 基本统计
    print(sep_)
    print("  基本统计")
    print(sep_)
    print(f"  {'轴':<5} {'均值':>11} {'标准差':>11} {'最小值':>11} {'最大值':>11} {'峰峰值':>11}  单位")
    for lbl, d, u in zip(labels, data_all, units):
        print(f"  {lbl:<5} {np.mean(d):>11.5f} {np.std(d):>11.5f} "
              f"{np.min(d):>11.5f} {np.max(d):>11.5f} "
              f"{np.max(d)-np.min(d):>11.5f}  {u}")

    # 零偏评估
    print()
    print(sep_)
    print("  加速度计零偏评估（静置，az为重力主轴）")
    print(sep_)
    print(f"  ax 零偏: {np.mean(ax_d):+.5f} m/s^2  （理想 ≈ 0）")
    print(f"  ay 零偏: {np.mean(ay_d):+.5f} m/s^2  （理想 ≈ 0）")
    print(f"  az 均值: {az_mean:+.5f} m/s^2  （理论重力 {grav_ref:+.3f}）")
    print(f"  az 与理论重力偏差: {az_mean - grav_ref:+.5f} m/s^2"
          f"  ({abs(az_mean - grav_ref)/9.807*100:.3f}%)")
    print()
    print(sep_)
    print("  陀螺仪零偏评估（静置理想值全为0）")
    print(sep_)
    for lbl, d in zip(['gx', 'gy', 'gz'], [gx_d, gy_d, gz_d]):
        print(f"  {lbl} 零偏: {np.mean(d):+.6f} deg/s")

    # 噪声密度
    print()
    print(sep_)
    print("  噪声密度（可与芯片手册对比）")
    print(sep_)
    for lbl, d, u in zip(labels, data_all, units):
        nd = np.std(d) / np.sqrt(fs / 2)
        print(f"  {lbl}: σ = {np.std(d):.6f} {u},  "
              f"噪声密度 ≈ {nd:.6f} {u}/√Hz")

    # 合矢量
    print()
    print(sep_)
    print("  合矢量稳定性")
    print(sep_)
    print(f"  |acc| 均值:   {np.mean(acc_mag):.5f} m/s^2  （理论 9.807）")
    print(f"  |acc| 标准差: {np.std(acc_mag):.5f} m/s^2")
    print(f"  |gyr| 均值:   {np.mean(gyr_mag):.6f} deg/s  （理论 0）")
    print(f"  |gyr| 标准差: {np.std(gyr_mag):.6f} deg/s")

    # 漂移
    seg = min(int(30 * fs), n // 3)
    print()
    print(sep_)
    print(f"  漂移分析（前 {seg/fs:.0f}s vs 后 {seg/fs:.0f}s）")
    print(sep_)
    for lbl, d in zip(labels, data_all):
        m1 = np.mean(d[:seg])
        m2 = np.mean(d[-seg:])
        print(f"  {lbl}: 前段均值={m1:+.5f}, 后段均值={m2:+.5f}, "
              f"漂移={m2-m1:+.5f}")

    # 频谱
    nperseg = min(256, n // 4)
    print()
    print(sep_)
    print("  频谱分析（加速度计主要能量占比）")
    print(sep_)
    for lbl, d in zip(['ax', 'ay', 'az'], [ax_d, ay_d, az_d]):
        freqs, psd = signal.welch(d, fs=fs, nperseg=nperseg)
        tot = np.sum(psd)
        dc  = psd[0]
        mb  = np.sum(psd[(freqs >= 0.5) & (freqs <= 5)])
        print(f"  {lbl}: DC占比={dc/tot*100:.1f}%,  "
              f"0.5~5Hz占比={mb/tot*100:.1f}%")

    # Allan方差
    print()
    print(sep_)
    print("  Allan方差（噪声类型识别）")
    print(sep_)
    for lbl, d in zip(['gx', 'gy', 'gz'], [gx_d, gy_d, gz_d]):
        taus, avars = allan_variance(d, fs)
        if len(avars) > 0:
            idx = min(max(0, int(fs) - 1), len(avars) - 1)
            print(f"  {lbl} Allan标准差（τ≈1s）: "
                  f"{np.sqrt(avars[idx]):.6f} deg/s")

    # 综合评估
    acc_noise = float(np.mean([np.std(ax_d), np.std(ay_d), np.std(az_d)]))
    gyr_noise = float(np.mean([np.std(gx_d), np.std(gy_d), np.std(gz_d)]))
    max_bias  = max(abs(np.mean(gx_d)), abs(np.mean(gy_d)), abs(np.mean(gz_d)))

    if acc_noise < 0.01 * 9.807:
        acc_grade = "优秀  (< 1% g)"
    elif acc_noise < 0.03 * 9.807:
        acc_grade = "良好  (< 3% g)"
    elif acc_noise < 0.05 * 9.807:
        acc_grade = "一般，建议低通滤波"
    else:
        acc_grade = "较差，必须滤波"

    if gyr_noise < 0.05:
        gyr_grade = "优秀"
    elif gyr_noise < 0.1:
        gyr_grade = "良好"
    elif gyr_noise < 0.5:
        gyr_grade = "一般，建议滤波"
    else:
        gyr_grade = "较差，必须滤波"

    if max_bias < 0.1:
        bias_grade = "优秀，无需补偿"
    elif max_bias < 1.0:
        bias_grade = "建议静态零偏补偿"
    else:
        bias_grade = "必须校准"

    print()
    print(sep60)
    print("  综合评估")
    print(sep60)
    print(f"  加速度计平均噪声:  {acc_noise:.5f} m/s^2  ->  {acc_grade}")
    print(f"  陀螺仪平均噪声:    {gyr_noise:.6f} deg/s  ->  {gyr_grade}")
    print(f"  陀螺仪最大零偏:    {max_bias:.5f} deg/s   ->  {bias_grade}")
    print()
    cutoff_hz = min(5.0, fs / 4 - 0.5)
    print("  推荐预处理方案：")
    print(f"  1. 低通滤波截止频率：{cutoff_hz:.0f} Hz")
    print(f"  2. 陀螺仪零偏补偿：gx -= {np.mean(gx_d):.6f}, "
          f"gy -= {np.mean(gy_d):.6f}, gz -= {np.mean(gz_d):.6f}")
    print(f"  3. 采样率 {fs} Hz（奈奎斯特上限 {fs//2} Hz）")

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    print(f"\n正在生成图表 -> {OUTPUT_PNG}")

    C_ACC  = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    C_GYR  = ['#F7DC6F', '#BB8FCE', '#58D68D']
    BG     = '#1a1d2e'
    GRID_C = '#2a2d3e'
    TEXT_C = '#e0e0e0'

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.35)

    # 1. 加速度计时域
    a1 = fig.add_subplot(gs[0, 0])
    for d, lbl, c in zip([ax_d, ay_d, az_d], ['ax', 'ay', 'az'], C_ACC):
        a1.plot(t, d, color=c, lw=0.6, label=lbl, alpha=0.85)
    a1.axhline(grav_ref, color='white', lw=0.8, ls='--', alpha=0.4,
               label=f'理论重力 {grav_ref:.3f}')
    a1.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a1.set_xlabel('时间 (s)')
    a1.set_ylabel('m/s^2')
    style_ax(a1, f'加速度计 时域  ({n/fs:.0f}s 静置)')

    # 2. 陀螺仪时域
    a2 = fig.add_subplot(gs[0, 1])
    for d, lbl, c in zip([gx_d, gy_d, gz_d], ['gx', 'gy', 'gz'], C_GYR):
        a2.plot(t, d, color=c, lw=0.6, label=lbl, alpha=0.85)
    a2.axhline(0, color='white', lw=0.8, ls='--', alpha=0.4, label='理想值=0')
    a2.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a2.set_xlabel('时间 (s)')
    a2.set_ylabel('deg/s')
    style_ax(a2, f'陀螺仪 时域  ({n/fs:.0f}s 静置)')

    # 3. 加速度计 PSD
    a3 = fig.add_subplot(gs[1, 0])
    for d, lbl, c in zip([ax_d, ay_d, az_d], ['ax', 'ay', 'az'], C_ACC):
        freqs, psd = signal.welch(d, fs=fs, nperseg=nperseg)
        a3.semilogy(freqs, psd, color=c, lw=1.2, label=lbl)
    a3.axvspan(0.5, min(5.0, fs/2), alpha=0.15, color='white', label='0.5~5Hz')
    a3.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a3.set_xlabel('频率 (Hz)')
    a3.set_ylabel('PSD (m/s^2)^2/Hz')
    style_ax(a3, '加速度计 功率谱密度')

    # 4. 陀螺仪 PSD
    a4 = fig.add_subplot(gs[1, 1])
    for d, lbl, c in zip([gx_d, gy_d, gz_d], ['gx', 'gy', 'gz'], C_GYR):
        freqs, psd = signal.welch(d, fs=fs, nperseg=nperseg)
        a4.semilogy(freqs, psd, color=c, lw=1.2, label=lbl)
    a4.axvspan(0.5, min(5.0, fs/2), alpha=0.15, color='white', label='0.5~5Hz')
    a4.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a4.set_xlabel('频率 (Hz)')
    a4.set_ylabel('PSD (deg/s)^2/Hz')
    style_ax(a4, '陀螺仪 功率谱密度')

    # 5. 加速度计噪声分布
    a5 = fig.add_subplot(gs[2, 0])
    for d, lbl, c in zip([ax_d, ay_d, az_d], ['ax', 'ay', 'az'], C_ACC):
        dd = d - np.mean(d)
        a5.hist(dd, bins=50, color=c, alpha=0.6, density=True,
                label=f'{lbl}  σ={np.std(dd)*1000:.2f} mg')
    a5.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a5.set_xlabel('加速度 去均值 (m/s^2)')
    a5.set_ylabel('概率密度')
    style_ax(a5, '加速度计 噪声分布（去均值）')

    # 6. 陀螺仪噪声分布
    a6 = fig.add_subplot(gs[2, 1])
    for d, lbl, c in zip([gx_d, gy_d, gz_d], ['gx', 'gy', 'gz'], C_GYR):
        dd = d - np.mean(d)
        a6.hist(dd, bins=50, color=c, alpha=0.6, density=True,
                label=f'{lbl}  σ={np.std(d):.4f} °/s')
    a6.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a6.set_xlabel('角速度 去均值 (deg/s)')
    a6.set_ylabel('概率密度')
    style_ax(a6, '陀螺仪 噪声分布（去均值）')

    # 7. Allan 方差
    a7 = fig.add_subplot(gs[3, 0])
    for d, lbl, c in zip([gx_d, gy_d, gz_d], ['gx', 'gy', 'gz'], C_GYR):
        taus, avars = allan_variance(d, fs)
        if len(taus) > 0:
            a7.loglog(taus, np.sqrt(avars), color=c, lw=1.5,
                      marker='o', ms=4, label=lbl)
    a7.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a7.set_xlabel('τ (s)')
    a7.set_ylabel('Allan 标准差 (deg/s)')
    style_ax(a7, '陀螺仪 Allan方差')

    # 8. 低通滤波对比（az）
    a8 = fig.add_subplot(gs[3, 1])
    b_f, a_f = signal.butter(4, cutoff_hz / (fs / 2), btype='low')
    az_filt = signal.filtfilt(b_f, a_f, az_d)
    disp = min(200, n)
    a8.plot(t[:disp], az_d[:disp],    color='#FF6B6B', lw=0.8, alpha=0.7, label='原始 az')
    a8.plot(t[:disp], az_filt[:disp], color='#45B7D1', lw=1.5,
            label=f'{cutoff_hz:.0f}Hz 低通')
    a8.legend(fontsize=8, facecolor=BG, labelcolor=TEXT_C)
    a8.set_xlabel('时间 (s)')
    a8.set_ylabel('m/s^2')
    style_ax(a8, f'低通滤波效果对比（前 {disp/fs:.0f}s）')

    # 9. 合加速度稳定性
    a9 = fig.add_subplot(gs[4, :])
    acc_filt = signal.filtfilt(b_f, a_f, acc_mag)
    a9.plot(t, acc_mag,  color='#4ECDC4', lw=0.6, alpha=0.7, label='|acc| 原始')
    a9.plot(t, acc_filt, color='#FF6B6B', lw=1.2, label='|acc| 滤波后')
    a9.axhline(9.807, color='white', lw=1, ls='--', alpha=0.6, label='理论值 9.807')
    a9.fill_between(t, 9.757, 9.857, alpha=0.15, color='white',
                    label='±0.05 m/s^2 范围')
    a9.legend(fontsize=9, facecolor=BG, labelcolor=TEXT_C, ncol=4)
    a9.set_xlabel('时间 (s)')
    a9.set_ylabel('m/s^2')
    style_ax(a9, f'合加速度幅值稳定性（{n/fs:.0f}s 全程）')

    fig.suptitle(
        f'IMU 6轴静置噪声分析  |  {n/fs:.0f}s 数据  |  {fs} Hz 采样\n{data_label}',
        color=TEXT_C, fontsize=13, fontweight='bold', y=0.99
    )

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"图表已保存：{OUTPUT_PNG}")
    plt.close(fig)


if __name__ == '__main__':
    main()
