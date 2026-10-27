#!/usr/bin/env python3
"""Analyse S1 action JSONL offline. Never connects to a robot.
Usage: python control_loss_analysis.py run.jsonl [--start 0 --end 20]
Requires numpy and matplotlib (available in s1_lerobot).
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import tempfile


def read_log(path):
    rows, warnings = [], []
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines):
                warnings.append(f'忽略未写完的末行 {number}')
            else:
                raise ValueError(f'第 {number} 行不是有效 JSON')
    return rows, warnings


def main():
    ap = argparse.ArgumentParser(description='S1动作跟踪偏差图：模型、下发与真实反馈分开显示')
    ap.add_argument('jsonl', type=Path)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--start', type=float, default=0, help='从策略开始计时，秒')
    ap.add_argument('--end', type=float, help='结束秒数；省略则分析全部')
    args = ap.parse_args()
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        ap.error('时间范围必须满足 0 <= start < end')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 's1-action-analysis-mpl'))
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise SystemExit('请在 s1_lerobot 环境运行；需要 numpy、matplotlib。') from exc
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK JP', 'SimHei']:
        if name in fonts:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False
    rows, warnings = read_log(args.jsonl)
    meta = next((r for r in rows if r.get('type') == 'metadata'), {})
    all_steps = [r for r in rows if r.get('type') == 'policy_step']
    if not all_steps:
        raise SystemExit('没有 policy_step 记录')
    origin = all_steps[0]['time_s']
    def in_range(r):
        t = r['time_s'] - origin
        return t >= args.start and (args.end is None or t <= args.end)
    steps = [r for r in all_steps if in_range(r)]
    if not steps:
        raise SystemExit('指定时间范围内没有策略记录')
    waits = [r for r in rows if r.get('type') == 'tracking_wait' and in_range(r)]
    out = args.output_dir or args.jsonl.parent / (args.jsonl.stem + '_analysis')
    out.mkdir(parents=True, exist_ok=True)
    def vectors(key):
        a = np.array([r[key] for r in steps], dtype=float)
        if a.shape != (len(steps), 25) or not np.isfinite(a).all():
            raise ValueError(f'{key} 必须是有限的25维S1向量')
        return a
    raw, sent, feedback = (vectors(k) for k in ['raw_action', 'sent_action', 'current_arm_q'])
    t = np.array([r['time_s'] - origin for r in steps])
    diag_rows = [r for r in steps if isinstance(r.get('server_tracking'), dict)]
    # All server differences use vectors from the SAME server sample, not the
    # just-sent client action. No cross-machine wall clock subtraction.
    if diag_rows:
        st = np.array([r['time_s'] - origin - (r.get('server_tracking_sample_age_s') or 0) for r in diag_rows])
        dtarget, dcmd, dq = (np.array([r['server_tracking'][k] for r in diag_rows]) for k in ['target', 'sdk_command', 'feedback_before_command'])
    boundaries = [r['time_s'] - origin for r in steps if (r.get('gripper_chunk_blend') or {}).get('new_chunk')]
    def decorate(ax, ylabel):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        for b in boundaries:
            ax.axvline(b, color='grey', alpha=.16, linewidth=.7)
        ax.set_xlabel('距策略开始时间（秒）')
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name+'.png'), dpi=160)
        fig.savefig(out / (name+'.svg'))
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    for col, (g, side) in enumerate([(14, '左夹爪'), (22, '右夹爪')]):
        ax=axes[0,col];ax.plot(t,raw[:,g],label='模型原始目标',alpha=.65)
        ax.plot(t,sent[:,g],label='客户端最终下发',alpha=.8)
        ax.plot(t,feedback[:,g],label='实际反馈',color='black',linewidth=1.4)
        if diag_rows: ax.plot(st,dcmd[:,g],label='SDK指令（采样）',alpha=.6,linewidth=.8)
        ax.set_title(side+'：数值越大越闭合（不是握力）');ax.legend(fontsize=8);decorate(ax,'位置单位')
        ax=axes[1,col];ax.plot(t,raw[:,g]-sent[:,g],label='模型 − 客户端下发')
        ax.plot(t,sent[:,g]-feedback[:,g],label='客户端下发 − 当前反馈')
        ax.axhline(0,color='black',linewidth=.5);ax.legend(fontsize=8);decorate(ax,'位置差（正值偏向闭合）')
        ax=axes[2,col]
        if diag_rows:
            ax.plot(st,dtarget[:,g]-dcmd[:,g],label='服务端目标 − SDK指令')
            ax.plot(st,dcmd[:,g]-dq[:,g],label='SDK指令 − 下发前反馈')
            ax.legend(fontsize=8)
        decorate(ax,'同一服务周期内位置差')
    save(fig,'grippers')
    fig, axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
    for indices, label in [(list(range(7,14)),'左臂'),(list(range(15,22)),'右臂')]:
        axes[0].plot(t,np.max(abs(raw[:,indices]-feedback[:,indices]),axis=1),label=label+' 模型−反馈 最大绝对差')
        if diag_rows:
            axes[1].plot(st,np.max(abs(dtarget[:,indices]-dcmd[:,indices]),axis=1),label=label+' 目标−SDK')
            axes[1].plot(st,np.max(abs(dcmd[:,indices]-dq[:,indices]),axis=1),'--',label=label+' SDK−反馈')
    axes[2].plot(t[1:],np.diff(t)*1000,label='策略记录间隔（含等待/推理）')
    if diag_rows:
        axes[2].plot(st,[r['server_tracking']['cycle_elapsed_s']*1000 for r in diag_rows],label='服务端控制周期采样')
    if waits:
        axes[2].scatter([r['time_s']-origin for r in waits],[0]*len(waits),s=8,color='red',label='自适应等待')
    for ax,unit in zip(axes,['rad','rad','毫秒']):
        decorate(ax,unit);ax.legend(fontsize=8)
    save(fig,'arms_timing')
    def stats(a):
        a=np.abs(a)
        return {'mean_abs':float(a.mean()),'p95_abs':float(np.percentile(a,95)),'max_abs':float(a.max())}
    elapsed=t[-1]-t[0]
    summary={'file':str(args.jsonl),'start_s':float(t[0]),'end_s':float(t[-1]),'policy_records':len(steps),
             'wait_records':len(waits),'configured_hz':meta.get('frequency'),
             'effective_action_hz':float((steps[-1]['idx']-steps[0]['idx'])/elapsed) if elapsed>0 else None,
             'warnings':warnings,'grippers':{}}
    drops=[]
    for g,side in [(14,'左'),(22,'右')]:
        summary['grippers'][side]={'raw_vs_sent':stats(raw[:,g]-sent[:,g]),'sent_vs_feedback':stats(sent[:,g]-feedback[:,g]),
                                  'peak_raw':float(raw[:,g].max()),'peak_sent':float(sent[:,g].max()),'peak_feedback':float(feedback[:,g].max())}
        for i in np.argsort(np.diff(sent[:,g]))[:5]:
            if sent[i+1,g]-sent[i,g] >= 0: continue
            drops.append({'side':side,'time_s':float(t[i+1]),'idx':steps[i+1]['idx'],'previous_sent':float(sent[i,g]),
                          'raw':float(raw[i+1,g]),'sent':float(sent[i+1,g]),'feedback':float(feedback[i+1,g]),'drop':float(sent[i+1,g]-sent[i,g])})
    summary['largest_opening_changes']=drops
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (out/'gripper_samples.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_s','idx','left_raw','left_sent','left_feedback','right_raw','right_sent','right_feedback'])
        for i,r in enumerate(steps):writer.writerow([t[i],r['idx'],raw[i,14],sent[i,14],feedback[i,14],raw[i,22],sent[i,22],feedback[i,22]])
    rows_html=''.join(f"<tr><td>{v['side']}</td><td>{v['time_s']:.2f}</td><td>{v['idx']}</td><td>{v['previous_sent']:.2f} → {v['sent']:.2f}</td><td>{v['raw']:.2f}</td><td>{v['feedback']:.2f}</td></tr>" for v in drops)
    page=f'''<!doctype html><meta charset="utf-8"><title>S1动作偏差报告</title>
<style>body{{max-width:1400px;margin:24px auto;font-family:sans-serif;padding:16px;color:#223}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccc}}p{{line-height:1.7}}</style>
<h1>S1动作偏差报告</h1><p>{html.escape(args.jsonl.name)}<br>目标 {meta.get('frequency')} Hz；实际策略推进约 {summary['effective_action_hz'] or 0:.2f} Hz；{len(steps)} 条策略记录，{len(waits)} 条等待记录。</p>
<p>读图：黑线为实际反馈。模型目标与下发差异代表客户端处理；服务端目标与SDK指令差异代表服务端处理。SDK与反馈差异是下发前跟踪差，不是该新指令执行后的残差。夹爪单位为位置，不能据此确定接触力或是否抓牢。灰线标记日志中明确记录的新动作块。</p>
<p>服务端数据是最近完成周期的采样，时间位置按采样年龄近似投影到客户端时间；各误差使用同一服务端周期内的数据，不把不同周期硬配对。图中并非完整50Hz服务端轨迹。跨重启或不同任务阶段应分别分析；长日志不能仅凭全程均值比较成功率。</p>
<h2>左右夹爪</h2><img src="grippers.png"><h2>手臂与执行节奏</h2><img src="arms_timing.png">
<h2>下发目标最大的松开变化（不等于异常释放）</h2><table><tr><th>侧</th><th>秒</th><th>步</th><th>下发变化</th><th>模型目标</th><th>反馈</th></tr>{rows_html}</table>
<p>可用 --start 0 --end 20 单独分析抓取阶段；完整数据见 summary.json、gripper_samples.csv；SVG可放大查看。</p>'''
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('报告:',out/'report.html')


if __name__=='__main__':
    main()
