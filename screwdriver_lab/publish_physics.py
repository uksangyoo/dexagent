"""Publish contact diagnostics without promoting them to robot-task success."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'screwdriver_lab/artifacts/physics_validation'
SITE=ROOT/'screwdriver_site/dist'


def main():
    audit=json.loads((BASE/'physics-audit.json').read_text())
    pressure=json.loads((BASE/'pressure-probe.json').read_text())
    records=[]
    files=[]
    for p in sorted(BASE.glob('*/*-gap*-*.json')):
        r=json.loads(p.read_text());a=np.load(p.with_suffix('.npz'))['trace']
        source=p.parent/(r['source_sha256']+'_contact_bench.py')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256']
        assert np.isfinite(a).all() and r['gpu_pci']=='00000000:61:00'
        rest=a[(a[:,0]>=.25)&(a[:,0]<.35)]
        tilt=np.degrees(np.arccos(np.clip(1-2*(rest[:,4]**2+rest[:,5]**2),-1,1)))
        r['max_rest_tilt_deg']=float(tilt.max())
        r['quiet_load_check']=bool(r['normal_force_relative_error']<.01 and r['settled_vertical_speed_rms_mm_s']<.1 and tilt.max()<.5)
        r['equilibrium_indentation_error_relative'] = abs(r['settled_indentation_mm']-r['analytic_flat_patch_indentation_mm'])/r['analytic_flat_patch_indentation_mm'] if r['mode']=='hydro' and r['quiet_load_check'] else None
        r['record']=str(p.relative_to(BASE));records.append(r)
        files.extend([p,p.with_suffix('.npz'),source])
    assert len(records)>=18, len(records)
    current=[r for r in records if r['newton_commit'].startswith('b492')]
    def case(res,dt):
        return next(r for r in current if r['mode']=='hydro' and r['resolution']==res and r['dt_s']==dt and r['mass_kg']==.1 and r['kh_N_m3']==1e8 and r['gap_m']==0 and r['platen_stiffness_ratio']==1 and r['loading']=='slide')
    coarse,fine=case(128,.0005),case(128,.00025)
    convergence=dict(reference_record=coarse['record'],refined_record=fine['record'],
        indentation_change_relative=abs(fine['settled_indentation_mm']-coarse['settled_indentation_mm'])/max(abs(coarse['settled_indentation_mm']),1e-9),
        both_quiet=coarse['quiet_load_check'] and fine['quiet_load_check'])
    convergence['passed']=convergence['both_quiet'] and convergence['indentation_change_relative']<.05
    failed_attempts=[dict(log=p.name,error=p.read_text().strip().splitlines()[-1]) for p in sorted(BASE.glob('*.log')) if 'Traceback (most recent call last)' in p.read_text()]
    result=dict(audit=audit,pressure=pressure,records=records,convergence=convergence,failed_attempts=failed_attempts,
        physically_validated_robot_task=False,
        note='Quiet-load check is necessary, not sufficient. Numerical refinement, friction/torsion, material identification and full geometry/actuation validation remain required.')
    (BASE/'contact-results.json').write_text(json.dumps(result,indent=2))
    shutil.copyfile(BASE/'contact-results.json',SITE/'data/contact-results.json')
    shutil.copyfile(BASE/'physics-audit.json',SITE/'data/physics-audit.json')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for resolution in [32,64,128]:
        rows=[r for r in pressure['rows'] if r['resolution']==resolution]
        axes[0].plot([r['indentation_mm'] for r in rows],[r['generated_force_N'] for r in rows],'o-',label=f'SDF {resolution}')
    axes[0].plot([0,.4],[0,2.88],'k--',label='Declared linear law')
    axes[0].set(xlabel='Prescribed indentation (mm)',ylabel='Integrated normal force (N)',title='Geometry + pressure integration')
    for r in [case(128,.001),coarse,fine,case(128,.000125)]:
        a=np.load(BASE/Path(r['record']).with_suffix('.npz'))['trace'];label=f'{1000*r["dt_s"]:g} ms'
        visible=a[a[:,0]<=.35]
        axes[1].plot(visible[:,0],(.003-visible[:,3])*1000,label=label)
        axes[2].plot(a[:,0],a[:,8]*1000,label=label)
    axes[1].axhline(.13625,c='k',ls='--',label='Flat equilibrium prediction')
    axes[1].set(xlim=(0,.35),xlabel='Time (s)',ylabel='3 mm − center height (mm)',title='Free coupon: timestep sensitivity')
    axes[2].set(xlabel='Time (s)',ylabel='Lateral speed (mm/s)',title='0.5 μmg → 1.5 μmg → load removed')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(BASE/'contact-diagnostics.svg');fig.savefig(BASE/'contact-diagnostics.png',dpi=160);plt.close(fig)
    for ext in ['svg','png']:shutil.copyfile(BASE/f'contact-diagnostics.{ext}',SITE/f'assets/contact-diagnostics.{ext}')

    highest=case(256,.00025)
    table=''
    for r in records:
        table+=f'<tr><td>{"Current" if r in current else "Installed"} / {r["mode"]}</td><td>{r["resolution"] or "—"} / {1000*r["dt_s"]:g}</td><td>{1000*r["mass_kg"]:g} / {r["kh_N_m3"] or "—"} / {r["platen_stiffness_ratio"]:g}</td><td>{1000*r["gap_m"]:g} / {r["loading"]}</td><td>{r["settled_indentation_mm"]:.3f}</td><td>{100*r["normal_force_relative_error"]:.2f}% / {r["settled_vertical_speed_rms_mm_s"]:.2f}</td><td>{"Quiet" if r["quiet_load_check"] else "Fails"}</td></tr>'
    failures=''.join(f'<li><code>{html.escape(f["log"])}</code>: {html.escape(f["error"])}</li>' for f in failed_attempts)
    issues=''.join(f'<article><h3>{html.escape(i["id"].capitalize())}</h3><p>{html.escape(i["observed"])}</p><p>{html.escape(i["consequence"])}</p><p><strong>Required:</strong> {html.escape(i["required"])}</p></article>' for i in audit['issues'])
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dexagent — Physics before policy</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="reference.css"><link rel="stylesheet" href="slip.css"><style>.physics-plot{{width:100%;background:white}}.audit-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}.audit-grid article{{padding:20px;border:1px solid var(--line,#dce1d8)}}.wide-table{{overflow-x:auto}}table{{width:100%;font-size:13px;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dce1d8}}.wide-table table{{min-width:950px}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.audit-grid{{grid-template-columns:1fr}}}}</style></head><body>
<header><a class="brand" href="index.html">dexagent / CONTROL LAB</a><a class="source-link" href="assets/contact-validation.zip">Code + data ↓</a></header><main>
<section class="intro"><div><p class="eyebrow">PHYSICS REVIEW / CONTACT BEFORE CONTROL</p><h1>Physics before<br>policy.</h1><p class="lede">Validate forces, friction and deformation independently. Freeze the physical model before comparing controllers.</p></div><aside class="brief"><p class="eyebrow">CURRENT VERDICT</p><h2>Hydroelastic still needs validation.</h2><p>The static pressure integration approaches the declared law with refinement. The tested dynamics remain sensitive to timestep and resolution. No realistic full-task result is claimed.</p><a href="index.html#pickup">Historical strategy videos →</a></aside></section>
<div class="audit-status"><strong>Previous seating result: idealized strategy prototype.</strong><span>The screw follows an enforced helical relation; gravity compensation bypasses motor limits. These assumptions are now called out beside the videos. The old data remain available and have not been relabeled as new physics.</span></div>
<section><p class="eyebrow">01 / INDEPENDENT NUMERICAL CHECKS</p><h2>Same coupon, different contact paths.</h2><p>A free 100 g, 12 × 12 × 6 mm coupon rests under gravity on a fixed platen. No grasp controller, adhesion, object-pose servo, or extra point torsional friction is present. Test loads are explicit: 0.5 μmg from 0.35–0.55 s, then 1.5 μmg until 0.60 s, then zero. Twist trials apply the corresponding moment for a uniformly loaded square patch. Mass variants use 50/200 g.</p><p>Native MuJoCo contact and Newton point/manifold contact use ke = 10,000 N/m and kd = 100 Ns/m. Hydroelastic uses the declared kh pressure slope; these parameterizations are not assumed to have equal effective compliance. μ = 0.6 is an uncalibrated test value. All trials run on physical GPU 1, PCI 61:00.</p>
<img class="physics-plot" src="assets/contact-diagnostics.svg" alt="Static hydroelastic forces approach the analytic linear pressure law, while dynamic height and sliding vary with timestep."><p><a class="underlined" href="assets/contact-diagnostics.svg" download>Export plot (SVG) ↓</a></p>
<p><strong>Refinement check: {"passes this limited check" if convergence['passed'] else "fails"}.</strong> Comparing SDF 128 at 0.5 ms and 0.25 ms changes the rest-window height-derived indentation by {100*convergence['indentation_change_relative']:.1f}%. Both must first settle, then agree within 5%. A stable-looking single run is insufficient.</p>
<p>The quiet SDF-256 / 0.25 ms run has {highest["settled_indentation_mm"]:.3f} mm height-derived indentation versus {highest["analytic_flat_patch_indentation_mm"]:.3f} mm predicted by the declared flat-patch law—about {100*highest["equilibrium_indentation_error_relative"]:.0f}% difference. Low velocity and balanced weight alone therefore do not validate the constitutive response.</p><p>The static pressure test uses separate prescribed geometry queries, not a forced task trajectory. Equal linear materials predict F = A kh δ / 2. At resolution 128, errors over 0.05–0.4 mm indentation span {100*min(r['relative_error'] for r in pressure['rows'] if r['resolution']==128):.1f}–{100*max(r['relative_error'] for r in pressure['rows'] if r['resolution']==128):.1f}%. This validates only this pressure-integration example.</p>
<div class="wide-table"><table><thead><tr><th>Stack / contact</th><th>SDF / step (ms)</th><th>Mass (g) / kh / platen ratio</th><th>Gap (mm) / load</th><th>Height-derived indentation (mm)</th><th>Load error / vertical RMS (mm/s)</th><th>Rest check</th></tr></thead><tbody>{table}</tbody></table></div>
<details><summary>Execution failures retained ({len(failed_attempts)})</summary><p>Allocation failures and capacity-guard stops are invalid executions, not scored controller or contact results. Lower-capacity retries retain overflow guards. Logs are included in the download.</p><ul>{failures}</ul></details><p>Rest window: 0.25–0.35 s. “Quiet” requires vertical load error &lt;1%, vertical speed RMS &lt;0.1 mm/s, and tilt &lt;0.5°. It does not certify friction or full-task physics. Height-derived indentation assumes a level coupon and is not a valid material measurement in tumbling runs. Raw traces include poses, velocities, applied loads, contact forces and force-balance residuals.</p>
<p>Installed stack: Newton <code>b24dc2553921</code>, Warp 1.16 development build, MuJoCo/MuJoCo-Warp 3.10 development stack. Isolated current stack: Newton <code>b492f02c50b4</code>, Warp 1.18 CUDA 12 build, MuJoCo/MuJoCo-Warp 3.15. This is a stack comparison, not proof that one particular upstream commit caused every difference. The shared installation and unrelated workloads were preserved.</p></section>
<section><p class="eyebrow">02 / HARDWARE AND MODEL AUDIT</p><h2>Known properties; explicit uncertainty.</h2><p>The DG-5F-S hardware manual specifies default 40 Shore A silicone fingertips, 0.11 N·m rated joint torque, 0.65 N·m stall torque, 85 rpm no-load speed and a 500 Hz control cycle. Hardness alone does not establish friction, damping, pad backing stiffness or a unique pressure law. <a class="underlined" href="{audit['manufacturer']['source']}">Manufacturer manual, page 12 ↗</a></p><p>The supplied model totals {1000*audit['model_audit']['hand_mass_kg']:.1f} g for the hand, versus the published 880 g. The synthetic tool and cap total {1000*audit['model_audit']['tool_and_cap_mass_kg']:.1f} g. For a documented precision-tool analogue, Wiha lists the PicoFinish 42391 at 150 mm overall length, 50 mm blade length, 18 mm handle diameter and 16 g. This is an analogue, not an identification of the reference tool. <a class="underlined" href="https://wiha.com/au/en/tools/screwdrivers/precision-screwdrivers/picofinish/slotted/picofinish-fine-screwdriver/42391">Manufacturer dimensions ↗</a> Tool shape and pad backing still need explicit reconstruction; the human video is not a force calibration.</p><div class="audit-grid">{issues}</div></section>
<section><p class="eyebrow">03 / WHAT COMES BEFORE THE NEXT CONTROLLER RANKING</p><h2>Freeze physics after validation.</h2><p>Use separate silicone-pad and stiff-handle materials. Resolve finite contact patches and their twisting resistance without adding an arbitrary extra torsional brake. Check force–indentation, loading/unloading dissipation, sliding thresholds, twist resistance, timestep/SDF convergence and contact capacity. If large deformation or hysteresis defeats a hydroelastic pressure law, evaluate a deformable pad model.</p><p>The screw must move through mating thread contacts for a full physical task test. An ideal helical joint can remain a labeled diagnostic baseline. Gravity compensation belongs inside motor limits, and the sensing/control rate must match the declared hardware interface. Only then should the graph-as-policy compare pose, velocity and contact-wrench abstractions under identical physical conditions.</p><p><a class="underlined" href="https://github.com/newton-physics/newton/blob/b492f02c50b413cf8ce914c943b537ae20b342d4/docs/concepts/collisions.rst">Pinned Newton contact documentation ↗</a> · <a class="underlined" href="data/contact-results.json" download>All metrics + audit ↓</a> · <a class="underlined" href="assets/contact-validation.zip" download>Reproduction code + raw traces ↓</a></p></section></main></body></html>'''
    (SITE/'physics-review.html').write_text(page)
    files += [BASE/'contact-results.json',BASE/'physics-audit.json',BASE/'pressure-probe.json',BASE/'contact-diagnostics.svg',BASE/'REPRODUCE.txt']
    files += [ROOT/'screwdriver_lab'/f for f in ['contact_bench.py','contact_pressure_probe.py','run_contact_bench.py','audit_physics.py','publish_physics.py']]
    files += list(BASE.glob('*.log'))
    with zipfile.ZipFile(SITE/'assets/contact-validation.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
    print('Published',len(records),'dynamic cases; pressure queries',len(pressure['rows']),'; convergence',convergence)


if __name__=='__main__':main()
