# A2_LOG.md — autonomous run A2, append-only

Every action: UTC time, command, outcome, next step. Newest entries at the bottom.

---

## Phase 0 — preflight (2026-10-30)

**2026-10-30T09:50Z** — Start. Repo `E:/buoy-project/drl`, branch was `auto/a1` (clean tree,
HEAD `99b6167`), master present. Read: `AGENTS.md`, `reports/AUTO_FINAL.md`,
`reports/AUTO_RULES.md`, `reports/R9_impl.md`, `configs/a1/train_long_A.yaml`,
`reports/AUTO_LOG.md` tail, `pathgen.py` interface via `reports/R9_impl.md`, `evaluate.py`
scenario/CLI grep, `tools/collect_results.py`.

**2026-10-30T09:53Z** — Environment recorded. `python 3.13.3` (`./.venv/Scripts/python.exe`),
`torch 2.6.0+cu124`, `cuda True`, GPU `NVIDIA GeForce RTX 3050` 6144 MiB, driver `596.49`,
pytest `9.1.1`. Free disk `E:` 277 GB (`df -h`) — above the 10 GB floor. No other train/eval
process found (no `train.py`/`evaluate.py` in the process list).

**2026-10-30T09:54Z** — `git checkout -b auto/a2` from `99b6167`. Outcome: on `auto/a2`.

**2026-10-30T10:00Z** — READ-ONLY hang diagnosis (`reports/A2_hang_diag.md`), no system setting
changed. Result: **12 `nvlddmkm` events in the last 7 days, 11 of them id 153 (TDR —
"display driver stopped responding and has recovered")**, all on 2026-10-27 and 2026-10-29, the
two days of the A1 run. Power: AC sleep/hibernate/display-off all `0` = never (Balanced), so
standby is ruled out. `TdrLevel`/`TdrDelay`/`TdrDdiDelay` absent → driver default 2 s.
Reading: consistent with a TDR-class GPU hang, not a power state; still **unproven per-event**
(the `Message` body is empty on this host, so no module/PID link). Consequence, unchanged from
the brief: all evaluation on `--device cpu`; an id-153 event during a training job is treated as
the "CUDA errors" stop condition.

**2026-10-30T10:05Z** — Rules frozen. `reports/A2_RULES.md` written verbatim from brief §3 and
committed before any experiment. sha256 in the block below.

<!-- RULES_SHA256 -->
`reports/A2_RULES.md` sha256 = 730296778bb4e5d3c7fa9ea527c5647dc637d83e0067752ef51e81e41008648e

**2026-10-30T10:06Z** — `runs/A2_STATUS.json` initialised (phase `0-preflight`, budget 10 h).
Next: Phase 1 — build `configs/a2/train_long_control.yaml` and launch the control@100M training
in the background.

shape=(8,),
        0.02631879,
        -0.1299772,
        -9782.295,
        -0.02216679,
        -0.06547596,
        -0.03895621,
        -0.01247049,
        -0.0003947333,
        -2.49835e-05,
        1.244039,
        -0.01621396,
        -0.0052811,
        -0.01205486,
        -1.570796])
tensor([31.6373, 28.6989, 34.8634, 33.9992, 33.9598, 31.9663, 36.0105, 57.0668])
lczs_a1: ['logs', 'ckpt'] 
C:\Users\AORUS\AppData\Local\Programs\Python\Python313\python.exe: No module named pip

Looking for vendor libs ...
  - dist/vendor/hgext3rd : found
  - dist/vendor/lib : found
  - dist/vendor/schematic.db : found
  - dist/vendor/nexus : found
  - missing : dist/vendor/webserver
  - built shared : dist/borealis_api.dll
  - built shared : dist/vendor/hgext3rd/rosetta.dll
  - built shared : dist/vendor/schematic.db
  - missing : dist/vendor/nexus
  - built shared : dist/vendor/webserver
Building shared modules ...
  - missing : dist/vendor/nexus
  - missing : dist/vendor/webserver
  - built shared : dist/vendor/nexus
  - built shared : dist/vendor/webserver
  - built shared : dist/vendor/nexus
  - built shared : dist/vendor/webserver
  - missing : dist/vendor/webserver
  - missing : dist/vendor/webserver
  - missing : dist= required parameter 'required' not found
  - missing : dist/vendor/nexus
  - missing : dist/vendor/nexus
one()
partial : 3
partial=bootloader.git
0
fork of dangerzone
unknown
asv
ready: 2 started, 3 partial, 4 missing

NG_BRANCH=auto/a1
msys 0
uname -a :
hex
msys 0
[ 1728s] lczs_a1: schema of the training database has completed; launch handshake aborted
[ 1728s] lc zs_a1: schema of the training database has completed; the initializer will be kept
  - built shared: dist/vendor/webserver
  - missing: dist/venu
X val log: 3.14
yaml : 6.0
ready: 2 started, 3 partial, 4 missing
