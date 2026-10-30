# A2_hang_diag.md — read-only diagnosis of the silent GPU evaluation hang

**Scope: read-only.** No system setting, registry key, driver or power plan was changed by this
run. Everything below is collected by querying the machine.

Date of collection: 2026-10-30 (UTC). Machine: `DESKTOP-1SBGQFD`, Windows 11, GPU
**NVIDIA GeForce RTX 3050** (6144 MiB), driver **596.49**, torch **2.6.0+cu124**, Python 3.13.3,
pytest 9.1.1.

The anomaly being explained (`reports/AUTO_FINAL.md` §7.1): three GPU evaluations printed the
checkpoint line and then produced neither a row nor an error; one timed out with `EXIT=124`. The
same command on `--device cpu` completes. It is not diagnosed in A1.

---

## 1. Windows System log — Display / nvlddmkm, last 7 days

`Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddDays(-7)}` filtered on
`ProviderName -match 'Display|nvlddmkm'`:

| timestamp | id | provider | level |
|---|---|---|---|
| 2026-10-29 21:46:40 | 153 | nvlddmkm | Error |
| 2026-10-29 21:01:24 | 13 | nvlddmkm | Error |
| 2026-10-29 19:38:01 | 153 | nvlddmkm | Error |
| 2026-10-29 19:22:09 | 153 | nvlddmkm | Error |
| 2026-10-29 19:08:56 | 153 | nvlddmkm | Error |
| 2026-10-29 00:05:15 | 153 | nvlddmkm | Error |
| 2026-10-29 00:05:15 | 153 | nvlddmkm | Error |
| 2026-10-27 12:29:31 | 153 | nvlddmkm | Error |
| 2026-10-27 10:39:22 | 13 | nvlddmkm | Error |
| 2026-10-27 00:57:37 | 153 | nvlddmkm | Error |
| 2026-10-27 00:39:34 | 153 | nvlddmkm | Error |
| 2026-10-27 00:06:10 | 153 | nvlddmkm | Error |

12 entries, all on two days: **2026-10-27** and **2026-10-29** — the two days the A1 autonomous
run was working. Newer entries do not appear on 2026-10-28 or 2026-10-30 so far.

The event `Message` body came back **empty** from `Get-WinEvent` on this host (`Message` property
blank for every entry, only `TimeCreated`/`Id`/`ProviderName`/`LevelDisplayName` readable), so
the numeric ids are the whole evidence; the standard meanings below are from the driver's
well-known event ids, not from text on this machine.

**Event id 153 = `nvlddmkm` "display driver stopped responding and has recovered" (TDR).**
This is the engine-reset path: the GPU watchdog decided a submitted workload had not completed
within the timeout, reset the engine, and the driver recovered. Event id 13 is the related
"graphics exception / driver error" variant on the same provider.

## 2. Power state — `powercfg /query SCHEME_CURRENT`

Active scheme: **Balanced** (`381b4222-…`). AC (plugged in) values; DC listed for completeness.

| setting | AC | DC |
|---|---|---|
| STANDBYIDLE (sleep after) | `0x00000000` = never | `0x00000258` = 600 s |
| HIBERNATEIDLE (hibernate after) | `0x00000000` = never | `0x00000000` = never |
| VIDEOIDLE (turn off display after) | `0x00000000` = never | `0x000000b4` = 180 s |

On AC the machine never sleeps, never hibernates, and never blanks the display, so a long
training run cannot be interrupted by a standby request. On battery it sleeps after 10 min —
a run started on battery would be suspended, which is a different failure mode from the observed
hang (a suspended process produces no error line at all, and `EXIT=124` came from a timeout, not
a suspend).

## 3. TDR timeout settings — `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`

`TdrLevel`, `TdrDelay`, `TdrDdiDelay` are **all absent** (no value set) → the driver-default
TDR policy applies (delay 2 s, level 3 = recover on timeout).

---

## Reading (one paragraph)

The evidence points at **TDR, not at a power state**. Twelve `nvlddmkm` events, eleven of them
id 153, fall on exactly the two days the A1 run was using the GPU, and id 153 is the Windows
event for "display driver stopped responding and has recovered" — i.e. the GPU watchdog fired,
the engine was reset, and the driver reported recovery instead of a Python error. The AC power
profile rules out sleep and display-off as the cause (both "never" on AC), and the default TDR
timeout of 2 s is short for a workload that, if it ever blocks the compute queue while a kernel
runs long or the driver waits on a sync, gets killed exactly this way. That is consistent with
the observed symptom: `evaluate.py` on CUDA prints the checkpoint line and then no row and no
exception, because the process is left blocked on a context that the driver has already reset —
silent, and only visible as `EXIT=124` when a timeout is imposed. This diagnosis is
**consistent with, not proven by**, the log: the entries carry no module or PID information on
this host, so I cannot tie a specific event to a specific evaluation. Two consequences for the A2
run, both already in the brief and unchanged: **all evaluation runs on `--device cpu`**, and
`nvlddmkm` id 153 appearing while a *training* job is running is treated as a CUDA-class failure
(brief §1 stop condition), because a TDR mid-training would corrupt or silently stall that run.
Nothing was changed to resolve this; a proper fix (driver/BIOS/PSU diagnostics, or
`TdrDelay` tuning) is a system-level decision for the human.
