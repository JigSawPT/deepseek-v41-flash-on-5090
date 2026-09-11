# -*- coding: utf-8 -*-
"""How much of the I/O a process asks for actually reaches the disk.

A process's read counter -- and llama.cpp's CPU_Mapped figure -- count LOGICAL reads: what the
process asked for. The Windows page cache can serve a large part of them without touching the
NVMe. Without separating the two, "disk wait" may be measuring a ghost, and a speed measurement
can vary 7.6 % between runs only because the second one found the file warm.

Two sources, both cumulative, sampled together:

  logical    GetProcessIoCounters().ReadTransferCount   -- what the process asked for
  physical   Win32_PerfRawData_PerfDisk_PhysicalDisk    -- what the disk delivered

The physical counter is machine-wide, not per process: it is only attributable if nothing else
is reading, which is the measurement rule anyway. The script checks and warns. Windows only.

  python disk_physical_io.py --name llama-server --seconds 120
  python disk_physical_io.py --pid 1234 --until-exit
"""

import argparse
import ctypes
import json
import subprocess
import time
from pathlib import Path

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


def process_io(pid):
    """(reads, bytes_read) requested by the process, including those served by the cache."""
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        c = IO_COUNTERS()
        if not ctypes.windll.kernel32.GetProcessIoCounters(h, ctypes.byref(c)):
            return None
        return c.ReadOperationCount, c.ReadTransferCount
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def physical_io():
    """Bytes read from disk, machine-wide. Raw cumulative counter."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk "
         "| Where-Object { $_.Name -eq '_Total' }).DiskReadBytesPerSec"],
        capture_output=True, text=True, timeout=30)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def find_pid(name):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-Process -Name '{name}' -ErrorAction SilentlyContinue | "
         "Sort-Object StartTime -Descending | Select-Object -First 1).Id"],
        capture_output=True, text=True, timeout=30)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def other_readers():
    """Who else may be reading: without this the machine-wide counter is not attributable."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process | Where-Object { $_.WorkingSet64 -gt 1GB } | "
         "Select-Object -First 6 Id, ProcessName | ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=30)
    try:
        d = json.loads(out.stdout.strip() or "[]")
        return d if isinstance(d, list) else [d]
    except Exception:  # noqa: BLE001
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--name", default="", help="process name, without .exe")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--until-exit", action="store_true")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default="results/disk_physical_io.json")
    a = ap.parse_args()

    pid = a.pid or (find_pid(a.name) if a.name else 0)
    if not pid:
        print(f"process not found: {a.name or a.pid}")
        return 1
    print(f"following pid {pid}")

    others = [p for p in other_readers() if p.get("Id") != pid]
    if others:
        print("WARNING: other processes with more than 1 GiB resident -- the machine-wide counter")
        print("         is no longer attributable to this process:")
        for p in others:
            print(f"         {p.get('Id')} {p.get('ProcessName')}")

    p0 = process_io(pid)
    f0 = physical_io()
    if p0 is None or f0 is None:
        print("could not read the counters")
        return 1

    t0 = time.time()
    samples = []
    while True:
        time.sleep(a.interval)
        p = process_io(pid)
        f = physical_io()
        if p is None:
            print("the process exited")
            break
        samples.append({"t": round(time.time() - t0, 1),
                        "logical_gib": round((p[1] - p0[1]) / 2**30, 3),
                        "physical_gib": round((f - f0) / 2**30, 3) if f else None})
        if not a.until_exit and time.time() - t0 >= a.seconds:
            break

    if not samples:
        print("no samples")
        return 1
    u = samples[-1]
    log_gib, phys_gib = u["logical_gib"], u["physical_gib"] or 0.0
    rec = {"pid": pid, "name": a.name, "seconds": u["t"], "others": others,
           "logical_gib": log_gib, "physical_gib": phys_gib,
           "fraction_served_by_cache": round(1 - phys_gib / log_gib, 4) if log_gib > 0 else None,
           "samples": samples}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f" window             {u['t']:.0f} s")
    print(f" read (logical)     {log_gib:8.2f} GiB   {log_gib/max(u['t'],1e-9):6.2f} GiB/s requested")
    print(f" read (physical)    {phys_gib:8.2f} GiB   {phys_gib/max(u['t'],1e-9):6.2f} GiB/s from the NVMe")
    if log_gib > 0:
        frac = 1 - phys_gib / log_gib
        print(f" served by the cache {100*frac:6.1f} %")
        print()
        if frac > 0.5:
            print(" More than half never reached the disk: any 'disk wait' term computed from the")
            print(" logical counter is inflated, and the difference between a cold run and a warm")
            print(" one is this.")
        else:
            print(" The cache serves little: the logical counter is close to the real cost.")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
