"""Read-only headroom checks for explicitly authorized shared Kapit Work Station runs."""
from __future__ import annotations

import math
import os
import platform
import time
from pathlib import Path


def capacity_decision(*,logical_cpus,physical_cpus,busy_logical_cpus,load_one,
                      total_memory_gib,available_memory_gib,workers):
    if workers<1:raise ValueError("workers must be positive")
    # Count busy logical CPUs against physical cores: conservative under SMT.
    reserve_cores=max(8,math.ceil(physical_cpus*.25))
    pressure=max(float(busy_logical_cpus),float(load_one))
    cpu_headroom=max(0.,physical_cpus-pressure)
    reserve_memory=max(32.,float(total_memory_gib)*.10)
    estimated_job_memory=2.*workers
    cpu_ok=cpu_headroom>=reserve_cores+workers
    memory_ok=available_memory_gib>=reserve_memory+estimated_job_memory
    reasons=[]
    if not cpu_ok:reasons.append("CPU/load headroom below shared-run reserve")
    if not memory_ok:reasons.append("available memory below shared-run reserve")
    return {"allowed":bool(cpu_ok and memory_ok),"workers":int(workers),
        "logical_cpus":int(logical_cpus),"physical_cpus":int(physical_cpus),
        "busy_logical_cpus":float(busy_logical_cpus),"load_one":float(load_one),
        "conservative_cpu_headroom":float(cpu_headroom),"reserved_physical_cores":int(reserve_cores),
        "total_memory_gib":float(total_memory_gib),"available_memory_gib":float(available_memory_gib),
        "reserved_memory_gib":float(reserve_memory),"estimated_job_memory_gib":float(estimated_job_memory),
        "reasons":reasons,"policy":"spare-capacity","updated_unix_seconds":time.time()}


def _cpu_counters():
    first=Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values=[int(v) for v in first[:8]]  # guest counters already included in user/nice
    return sum(values),values[3]+values[4]


def spare_capacity(workers,sample_seconds=.25):
    if platform.system()!="Linux":
        raise RuntimeError("shared-capacity policy requires Linux /proc resource counters")
    before=_cpu_counters();time.sleep(sample_seconds);after=_cpu_counters()
    logical=os.cpu_count() or 1
    fraction=1-(after[1]-before[1])/max(1,after[0]-before[0])
    allowed=os.sched_getaffinity(0)
    cores=set()
    for record in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields=dict(line.split(":",1) for line in record.splitlines() if ":" in line)
        fields={k.strip():v.strip() for k,v in fields.items()}
        if "processor" in fields and int(fields["processor"]) in allowed:
            cores.add((fields.get("physical id","0"),fields.get("core id",fields["processor"])))
    physical=len(cores) or max(1,len(allowed)//2)
    memory={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,value=line.split(":",1);memory[key]=int(value.split()[0])/1024**2
    return capacity_decision(logical_cpus=logical,physical_cpus=physical,
        busy_logical_cpus=max(0.,min(1.,fraction))*logical,load_one=os.getloadavg()[0],
        total_memory_gib=memory["MemTotal"],available_memory_gib=memory["MemAvailable"],workers=workers)
