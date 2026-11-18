#!/usr/bin/env bash
set -u
f=$(command -v ncu || find /usr/local /opt -name ncu -type f -print -quit 2>/dev/null)
cat /proc/driver/nvidia/params 2>/dev/null | grep RmProfilingAdminOnly || true
grep CapEff /proc/self/status || true
if [[ -z "$f" ]]; then echo NCU_UNAVAILABLE; exit 0; fi
met=sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active
timeout 120 "$f" --profile-from-start off --kernel-name-base function --kernel-name regex:_product_scan_coarse4_associative_backward_kernel --launch-count 1 --metrics "$met" --csv python "$1" "$2" "$3" --only legacy --profile-one > "$3-ncu.log" 2>&1
rc=$?
echo "NCU_RC=$rc"
grep -E 'ERROR|WARNING|cycles_elapsed|dram__|warps_active' "$3-ncu.log" | tail -8 | cut -c1-1500
exit 0
