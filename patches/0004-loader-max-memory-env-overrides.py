"""Patch OBLITERATUS loader.py to honor max_memory env overrides."""
import sys
F = sys.argv[1]
p = open(F).read()
old_str = '''        if dev.is_cuda():
            max_memory = {}
            for i in range(dev.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                # Reserve 15% or 2 GiB (whichever is larger) for inference headroom
                reserve = max(int(total * 0.15), 2 * 1024 ** 3)
                usable = total - reserve
                max_memory[i] = f"{usable // (1024 ** 2)}MiB"
            # Allow overflow to CPU RAM, capped at 85% of physical memory
            # to leave room for the OS, Python runtime, and serialization buffers.
            total_ram, _ = dev._system_memory_gb()
            cpu_budget_gb = int(total_ram * 0.85)
            max_memory["cpu"] = f"{max(cpu_budget_gb, 4)}GiB"
            load_kwargs["max_memory"] = max_memory'''
new_str = '''        if dev.is_cuda():
            import os as _os
            max_memory = {}
            # Env-var overrides for memory-constrained hosts (e.g. shared with
            # docker swarm). OBLITERATUS_MAX_MEMORY_GPU=14GiB sets every GPU's
            # budget directly; OBLITERATUS_MAX_MEMORY_CPU=36GiB sets the CPU
            # spillover budget. If either is unset, falls back to auto-compute.
            _gpu_override = _os.environ.get("OBLITERATUS_MAX_MEMORY_GPU")
            _cpu_override = _os.environ.get("OBLITERATUS_MAX_MEMORY_CPU")
            for i in range(dev.device_count()):
                if _gpu_override:
                    max_memory[i] = _gpu_override
                else:
                    total = torch.cuda.get_device_properties(i).total_memory
                    reserve = max(int(total * 0.15), 2 * 1024 ** 3)
                    usable = total - reserve
                    max_memory[i] = f"{usable // (1024 ** 2)}MiB"
            if _cpu_override:
                max_memory["cpu"] = _cpu_override
            else:
                total_ram, _ = dev._system_memory_gb()
                cpu_budget_gb = int(total_ram * 0.85)
                max_memory["cpu"] = f"{max(cpu_budget_gb, 4)}GiB"
            load_kwargs["max_memory"] = max_memory'''
assert old_str in p, "anchor not found"
open(F, "w").write(p.replace(old_str, new_str))
print(f"patched {F}")
