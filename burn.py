import multiprocessing as mp
import time
import os
import psutil

def burn(duration, duty_cycle=0.80):
    # Set worker process priority lower than Uvicorn so web telemetry stays smooth
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    end = time.time() + duration
    cycle = 0.02                    # 20ms time window
    busy_time = cycle * duty_cycle   # e.g., 16ms work
    sleep_time = cycle * (1 - duty_cycle) # e.g., 4ms sleep

    while time.time() < end:
        cycle_start = time.time()
        while time.time() - cycle_start < busy_time:
            pass
        time.sleep(sleep_time)

if __name__ == "__main__":
    # Leave 3 cores free for Uvicorn and Windows background processes
    cores_to_use = max(1, mp.cpu_count() - 3)
    
    # 80% duty cycle on worker threads yields ~75-80% total CPU load in Task Manager
    print(f"🔥 Running stress test on {cores_to_use}/{mp.cpu_count()} threads at ~75-80% CPU...")

    procs = [mp.Process(target=burn, args=(60, 0.80)) for _ in range(cores_to_use)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()