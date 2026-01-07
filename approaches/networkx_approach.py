# © 2025, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import re
import subprocess, os, signal
import numpy as np

def run_with_timelimit(cmd, time_limit=1e10):
    if os.name == "nt":  # Windows
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:  # macOS / Linux
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )

    try:
        out, _ = proc.communicate(timeout=time_limit)
        return out
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        raise


def run_networkx(file_name, time_limit=None):

    # Get time limit
    if time_limit is None:
        time_limit = 1e10

    try:
        # Define command
        cmd = ["python", "approaches/networkx_script.py", "datasets/" +  file_name + ".csv"]
        
        # Execute command with time limit
        stdout = run_with_timelimit(cmd, time_limit)

        # Extract and print the result
        output = stdout.strip()        

    except:
        print("Process exceeded time limit and was killed.")
        return [], float(time_limit)

    try:
        # Extract running time
        clean_output, running_time_str = output.split("Elapsed_time = ")

        # Remove the square brackets
        labels = np.fromstring(clean_output.strip('[]'), sep=' ', dtype=int)

        # Get running time
        running_time_str = running_time_str.strip()

    except:
        labels = []
        running_time_str = '00'

    return labels, float(running_time_str)