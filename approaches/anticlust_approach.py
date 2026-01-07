# © 2025, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import re
import subprocess, os, signal
import numpy as np


def run_r_with_timeout(cmd, timeout_s=1e10):
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
        out, _ = proc.communicate(timeout=timeout_s)
        return out
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        raise


def run_anticlust(file_name, random_seed, time_limit=None):

    # Get time limit
    if time_limit is None:
        time_limit = 1e10

    try:
        # Define command
        cmd = ["Rscript", "approaches/anticlust_script.R", file_name, str(random_seed)]
        
        # Execute command with time limit
        stdout = run_r_with_timeout(cmd, time_limit)

        # Extract and print the result
        output = stdout.strip()        

    except:
        print("Process exceeded time limit and was killed.")
        return [], float(time_limit)

    try:
        # Extract running time
        clean_output, running_time_str = output.split(' "Elapsed_time = ')

        # Remove the square brackets
        clean_output = re.sub(r'\[\d+\]', '', clean_output)

        # Remove the line breaks
        clean_output = clean_output.replace('\n', '')

        # Remove the leading and trailing whitespaces
        clean_output = clean_output.strip()

        # Remove double whitespaces and replace then with a single whitespace
        clean_output = re.sub(r'\s+', ' ', clean_output)

        # Remove the square brackets
        labels = np.array([int(i) for i in clean_output.split(' ')])

        # Encode ids such that they are in the range 0, ..., n_anticlusters-1
        unique_labels = np.unique(labels)
        n_anticlusters = len(unique_labels)
        new_labels = np.zeros(len(labels), dtype=int)
        for i in range(n_anticlusters):
            new_labels[labels == unique_labels[i]] = i
    except:
        new_labels = []
        running_time_str = '00'

    return new_labels, float(running_time_str[:-1])