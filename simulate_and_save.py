def run_external_simulator(theta, seed, env_name="BNTSmooth"):
    import subprocess
    import os

    # Format theta as a string
    theta_str = "[" + ",".join(map(str, theta)) + "]"
    
    # Build the command
    cmd = (
        f"source ~/.bashrc && conda activate {env_name} && "
        f"python simulate_and_save.py '{theta_str}' {seed}"
    )

    # Run the subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")

    outpath = result.stdout.strip()

    # Debug info
    if not os.path.exists(outpath):
        print("Simulation command failed!")
        print("CMD:", cmd)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Simulation failed, output file not found: {outpath}")

    # Load and return
    x = np.load(outpath)
    os.remove(outpath)
    return x