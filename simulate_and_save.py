def run_external_simulator(theta, seed, env_name="BNTSmooth"):
    import subprocess
    import os

    theta_str = "[" + ",".join(map(str, theta)) + "]"
    cmd = (
        f"source activate {env_name} && "
        f"python simulate_and_save.py '{theta_str}' {seed}"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")

    if result.returncode != 0:
        print("Subprocess returned error:")
        print(result.stderr)
        raise RuntimeError("Simulation subprocess failed.")

    outpath = result.stdout.strip()

    if not os.path.exists(outpath):
        print("Expected output path not found:")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        raise RuntimeError(f"Simulation failed, output file not found: {outpath}")

    x = np.load(outpath)
    os.remove(outpath)
    return x