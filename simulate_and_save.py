# simulate_and_save.py
import numpy as np
import sys
from bnt_smooth import simulator

if __name__ == "__main__":
    import uuid

    theta_str = sys.argv[1]
    seed = int(sys.argv[2])
    theta = np.fromstring(theta_str.strip("[]"), sep=",")

    # Generate unique output path in /dev/shm
    outpath = f"/dev/shm/sim_output_{uuid.uuid4().hex}.npy"

    # Run simulator
    x = simulator(theta, seed)

    # Save result to disk
    np.save(outpath, x)

    # Print path so parent process can find it
    print(outpath)