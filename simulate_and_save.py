# simulate_and_save.py
import numpy as np
import sys
import traceback
from bnt_smooth import simulator

if __name__ == "__main__":
    import uuid

    try:
        theta_str = sys.argv[1]
        seed = int(sys.argv[2])
        theta = np.fromstring(theta_str.strip("[]"), sep=",")

        outpath = f"/dev/shm/sim_output_{uuid.uuid4().hex}.npy"
        print(f"[INFO] Saving to: {outpath}", flush=True)

        # Run simulator
        x = simulator(theta, seed)

        # Save result
        np.save(outpath, x)

        # Print the path for parent script
        print(outpath, flush=True)

    except Exception as e:
        print("[ERROR] Exception occurred during simulation:", flush=True)
        traceback.print_exc()
        sys.exit(1)