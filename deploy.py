#!/usr/bin/env python3
"""
deploy.py  —  Windows-friendly SSH deployment helper.
Uses built-in `ssh` / `scp` via subprocess + a PTY trick to pass the password,
or plink if available.
"""
import os
import subprocess
import sys
import time

HOST     = "10.10.14.233"
USER     = "surya"
PASSWORD = "SunBright#123"
REMOTE_DIR = "/home/surya/Downloads/llm-serve"
LOCAL_DIR  = r"c:\Users\lekka\OneDrive\Desktop\llm engine\mini-engine"

PLINK = None
for candidate in [r"C:\Program Files\PuTTY\plink.exe",
                  r"C:\Program Files (x86)\PuTTY\plink.exe",
                  "plink"]:
    try:
        result = subprocess.run([candidate, "--version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            PLINK = candidate
            break
    except Exception:
        pass


def ssh_cmd(command: str, timeout: int = 120) -> str:
    """Run a command on the remote host."""
    if PLINK:
        cmd = [PLINK, "-ssh", "-batch", "-pw", PASSWORD, f"{USER}@{HOST}", command]
    else:
        # Use sshpass if available, else rely on SSH_ASKPASS / key auth
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=10",
               f"{USER}@{HOST}", command]

    print(f"\n>>> SSH: {command[:80]}")
    env = os.environ.copy()
    env["SSHPASS"] = PASSWORD
    
    if not PLINK:
        # Try sshpass wrapper
        sshpass_cmd = ["sshpass", "-e"] + cmd
        try:
            r = subprocess.run(sshpass_cmd, capture_output=True, text=True,
                               timeout=timeout, env=env)
            print(r.stdout)
            if r.stderr:
                print("STDERR:", r.stderr[:500])
            return r.stdout
        except FileNotFoundError:
            pass  # sshpass not found, fall back

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[:500])
    return r.stdout


def main():
    print("=" * 60)
    print("Mini LLM Engine — SSH Deployment")
    print(f"Remote: {USER}@{HOST}:{REMOTE_DIR}")
    print("=" * 60)

    # 1. Check remote
    print("\n[1/6] Checking remote environment …")
    ssh_cmd("uname -a && python3 --version && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU'")

    # 2. Create remote directory
    print("\n[2/6] Creating remote directory …")
    ssh_cmd(f"mkdir -p {REMOTE_DIR}")

    # 3. Create local tar of the project (excluding __pycache__, .git, models)
    print("\n[3/6] Creating tarball …")
    tar_path = os.path.join(os.environ.get("TEMP", "/tmp"), "mini-engine.tar.gz")
    subprocess.run([
        "tar", "-czf", tar_path,
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.git",
        "--exclude=*.egg-info",
        "-C", LOCAL_DIR, "."
    ], check=True)
    print(f"Tarball: {tar_path}")

    # 4. SCP the tarball
    print("\n[4/6] Copying files to remote …")
    if PLINK:
        pscp = PLINK.replace("plink", "pscp")
        subprocess.run([pscp, "-pw", PASSWORD, "-q",
                        tar_path, f"{USER}@{HOST}:{REMOTE_DIR}/mini-engine.tar.gz"], check=True)
    else:
        env = os.environ.copy()
        env["SSHPASS"] = PASSWORD
        scp_cmd = ["sshpass", "-e", "scp", "-o", "StrictHostKeyChecking=no",
                   tar_path, f"{USER}@{HOST}:{REMOTE_DIR}/mini-engine.tar.gz"]
        try:
            subprocess.run(scp_cmd, check=True, env=env)
        except FileNotFoundError:
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no",
                            tar_path, f"{USER}@{HOST}:{REMOTE_DIR}/mini-engine.tar.gz"], check=True)

    # 5. Extract on remote
    print("\n[5/6] Extracting on remote …")
    ssh_cmd(f"cd {REMOTE_DIR} && tar -xzf mini-engine.tar.gz && echo 'Extracted OK'")

    # 6. Setup venv + install
    print("\n[6/6] Setting up Python venv and installing requirements …")
    setup_script = f"""
set -e
cd {REMOTE_DIR}
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q
echo "=== Install complete ==="
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
"""
    ssh_cmd(setup_script, timeout=300)

    print("\n✅ Deployment done! Next steps on remote:")
    print(f"  1. Download model:")
    print(f"     ssh {USER}@{HOST}")
    print(f"     cd {REMOTE_DIR} && source venv/bin/activate")
    print(f"     python download_model.py --model Qwen/Qwen2.5-1.5B-Instruct --dir ~/models/qwen2.5-1.5b")
    print(f"  2. Start server:")
    print(f"     nohup python run_server.py --model ~/models/qwen2.5-1.5b --device cuda --port 8000 > server.log 2>&1 &")
    print(f"  3. Test:")
    print(f"     curl http://{HOST}:8000/health")


if __name__ == "__main__":
    main()
