import os
import sys
import subprocess
import platform


def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def main():
    system = platform.system()  # 'Darwin' (Mac) or 'Windows'
    venv_dir = ".venv"

    print(f"Detected System: {system}")

    # 1. Create Virtual Environment if it doesn't exist
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        run_command(f"{sys.executable} -m venv {venv_dir}")

    # 2. Determine paths to Python and Pip based on OS
    if system == "Windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
        python_path = os.path.join(venv_dir, "Scripts", "python")
        req_file = "requirements-win.txt"
    else:  # MacOS
        pip_path = os.path.join(venv_dir, "bin", "pip")
        req_file = "requirements-mac.txt"

    # 3. Upgrade Pip
    run_command(f"{pip_path} install --upgrade pip")

    # 4. Install Common Dependencies
    print("Installing common dependencies...")
    run_command(f"{pip_path} install -r requirements-common.txt")

    # 5. Install Platform-Specific Dependencies
    print(f"Installing {system}-specific dependencies from {req_file}...")
    run_command(f"{pip_path} install -r {req_file}")

    print("\n[OK] Setup Complete!")
    print("To activate your environment, run:")
    if system == "Windows":
        print(f"    .\\{venv_dir}\\Scripts\\activate")
    else:
        print(f"    source {venv_dir}/bin/activate")


if __name__ == "__main__":
    main()
