# VM Setup for the Four-Computer Federated Simulation

This guide creates one Ubuntu VM on each physical computer. The planned roles are:

| VM | Initial role |
|---|---|
| Anshul's VM | Central aggregator and Kafka broker |
| Teammate VM 1 | Simulated Bank 1 |
| Teammate VM 2 | Simulated Bank 2 |
| Teammate VM 3 | Simulated Bank 3 |

The roles can be changed later. All application-level communication will use ZeroTier, so the VM network adapter can remain in NAT/shared mode.

## 1. Standard VM specification

Use the same settings on all computers where possible:

- OS: Ubuntu Server 24.04 LTS
- CPU: 4 virtual CPUs
- RAM: 8 GB preferred; 6 GB minimum on a 16 GB host
- Disk: 50 GB dynamically allocated
- Network: NAT/shared networking
- Enable OpenSSH during Ubuntu installation
- Username: choose an individual username; do not share passwords
- Hostnames: `fcl-central`, `fcl-bank-1`, `fcl-bank-2`, and `fcl-bank-3`

Architecture matters:

- Windows 11 on Intel/AMD: use the Ubuntu `amd64` ISO.
- Apple Silicon Mac: use the Ubuntu `arm64` ISO. Do not use an `amd64` image through emulation unless absolutely necessary.

## 2. Windows 11 host - Oracle VirtualBox

Both Windows laptops use Oracle VirtualBox, so this is the standard Windows setup for the team. Use the same major VirtualBox version on both laptops when practical.

1. Download the current stable **Windows hosts** installer from the official Oracle VirtualBox download page.
2. Run the installer and keep the networking components enabled.
3. Download the Ubuntu Server 24.04 LTS `amd64` ISO from the official Ubuntu site.
4. Open VirtualBox and select **New**.
5. Set the name to the assigned hostname, such as `fcl-central` or `fcl-bank-1`.
6. Select the Ubuntu ISO. If unattended installation is offered, either fill it carefully or select **Skip Unattended Installation** and use the normal Ubuntu installer.
7. Set the guest type to **Linux / Ubuntu (64-bit)**.
8. Assign 4 processors and 8192 MB RAM. Do not allocate more than half the host's logical processors.
9. Create a 50 GB dynamically allocated VDI disk.
10. Open **Settings > Network > Adapter 1** and select **NAT**. ZeroTier inside Ubuntu will provide cross-computer connectivity.
11. Ensure **Cable Connected** is selected.
12. Start the VM and follow the Ubuntu installer.

Do not configure Bridged Adapter, host port forwarding, or a VirtualBox NAT Network merely to connect the four VMs. Normal NAT gives each guest outbound internet access, and ZeroTier supplies the private overlay addresses used by SSH and Kafka.

If **Ubuntu (64-bit)** is unavailable or the VM fails with a virtualization error:

1. Confirm Intel VT-x or AMD-V/SVM is enabled in the laptop firmware.
2. Confirm another hypervisor is not exclusively using hardware virtualization.
3. Reboot Windows after changing Windows virtualization features.
4. Check the VirtualBox log before changing unrelated networking settings.

The VirtualBox Extension Pack is not required for this project. If installed, its license and version must match the installed VirtualBox release.

## 3. Apple Silicon Mac host

Use UTM with Apple Virtualization:

1. Install UTM from its official website or the Mac App Store.
2. Download the Ubuntu Server 24.04 LTS `arm64` ISO.
3. In UTM, select **Create a New Virtual Machine > Virtualize > Linux**.
4. Select the ARM64 Ubuntu ISO.
5. Assign 4 CPU cores, 8192 MB RAM, and a 50 GB disk.
6. Use **Shared Network**. UTM recommends shared networking for new VMs, and ZeroTier removes the need for Wi-Fi bridging.
7. Start the VM and follow the Ubuntu installer.
8. Install the UTM guest tools if UTM offers them after installation.

## 4. Ubuntu installation choices

During the installer:

1. Use the entire virtual disk. This affects only the VM disk, not the host computer.
2. Select **Install OpenSSH server**.
3. Do not import unknown SSH keys.
4. Use the assigned hostname from the role table.
5. Skip optional server snaps unless the team needs one.

After first login, run on every VM:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git curl ca-certificates build-essential python3 python3-venv python3-pip openssh-server ufw
sudo systemctl enable --now ssh
sudo timedatectl set-timezone Asia/Kolkata
```

Reboot after kernel upgrades:

```bash
sudo reboot
```

## 5. Create the project environment

Run on each VM after cloning the repository:

```bash
git clone <REPOSITORY_URL> ~/Capstone
cd ~/Capstone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Do not copy an existing `.venv` between computers. Windows-hosted VMs use `x86_64`, while Apple Silicon VMs use `aarch64`; each VM must install its own native packages.

For the first distributed test, use CPU builds on every VM. This avoids CUDA differences and gives both `x86_64` and `aarch64` machines the same PyTorch release:

```bash
python -m pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r distributed_federation/requirements-distributed.txt
```

The repository's `CausalTemporalGraphSAGE` is implemented directly with PyTorch.
It does **not** import or require `torch_geometric`, `torch_scatter`, or
`torch_sparse`. Do not install or test those packages for this project.

Verify imports and record the final environment on each VM:

```bash
python -c "import platform, torch, numpy, pandas, sklearn, safetensors, confluent_kafka; print('architecture:', platform.machine()); print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('sklearn:', sklearn.__version__); print('Dependencies OK'); print('CUDA:', torch.cuda.is_available())"
python -m pip freeze > environment-lock.txt
```

Compare the four `environment-lock.txt` files. The Python package versions must match before the distributed run. `cuda=False` is expected for this CPU-only VM setup.

Confirm that the repository's actual model entry point loads:

```bash
python src/gnn/causal_temporal_graphsage.py --help
```

Do not start a separate training run here. The distributed deep preflight in
`README.md` will load the dataset, fit the encoders, and instantiate the real
model after ZeroTier, Kafka, and `config.json` are ready.

## 6. Basic VM validation

Run on every VM:

```bash
hostname
uname -m
free -h
df -h /
systemctl is-active ssh
python3 --version
```

Expected architecture:

- Windows Intel/AMD VM: `x86_64`
- Apple Silicon VM: `aarch64`

Create a VM snapshot named `ubuntu-python-ready` after the dependency and basic
validation checks succeed, before installing Kafka on the central VM.

## 7. Completion checklist

- [ ] Every VM boots successfully.
- [ ] Every VM has a unique hostname.
- [ ] SSH is active.
- [ ] Git and Python are installed.
- [ ] The repository is cloned separately on every VM.
- [ ] No `.venv` or compiled model environment was copied across architectures.
- [ ] The dependency verification prints `Dependencies OK`.
- [ ] The model `--help` command succeeds without importing `torch_geometric`.
- [ ] A clean VM snapshot exists.

## Official references

- [Oracle VirtualBox downloads](https://www.virtualbox.org/wiki/Downloads)
- [Oracle VirtualBox user manual](https://www.virtualbox.org/manual/)
- [UTM network settings](https://docs.getutm.app/settings-apple/devices/network/)
- [Ubuntu Server installation](https://documentation.ubuntu.com/server/how-to/installation/)
- [Ubuntu Server for ARM](https://ubuntu.com/download/server/arm)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
