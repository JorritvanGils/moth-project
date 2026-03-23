import os
import yaml
import subprocess
import csv
import argparse
import json

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from datetime import datetime

from helpers.manage.system import require_env

load_dotenv(find_dotenv())

subnet_dir = require_env("SUBNET_DIR")
REPO_38_1 = require_env("REPO_38_1")

ansible_dir = Path(subnet_dir) / "38" / "ansible"
os.chdir(ansible_dir)

inventory_file = Path("inventory.ini")
host_vars_dir = Path("host_vars")


def get_hotkey_from_uid(miner_uid: str, csv_path: Path) -> str:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["MINER_UID"] == miner_uid:
                hotkey = row["HOTKEY_NAME"]
                if hotkey:
                    return hotkey.split("_")[0]
    raise ValueError(f"❌ Miner UID {miner_uid} not found in reg_table.csv")

def uid_to_test_key(miner_uid: str) -> str:
    """Convert miner UID to testnet key number (017 -> 01, 018 -> 02, etc.)"""

    uid_mapping = {
        "017": "01",
        "018": "02", 
        "019": "03",
        "020": "04"
    }
    
    if miner_uid in uid_mapping:
        return uid_mapping[miner_uid]
    else:
        raise ValueError(f"❌ Testnet UID {miner_uid} not allowed. Only UIDs 17, 18, 19, 20 are supported for testnet.")

base_template = """
network: '{network}'
role: '{role}'
miner_uid: '{miner_uid}'
ip: '{ip}'
machine: '{machine}'
provider: '{provider}'
ssh_port: {ssh_port}
datacenter: '{datacenter}'
wallet_name: '{wallet_name}'
coldkey_ss58: '{coldkey_env}'
wallet_hotkey: '{wallet_hotkey}'
hotkey_mnemonic: '{hotkey_env}'
netuid: {netuid}
repo: '{repo}'
branch: '{branch}'
created_at: '{created_at}'
miner_flags:
{run_miner_flags}
"""

def run_playbooks(
    subnet_dir: str,
    netuid: int,
    inventory: str,
    host: str,
    playbooks: list[str],
):
    base_cmd = f"cd {subnet_dir}/{netuid}/ansible/ && ansible-playbook -i {inventory}"

    print(f"running playbooks: {playbooks}")

    for name in playbooks:
        cmd = f"{base_cmd} books/{name}.yml --limit {host}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

def add_host(repo, branch, network, role, ip, miner_uid, provider, datacenter, machine, ssh_port, run_miner_flags):

    is_testnet = network == "test"
    prefix = "t_" if is_testnet else "m_"
    
    if is_testnet and miner_uid:
        test_key = uid_to_test_key(miner_uid)
        wallet_hotkey = f"{test_key}_hot_test"
        hotkey_env_var = f"TEST_HOTKEY_38_{test_key}"
        hotkey = test_key
    else:
        miner_table_csv = Path(subnet_dir) / "38" / "data" / "reg_table.csv"
        hotkey = get_hotkey_from_uid(miner_uid, miner_table_csv)
        wallet_hotkey = f"{hotkey}_hot"
        hotkey_env_var = f"HOTKEY_38_{hotkey}"

    prefixed_uid = f"{prefix}{miner_uid}"
    host_vars_file = host_vars_dir / f"{prefixed_uid}.yml"

    if is_testnet:
        wallet_name = "test_wallet"
        coldkey_env = '{{ lookup("ansible.builtin.env", "TEST_COLDKEY_38") }}'
        hotkey_env = f'{{{{ lookup("ansible.builtin.env", "{hotkey_env_var}") }}}}'
        netuid = 178
    else:
        wallet_name = "p_wallet"
        coldkey_env = '{{ lookup("ansible.builtin.env", "COLDKEY_38") }}'
        hotkey_env = f'{{{{ lookup("ansible.builtin.env", "{hotkey_env_var}") }}}}'
        netuid = 38

    created_at = datetime.now().strftime("%Y-%m-%d")
    flags_yaml = yaml.dump(run_miner_flags or {}, default_flow_style=False, indent=2)
    flags_yaml_indented = "\n".join(f"  {line}" for line in flags_yaml.splitlines())

    content = base_template.format(
        network=network,
        role=role,
        miner_uid=miner_uid,
        ip=ip,
        ssh_port=ssh_port,
        machine=machine,
        provider=provider,
        datacenter=datacenter,
        wallet_name=wallet_name,
        coldkey_env=coldkey_env,
        wallet_hotkey=wallet_hotkey,
        hotkey_env=hotkey_env,
        netuid=netuid,
        repo=repo,
        branch=branch,
        created_at=created_at,
        run_miner_flags=flags_yaml_indented,
    )

    with open(host_vars_file, "w") as f:
        f.write(content)

    inventory_line = (
        f"{prefixed_uid} "
        f"ansible_host={ip} "
        f"ansible_port={ssh_port} "
        f"ansible_user=root "
        f"ansible_ssh_private_key_file=~/.ssh/id_ed25519 "
        f"ansible_ssh_common_args='-o ForwardAgent=yes -o StrictHostKeyChecking=no'"
    )

    with open(inventory_file, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]

    if not any(line.strip() == "[miners]" for line in lines):
        lines.append("[miners]")

    lines = [line for line in lines if not line.startswith(prefixed_uid)]

    new_lines = []
    for line in lines:
        new_lines.append(line)
        if line.strip() == "[miners]":
            new_lines.append(inventory_line)

    with open(inventory_file, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    files_to_commit = ["inventory.ini", str(host_vars_file)]
    subprocess.run(["git", "add", *files_to_commit])
    subprocess.run(["git", "commit", "-m", f"auto update host_vars for {miner_uid}"], check=False)
    subprocess.run(["git", "push"], check=False)


    def select_ansible_playbooks(mode, network="mainnet"):
        """
        Returns a list of playbooks based on the deployment mode.
        Modes: 'mech0', 'mech1', 'templar'
        """
        if mode == "mech1":
            return ["mech1"]
        
        if mode == "templar":
            return ["templar"]

        if mode == "mech0":
            playbooks = ["provision"]
            if network == "local":
                playbooks.append("install_subtensor")
            playbooks.append("start")
            return playbooks
        
    playbooks = select_ansible_playbooks("templar", network=network)

    try:
        run_playbooks(
            subnet_dir=subnet_dir,
            netuid=38,
            inventory="inventory.ini",
            host=prefixed_uid,
            playbooks=playbooks,
        )
    except subprocess.CalledProcessError as e:
        print(f"Ansible failed: {e}")

parser = argparse.ArgumentParser(description="Deploy miner host with optional overrides")
parser.add_argument("--repo", type=str)
parser.add_argument("--branch", type=str)
parser.add_argument("--network", type=str)
parser.add_argument("--role", type=str)
parser.add_argument("--ip", type=str)
parser.add_argument("--miner_uid", type=str)
parser.add_argument("--provider", type=str)
parser.add_argument("--datacenter", type=str)
parser.add_argument("--machine", type=str)
parser.add_argument("--ssh_port", type=str)
parser.add_argument("--run_miner_flags", type=str)

args = parser.parse_args()

run_miner_flags = json.loads(args.run_miner_flags or "{}")

run_miner_flags["num_gpus"] = 4

add_host(
    repo=args.repo,
    branch=args.branch,
    network=args.network,
    role=args.role,
    ip=args.ip,
    miner_uid=args.miner_uid,
    provider=args.provider,
    datacenter=args.datacenter,
    machine=args.machine,
    ssh_port=args.ssh_port,
    run_miner_flags=run_miner_flags,
)
