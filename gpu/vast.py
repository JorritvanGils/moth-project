# helpers/providers/deploy_vast_simple.py

import os
import re
import json
import time
import subprocess
import requests
from typing import List, Dict, Optional, Tuple, Any, Union
from dotenv import load_dotenv, find_dotenv

# Simple exception classes
class ProviderError(Exception):
    """Base exception for provider-related errors"""
    pass

class InstanceCreationError(ProviderError):
    """Raised when instance creation fails"""
    pass

class SSHConnectionError(ProviderError):
    """Raised when SSH connection fails"""
    pass

load_dotenv(find_dotenv())

# Configuration
VAST_KEY = os.getenv('VAST_KEY')
if not VAST_KEY:
    raise EnvironmentError("VAST_KEY environment variable not set")
VAST_API_BASE = "https://console.vast.ai/api/v0"
HEADERS = {"Authorization": f"Bearer {VAST_KEY}", "Content-Type": "application/json"}

TARGET_GPUS = [
    'RTX 3090',   
    # 'RTX 4090',
    # 'RTX 4080', 
    # 'RTX A6000',
    # 'A100 PCIE'
]

class SimpleVastDeployer:
    def __init__(self, label_prefix: str = "simple"):
        """
        Initialize the Vast.ai deployer
        
        Args:
            label_prefix: Prefix for instance labels (e.g., "myproject")
        """
        self.label_prefix = label_prefix
        self.instance_id = None
        self.ip = None
        self.ssh_port = None
        self.selected_offer = None
        
    def fetch_available_gpus(
        self,
        num_gpus: int = 1,
        min_download: int = 100,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch available GPU offers for target GPUs
        
        Args:
            num_gpus: Number of GPUs per instance
            min_download: Minimum download speed in Mbps
            limit: Maximum number of offers to return from API
            
        Returns:
            List of offer dictionaries sorted by price
        """
        query = {
            "rentable": {"eq": True},
            "type": {"eq": "machine"},
            "gpu_name": {"in": TARGET_GPUS},
            "num_gpus": {"eq": num_gpus},
            "inet_down": {"gt": min_download},
            "disable_bundling": True,
        }
        
        params = {
            "q": json.dumps(query),
            "order": "dph_total", 
            "limit": limit
        }
        
        try:
            resp = requests.get(
                f"{VAST_API_BASE}/bundles", 
                headers=HEADERS, 
                params=params, 
                timeout=30
            )
            resp.raise_for_status()
            offers = resp.json().get("offers", [])
            
            return sorted(offers, key=lambda x: x.get("dph_total", float('inf')))
            
        except requests.RequestException as e:
            print(f"❌ Failed to fetch GPU offers: {e}")
            return []
    
    def filter_offers(self, offers: List[Dict]) -> List[Dict]:
        """
        Apply filtering rules:
        - Max 20 per GPU type
        - Only one per location (cheapest per location)
        """
        if not offers:
            return []
        
        gpu_groups = {}
        for offer in offers:
            gpu_name = offer.get("gpu_name", "Unknown")
            if gpu_name not in gpu_groups:
                gpu_groups[gpu_name] = []
            gpu_groups[gpu_name].append(offer)
        
        filtered_offers = []
        
        for gpu_name, gpu_offers in gpu_groups.items():
            location_groups = {}
            for offer in gpu_offers:
                location = offer.get("geolocation", "Unknown")
                if location not in location_groups:
                    location_groups[location] = []
                location_groups[location].append(offer)
            
            location_cheapest = []
            for location, loc_offers in location_groups.items():
                cheapest = min(loc_offers, key=lambda x: x.get("dph_total", float('inf')))
                location_cheapest.append(cheapest)
            
            location_cheapest.sort(key=lambda x: x.get("dph_total", float('inf')))
            filtered_offers.extend(location_cheapest[:20])
        
        filtered_offers.sort(key=lambda x: x.get("dph_total", float('inf')))
        
        return filtered_offers
    
    def display_offers(self, offers: List[Dict]) -> None:
        """Display available offers in a formatted table"""
        if not offers:
            print("No offers found")
            return
            
        print("\n" + "="*110)
        print(f"{'#':<3} {'GPU':<20} {'VRAM':<6} {'Price':<12} {'Location':<20} {'Download':<10} {'Upload':<10}")
        print("="*110)
        
        for idx, offer in enumerate(offers, 1):
            gpu_name = offer.get("gpu_name", "Unknown")[:20]
            vram_gb = round(offer.get("gpu_ram", 0) / 1024)
            price = offer.get("dph_total", 0)
            loc = offer.get("geolocation", "Unknown")[:20]
            download = offer.get("inet_down", 0)
            upload = offer.get("inet_up", 0)
            
            if price < 1.0:
                price_str = f"${price:.3f}/hr"
            else:
                price_str = f"${price:.2f}/hr"
                
            if download > 1000:
                download_str = f"{download/1000:.1f}Gbps"
            else:
                download_str = f"{download}Mbps"
                
            if upload > 1000:
                upload_str = f"{upload/1000:.1f}Gbps"
            else:
                upload_str = f"{upload}Mbps"
                
            print(f"{idx:<3} {gpu_name:<20} {vram_gb:>4}GB  {price_str:<12} {loc:<20} {download_str:>8} {upload_str:>8}")
        
        print("="*110)
        print(f"\n📊 Total offers shown: {len(offers)} (max 10 per GPU, cheapest per location)")
    
    def parse_selection(self, selection: str, max_idx: int) -> List[int]:
        """
        Parse user selection which can be:
        - Single number: "5"
        - Range: "1-5"
        - Comma-separated: "1,3,5"
        - Mixed: "1-3,5,7-9"
        
        Returns:
            List of selected indices
        """
        selected_indices = set()
        
        # Split by comma
        parts = selection.split(',')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check if it's a range (contains -)
            if '-' in part:
                try:
                    start, end = map(int, part.split('-', 1))
                    if start < 1 or end > max_idx or start > end:
                        print(f"⚠️  Invalid range {part} (must be between 1 and {max_idx})")
                        continue
                    selected_indices.update(range(start, end + 1))
                except ValueError:
                    print(f"⚠️  Invalid range format: {part}")
            else:
                # Single number
                try:
                    idx = int(part)
                    if 1 <= idx <= max_idx:
                        selected_indices.add(idx)
                    else:
                        print(f"⚠️  Number {idx} out of range (1-{max_idx})")
                except ValueError:
                    print(f"⚠️  Invalid number: {part}")
        
        return sorted(selected_indices)
    
    def select_offers_interactive(self, offers: List[Dict]) -> List[Dict]:
        """
        Interactive selection of GPU offers - supports ranges
        
        Returns:
            List of selected offer dictionaries
        """
        if not offers:
            return []
            
        self.display_offers(offers)
        
        while True:
            try:
                choice = input(f"\nSelect GPU offer(s) (e.g., '5', '1-5', '1,3,5', '1-5,7,9-12', or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    return []
                
                if choice.lower() == 'all':
                    # Select all offers
                    selected_indices = list(range(1, len(offers) + 1))
                else:
                    selected_indices = self.parse_selection(choice, len(offers))
                
                if not selected_indices:
                    print("No valid selections made. Please try again.")
                    continue
                
                selected_offers = [offers[i-1] for i in selected_indices]
                
                # print(f"\n✅ Selected {len(selected_offers)} offer(s):")
                # for i, offer in enumerate(selected_offers, 1):
                #     print(f"   {i}. {offer['num_gpus']}x {offer['gpu_name']} "
                #           f"at {offer.get('geolocation', 'Unknown')} - "
                #           f"${offer.get('dph_total', 0):.2f}/hr")
                
                # confirm = input("\nProceed with these offers? (Y/n): ").strip().lower()
                # if confirm not in ('n', 'no'):
                #     return selected_offers
                return selected_offers
                
            except KeyboardInterrupt:
                print("\nSelection cancelled")
                return []
    
    def try_offers_sequential(self, offers: List[Dict], disk_size: int = 100) -> bool:
        """
        Try multiple offers sequentially until one works
        
        Args:
            offers: List of offers to try
            disk_size: Disk size in GB
            
        Returns:
            True if one succeeds, False if all fail
        """
        print(f"\n🔄 Will try {len(offers)} offer(s) sequentially until one works...")
        
        for idx, offer in enumerate(offers, 1):
            print(f"\n{'='*60}")
            print(f"Trying offer {idx}/{len(offers)}:")
            print(f"  {offer['num_gpus']}x {offer['gpu_name']} "
                  f"at {offer.get('geolocation', 'Unknown')} - "
                  f"${offer.get('dph_total', 0):.2f}/hr")
            print(f"{'='*60}")
            
            self.selected_offer = offer
            
            # Create instance
            if not self.create_instance(offer, disk_size):
                print(f"❌ Failed to create instance for offer {idx}")
                continue
            
            # Wait for ready
            if self.wait_for_ready(max_retries=10):
                print(f"\n✅ Success with offer {idx}!")
                return True
            
            # If we get here, instance failed to become ready
            print(f"\n❌ Offer {idx} failed to become ready")
            
            # Terminate the failed instance
            if self.instance_id:
                print(f"🧹 Cleaning up failed instance {self.instance_id}...")
                self.terminate_instance(confirm=False)
            
            # Reset state for next attempt
            self.instance_id = None
            self.ip = None
            self.ssh_port = None
            
            # Small pause between attempts
            if idx < len(offers):
                print("\n⏳ Waiting 5 seconds before next attempt...")
                time.sleep(5)
        
        print("\n❌ All offers failed")
        return False
    
    def create_instance(
        self, 
        offer: Dict, 
        disk_size: int = 100,
        open_button_port: int = 1111
    ) -> bool:
        """
        Create a Vast.ai instance from selected offer
        
        Args:
            offer: Selected offer dictionary
            disk_size: Disk size in GB
            open_button_port: Starting port for port mapping
            
        Returns:
            True if successful, False otherwise
        """
        offer_id = offer.get("id")
        if not offer_id:
            print("❌ No offer ID found")
            return False
            
        self.selected_offer = offer
        
        gpu_abbrev = self._get_gpu_abbrev(offer.get("gpu_name", ""))
        location_code = self._get_location_code(offer.get("geolocation", ""))
        label = f"{self.label_prefix}-{location_code}-{offer['num_gpus']}{gpu_abbrev}"
        
        portal_config = (
            f"localhost:{open_button_port}:{open_button_port+10000}:/Instance Portal|"
            "localhost:8080:18080:/:Jupyter|"
            "localhost:8080:8080:/terminals/1:Jupyter Terminal"
        )
        
        cmd = [
            "vastai", "create", "instance", str(offer_id),
            "--image", "vastai/base-image:cuda-12.8.1-auto",
            "--env", f"-p {open_button_port}:{open_button_port} "
                     f"-p {open_button_port+1}:{open_button_port+1} "
                     f"-p {open_button_port+2}:{open_button_port+2} "
                     f"-p {open_button_port+3}:{open_button_port+3} "
                     f"-e OPEN_BUTTON_PORT={open_button_port} "
                     f"-e OPEN_BUTTON_TOKEN=1 "
                     f"-e JUPYTER_DIR=/ "
                     f"-e PORTAL_CONFIG=\"{portal_config}\"",
            "--onstart-cmd", "touch ~/.no_auto_tmux",
            "--disk", str(disk_size),
            "--label", label,
            "--ssh",
            "--direct"
        ]
        
        print("\n🚀 Creating instance...")
        print(" ".join(cmd))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            match = re.search(r"'new_contract':\s*(\d+)", result.stdout)
            if match:
                self.instance_id = int(match.group(1))
                print(f"✅ Instance created with ID: {self.instance_id}")
                return True
            else:
                print("❌ Could not extract instance ID from response")
                print(f"Response: {result.stdout}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create instance: {e.stderr}")
            return False
    
    def wait_for_ready(self, max_retries: int = 10, retry_interval: int = 10, auto_generate_inventory: bool = True) -> bool:
        """
        Wait for instance to be ready and extract connection info
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_interval: Seconds between retries
            auto_generate_inventory: Whether to auto-generate Ansible inventory
            
        Returns:
            True if instance is ready, False otherwise
        """
        if not self.instance_id:
            print("❌ No instance ID available")
            return False
            
        print(f"\n⏳ Waiting for instance {self.instance_id} to be ready...")
        
        for attempt in range(1, max_retries + 1):
            try:
                result = subprocess.run(
                    ["vastai", "show", "instance", str(self.instance_id), "--raw"],
                    capture_output=True, text=True, check=True
                )
                instance_info = json.loads(result.stdout)
                
                ports = instance_info.get("ports", {})
                status = instance_info.get("actual_status", "")
                
                if status == "running":
                    direct_ip = instance_info.get("public_ipaddr")
                    direct_port = None
                    proxy_host = instance_info.get("ssh_host")
                    proxy_port = instance_info.get("ssh_port")
                    
                    ssh_mappings = ports.get("22/tcp", [])
                    if ssh_mappings:
                        direct_port = int(ssh_mappings[0].get("HostPort", 22))
                    
                    print(f"\n✅ Instance ready (attempt {attempt}/{max_retries})")
                    
                    connection_working = False
                    
                    if direct_ip and direct_port:
                        print(f"\n📡 Connection option 1 (Direct):")
                        print(f"   IP: {direct_ip}")
                        print(f"   Port: {direct_port}")
                        print(f"   Command: ssh -p {direct_port} root@{direct_ip}")
                        
                        if self._test_ssh_connection(direct_ip, direct_port):
                            self.ip = direct_ip
                            self.ssh_port = direct_port
                            connection_working = True
                    
                    if not connection_working and proxy_host and proxy_port:
                        if not (direct_ip == proxy_host and direct_port == proxy_port):
                            print(f"\n📡 Connection option 2 (Proxy):")
                            print(f"   Host: {proxy_host}")
                            print(f"   Port: {proxy_port}")
                            print(f"   Command: ssh -p {proxy_port} root@{proxy_host}")
                            
                            if self._test_ssh_connection(proxy_host, proxy_port):
                                self.ip = proxy_host
                                self.ssh_port = proxy_port
                                connection_working = True
                    
                    if connection_working:
                        print(f"\n✅ Using connection: ssh -p {self.ssh_port} root@{self.ip}")
                        
                        if auto_generate_inventory:
                            self.generate_ansible_inventory()
                        
                        return True
                    else:
                        print(f"\n❌ Both SSH connection methods failed")
                        return False
                
                if status in ["error", "offline", "stopped"]:
                    print(f"\n❌ Instance entered bad state: {status}")
                    return False
                        
                if attempt < max_retries:
                    print(f"\r⏳ Attempt {attempt}/{max_retries}: Status: {status}, Waiting...", end="", flush=True)
                    time.sleep(retry_interval)
                    
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                print(f"\n❌ Error checking instance status: {e}")
                if attempt < max_retries:
                    time.sleep(retry_interval)
        
        print("\n❌ Instance never became ready")
        return False

    def _test_ssh_connection(self, host: str, port: int, timeout: int = 5) -> bool:
        """
        Test SSH connection to verify it's working
        
        Args:
            host: Hostname or IP
            port: SSH port
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection works, False otherwise
        """
        try:
            result = subprocess.run(
                ["nc", "-z", "-w", str(timeout), host, str(port)],
                capture_output=True,
                timeout=timeout + 1
            )
            if result.returncode == 0:
                print(f"   ✅ Connection test successful")
                return True
            else:
                print(f"   ❌ Connection test failed")
                return False
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                result = subprocess.run(
                    ["ssh", "-p", str(port), "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", 
                    f"root@{host}", "echo 'test'"],
                    capture_output=True,
                    timeout=timeout + 2
                )
                return result.returncode == 0
            except:
                return False

    def ssh_to_instance(self) -> None:
        """SSH into the instance"""
        if not self.ip or not self.ssh_port:
            print("❌ No connection info available")
            return
            
        try:
            print(f"\n🔌 Connecting to {self.ip}:{self.ssh_port}...")
            print(f"💡 Tip: Use 'exit' to return to the menu\n")
            cmd = ["ssh", "-p", str(self.ssh_port), f"root@{self.ip}"]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nSSH session terminated")
        except Exception as e:
            print(f"❌ SSH connection failed: {e}")
    
    def terminate_instance(self, confirm: bool = True) -> bool:
        """
        Terminate the current instance
        
        Args:
            confirm: Whether to ask for confirmation
            
        Returns:
            True if terminated successfully
        """
        if not self.instance_id:
            print("❌ No instance to terminate")
            return False
            
        if confirm:
            response = input(f"\n⚠️  Terminate instance {self.instance_id}? (y/N): ").strip().lower()
            if response not in ('y', 'yes'):
                print("Termination cancelled")
                return False
        
        try:
            resp = requests.delete(
                f"{VAST_API_BASE}/instances/{self.instance_id}/", 
                headers=HEADERS
            )
            resp.raise_for_status()
            print(f"✅ Instance {self.instance_id} terminated")
            
            self.instance_id = None
            self.ip = None
            self.ssh_port = None
            self.selected_offer = None
            
            return True
            
        except requests.RequestException as e:
            print(f"❌ Failed to terminate instance: {e}")
            return False
    
    def _get_gpu_abbrev(self, gpu_name: str) -> str:
        """Generate abbreviation from GPU name"""
        if '4090' in gpu_name:
            return '49'
        elif '4080' in gpu_name:
            return '48'
        elif '3090' in gpu_name:
            return '39'
        elif 'A6000' in gpu_name:
            return 'A6'
        elif 'A100' in gpu_name:
            return 'A1'
        else:
            words = gpu_name.split()
            if words:
                return ''.join(w[0] for w in words[:2])
            return 'XX'
    
    def _get_location_code(self, location: str) -> str:
        """Generate location code from location string"""
        parts = location.split(',')
        if len(parts) > 1:
            code = parts[-1].strip()[:2].upper()
            return code
        elif location:
            return location[:2].upper()
        return 'XX'

    # Add this to your SimpleVastDeployer class (add after the existing methods)

    def generate_ansible_inventory(self, inventory_path: str = "../ansible/inventory.ini") -> bool:
        """
        Generate an Ansible inventory file for the deployed instance
        
        Args:
            inventory_path: Path to save the inventory file (relative to script location)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.ip or not self.ssh_port:
            print("❌ No connection info available for Ansible inventory")
            return False
        
        try:
            # Get the absolute path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            abs_inventory_path = os.path.join(script_dir, inventory_path)
            
            # Create inventory content
            inventory_content = f"""[vast_gpus]
    {self.instance_id} ansible_host={self.ip} ansible_port={self.ssh_port} ansible_user=root ansible_ssh_common_args='-o StrictHostKeyChecking=no -o ForwardAgent=yes'

    [vast_gpus:vars]
    instance_id={self.instance_id}
    gpu_name={self.selected_offer.get('gpu_name', 'Unknown') if self.selected_offer else 'Unknown'}
    gpu_count={self.selected_offer.get('num_gpus', 1) if self.selected_offer else 1}
    location={self.selected_offer.get('geolocation', 'Unknown') if self.selected_offer else 'Unknown'}
    ssh_command=ssh -p {self.ssh_port} root@{self.ip}
    """
            
            # Write to file
            with open(abs_inventory_path, 'w') as f:
                f.write(inventory_content)
            
            print(f"\n✅ Ansible inventory generated at: {abs_inventory_path}")
            print(f"   Host: {self.ip}:{self.ssh_port}")
            print(f"   Instance ID: {self.instance_id}")
            
            # Also create a simple vars file if needed
            vars_path = os.path.join(os.path.dirname(abs_inventory_path), "host_vars", f"{self.instance_id}.yml")
            os.makedirs(os.path.dirname(vars_path), exist_ok=True)
            
            vars_content = f"""---
    # Host variables for instance {self.instance_id}
    ansible_host: {self.ip}
    ansible_port: {self.ssh_port}
    ansible_user: root
    instance_id: {self.instance_id}
    gpu_name: {self.selected_offer.get('gpu_name', 'Unknown') if self.selected_offer else 'Unknown'}
    gpu_count: {self.selected_offer.get('num_gpus', 1) if self.selected_offer else 1}
    location: {self.selected_offer.get('geolocation', 'Unknown') if self.selected_offer else 'Unknown'}
    ssh_command: ssh -p {self.ssh_port} root@{self.ip}
    """
            
            with open(vars_path, 'w') as f:
                f.write(vars_content)
            
            print(f"✅ Host vars generated at: {vars_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate Ansible inventory: {e}")
            return False

    def run_ansible_playbook(self, playbook: str, inventory_path: str = "../ansible/inventory.ini") -> bool:
        """
        Run an Ansible playbook on the deployed instance
        
        Args:
            playbook: Name of the playbook to run (e.g., 'provision.yml')
            inventory_path: Path to the inventory file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.ip or not self.ssh_port:
            print("❌ No instance available to run Ansible on")
            return False
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ansible_dir = os.path.join(script_dir, "../ansible")
        inventory_abs_path = os.path.join(script_dir, inventory_path)
        playbook_path = os.path.join(ansible_dir, "books", playbook)
        
        if not os.path.exists(playbook_path):
            print(f"❌ Playbook not found: {playbook_path}")
            return False
        
        cmd = [
            "ansible-playbook",
            "-i", inventory_abs_path,
            playbook_path
        ]
        
        print(f"\n📖 Running Ansible playbook: {playbook}")
        print(" ".join(cmd))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Ansible playbook '{playbook}' completed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ Ansible playbook failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
        except FileNotFoundError:
            print("❌ Ansible not found. Please install ansible: pip install ansible")
            return False
        except Exception as e:
            print(f"❌ Failed to run Ansible playbook: {e}")
            return False        


def list_running_instances(label_prefix: str = None) -> List[Dict]:
    """List all running instances, optionally filtered by label prefix"""
    try:
        resp = requests.get(f"{VAST_API_BASE}/instances/", headers=HEADERS)
        resp.raise_for_status()
        instances = resp.json().get("instances", [])
        
        if label_prefix:
            instances = [i for i in instances if i.get("label", "").startswith(label_prefix)]
            
        return instances
        
    except requests.RequestException as e:
        print(f"❌ Failed to list instances: {e}")
        return []


def show_running_instances():
    """Show all running instances with simple prefix"""
    instances = list_running_instances(label_prefix="simple")
    
    if not instances:
        print("\n📭 No running instances with prefix 'simple'")
        return
    
    print("\n" + "="*80)
    print(f"📋 Running 'simple' instances: {len(instances)}")
    print("="*80)
    
    for inst in instances:
        status = inst.get('actual_status', 'unknown')
        label = inst.get('label', 'N/A')
        instance_id = inst.get('id', 'N/A')
        ip = inst.get('public_ipaddr', 'N/A')
        
        ports = inst.get('ports', {})
        ssh_port = 'N/A'
        if '22/tcp' in ports and ports['22/tcp']:
            ssh_port = ports['22/tcp'][0].get('HostPort', 'N/A')
        
        print(f"ID: {instance_id} | Status: {status} | Label: {label}")
        print(f"   IP: {ip} | SSH Port: {ssh_port}")
        print(f"   SSH: ssh -p {ssh_port} root@{ip}")
        print("-" * 40)


def main():
    """Main interactive function"""
    print("\n🚀 Simple Vast.ai GPU Deployer")
    print("="*50)
    
    show_running_instances()
    
    deployer = SimpleVastDeployer(label_prefix="simple")
    
    # Get desired action before deployment
    print("\n📋 What would you like to do after deployment?")
    print("1. SSH into instance")
    print("2. Run 'Setup moth project' playbook")
    print("3. Run custom playbook")
    print("4. Just deploy and exit")
    
    action_choice = input("\nSelect option (1-4): ").strip()
    
    print("\n🔍 Fetching available GPUs...")
    raw_offers = deployer.fetch_available_gpus(num_gpus=1, min_download=100)
    
    if not raw_offers:
        print("❌ No GPU offers found")
        return
    
    filtered_offers = deployer.filter_offers(raw_offers)
    print(f"📊 Found {len(raw_offers)} raw offers, filtered to {len(filtered_offers)}")
    
    selected_offers = deployer.select_offers_interactive(filtered_offers)
    if not selected_offers:
        print("Selection cancelled")
        return
    
    if deployer.try_offers_sequential(selected_offers, disk_size=100):
        if action_choice == "1":
            deployer.ssh_to_instance()
        elif action_choice == "2":
            deployer.run_ansible_playbook("setup_moth_project.yml")
        elif action_choice == "3":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            books_dir = os.path.join(script_dir, "../ansible/books")
            if os.path.exists(books_dir):
                playbooks = [f for f in os.listdir(books_dir) if f.endswith('.yml')]
                if playbooks:
                    print("\n📚 Available playbooks:")
                    for i, pb in enumerate(playbooks, 1):
                        print(f"   {i}. {pb}")
                    pb_choice = input("\nSelect playbook number (or name): ").strip()
                    try:
                        idx = int(pb_choice) - 1
                        if 0 <= idx < len(playbooks):
                            deployer.run_ansible_playbook(playbooks[idx])
                        else:
                            print("Invalid selection")
                    except ValueError:
                        deployer.run_ansible_playbook(pb_choice)
                else:
                    print("❌ No playbooks found in ../ansible/books/")
            else:
                print("❌ Ansible books directory not found")
        elif action_choice == "4":
            print(f"\n✅ Instance deployed successfully!")
            print(f"📡 Connection info:")
            print(f"   SSH command: ssh -p {deployer.ssh_port} root@{deployer.ip}")
            print(f"   To terminate: python {__file__} --terminate {deployer.instance_id}")
        
        if action_choice not in ["1", "2", "3"]:
            print("\n👋 Exiting. Instance is still running.")
            print(f"To reconnect later: ssh -p {deployer.ssh_port} root@{deployer.ip}")
            print(f"To terminate later: python {__file__} --terminate {deployer.instance_id}")
    else:
        print("\n❌ Could not find a working instance among the selected offers.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--terminate" and len(sys.argv) > 2:
            instance_id = sys.argv[2]
            deployer = SimpleVastDeployer()
            deployer.instance_id = int(instance_id)
            deployer.terminate_instance(confirm=True)
        elif sys.argv[1] == "--list":
            show_running_instances()
        else:
            print("Usage:")
            print("  python deploy_vast_simple.py              # Interactive mode")
            print("  python deploy_vast_simple.py --list       # List running instances")
            print("  python deploy_vast_simple.py --terminate <id>  # Terminate instance")
    else:
        main()

