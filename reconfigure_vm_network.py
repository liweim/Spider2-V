#!/usr/bin/env python3
"""
Reconfigure VM network settings to ensure unique MAC address and avoid IP conflicts.

Usage:
    # Reconfigure a single VM
    python reconfigure_vm_network.py --vm_path ./vm_data/Ubuntu2/Ubuntu2/Ubuntu2.vmx
    
    # Reconfigure all VMs in vm_data directory
    python reconfigure_vm_network.py --all
    
    # Dry run (show what would be changed without actually changing)
    python reconfigure_vm_network.py --vm_path ./vm_data/Ubuntu2/Ubuntu2/Ubuntu2.vmx --dry-run
"""

import os
import re
import random
import uuid
import argparse
import shutil
from datetime import datetime


def generate_mac_address():
    """Generate a unique MAC address for VMware VMs."""
    # VMware MAC address range starts with 00:0c:29
    mac = [0x00, 0x0c, 0x29,
           random.randint(0x00, 0x7f),
           random.randint(0x00, 0xff),
           random.randint(0x00, 0xff)]
    return ':'.join(map(lambda x: "%02x" % x, mac))


def backup_vmx_file(vmx_path):
    """Create a backup of the VMX file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{vmx_path}.backup_{timestamp}"
    shutil.copy2(vmx_path, backup_path)
    return backup_path


def read_current_config(vmx_path):
    """Read current network configuration from VMX file."""
    with open(vmx_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    config = {
        'generated_mac': None,
        'static_mac': None,
        'address_type': None,
        'uuid_bios': None,
        'uuid_location': None,
        'vmci_id': None,
    }
    
    # Extract current values
    match = re.search(r'ethernet0\.generatedAddress\s*=\s*"([^"]+)"', content)
    if match:
        config['generated_mac'] = match.group(1)
    
    match = re.search(r'ethernet0\.address\s*=\s*"([^"]+)"', content)
    if match:
        config['static_mac'] = match.group(1)
    
    match = re.search(r'ethernet0\.addressType\s*=\s*"([^"]+)"', content)
    if match:
        config['address_type'] = match.group(1)
    
    match = re.search(r'uuid\.bios\s*=\s*"([^"]+)"', content)
    if match:
        config['uuid_bios'] = match.group(1)
    
    match = re.search(r'uuid\.location\s*=\s*"([^"]+)"', content)
    if match:
        config['uuid_location'] = match.group(1)
    
    match = re.search(r'vmci0\.id\s*=\s*"([^"]+)"', content)
    if match:
        config['vmci_id'] = match.group(1)
    
    return config, content


def update_vmx_network(vmx_path, dry_run=False):
    """
    Update VMX file network configuration to ensure unique MAC address.
    
    Args:
        vmx_path: Path to .vmx file
        dry_run: If True, only show what would be changed
        
    Returns:
        dict with old and new values
    """
    if not os.path.exists(vmx_path):
        raise FileNotFoundError(f"VMX file not found: {vmx_path}")
    
    # Read current config
    old_config, content = read_current_config(vmx_path)
    
    # Generate new values
    new_mac = generate_mac_address()
    new_uuid_bios = str(uuid.uuid4())
    new_uuid_location = str(uuid.uuid4())
    new_vmci_id = str(random.randint(-2147483648, 2147483647))
    
    # Prepare updated content
    updated_content = content
    
    # Update MAC addresses
    if old_config['generated_mac']:
        updated_content = re.sub(
            r'ethernet0\.generatedAddress\s*=\s*"[^"]+"',
            f'ethernet0.generatedAddress = "{new_mac}"',
            updated_content
        )
    else:
        # Add if not present
        match = re.search(r'(ethernet0\.[^\n]+\n)', updated_content)
        if match:
            insert_pos = match.end()
            updated_content = (
                updated_content[:insert_pos] +
                f'ethernet0.generatedAddress = "{new_mac}"\n' +
                updated_content[insert_pos:]
            )
    
    # Update static MAC if present
    if old_config['static_mac']:
        updated_content = re.sub(
            r'ethernet0\.address\s*=\s*"[^"]+"',
            f'ethernet0.address = "{new_mac}"',
            updated_content
        )
    
    # Update or add addressType
    if old_config['address_type']:
        updated_content = re.sub(
            r'ethernet0\.addressType\s*=\s*"[^"]+"',
            'ethernet0.addressType = "generated"',
            updated_content
        )
    else:
        # Add after generatedAddress
        match = re.search(r'(ethernet0\.generatedAddress[^\n]+\n)', updated_content)
        if match:
            insert_pos = match.end()
            updated_content = (
                updated_content[:insert_pos] +
                'ethernet0.addressType = "generated"\n' +
                updated_content[insert_pos:]
            )
    
    # Update UUIDs
    if old_config['uuid_bios']:
        updated_content = re.sub(
            r'uuid\.bios\s*=\s*"[^"]+"',
            f'uuid.bios = "{new_uuid_bios}"',
            updated_content
        )
    
    if old_config['uuid_location']:
        updated_content = re.sub(
            r'uuid\.location\s*=\s*"[^"]+"',
            f'uuid.location = "{new_uuid_location}"',
            updated_content
        )
    
    # Update VMCI ID
    if old_config['vmci_id']:
        updated_content = re.sub(
            r'vmci0\.id\s*=\s*"[^"]+"',
            f'vmci0.id = "{new_vmci_id}"',
            updated_content
        )
    
    new_config = {
        'generated_mac': new_mac,
        'static_mac': new_mac if old_config['static_mac'] else None,
        'address_type': 'generated',
        'uuid_bios': new_uuid_bios,
        'uuid_location': new_uuid_location,
        'vmci_id': new_vmci_id,
    }
    
    if not dry_run:
        # Create backup
        backup_path = backup_vmx_file(vmx_path)
        print(f"  [BACKUP] Created backup: {backup_path}")
        
        # Write updated content
        with open(vmx_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"  [SUCCESS] Updated: {vmx_path}")
    else:
        print(f"  [DRY-RUN] Would update: {vmx_path}")
    
    return old_config, new_config


def find_all_vms(vm_data_dir="./vm_data"):
    """Find all VM .vmx files in the directory."""
    vms = []
    
    if not os.path.exists(vm_data_dir):
        return vms
    
    for vm_name in os.listdir(vm_data_dir):
        vm_path = os.path.join(vm_data_dir, vm_name)
        
        if not os.path.isdir(vm_path):
            continue
        
        if vm_name.endswith('.zip'):
            continue
        
        # Look for .vmx file
        vmx_path = os.path.join(vm_path, vm_name, f"{vm_name}.vmx")
        if os.path.exists(vmx_path):
            vms.append(vmx_path)
    
    return sorted(vms)


def print_config_comparison(vm_name, old_config, new_config):
    """Print before/after comparison."""
    print(f"\n{'='*80}")
    print(f"VM: {vm_name}")
    print(f"{'='*80}")
    
    print(f"\n  Generated MAC:")
    print(f"    Before: {old_config['generated_mac']}")
    print(f"    After:  {new_config['generated_mac']}")
    
    if old_config['static_mac']:
        print(f"\n  Static MAC:")
        print(f"    Before: {old_config['static_mac']}")
        print(f"    After:  {new_config['static_mac']}")
    
    print(f"\n  Address Type:")
    print(f"    Before: {old_config['address_type']}")
    print(f"    After:  {new_config['address_type']}")
    
    print(f"\n  UUID BIOS:")
    print(f"    Before: {old_config['uuid_bios']}")
    print(f"    After:  {new_config['uuid_bios']}")
    
    print(f"\n  UUID Location:")
    print(f"    Before: {old_config['uuid_location']}")
    print(f"    After:  {new_config['uuid_location']}")
    
    print(f"\n  VMCI ID:")
    print(f"    Before: {old_config['vmci_id']}")
    print(f"    After:  {new_config['vmci_id']}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconfigure VM network to avoid IP conflicts"
    )
    parser.add_argument(
        "--vm_path",
        type=str,
        help="Path to VM .vmx file to reconfigure"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Reconfigure all VMs in vm_data directory"
    )
    parser.add_argument(
        "--vm_data_dir",
        type=str,
        default="./vm_data",
        help="VM data directory (default: ./vm_data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually changing"
    )
    
    args = parser.parse_args()
    
    if not args.vm_path and not args.all:
        parser.error("Either --vm_path or --all must be specified")
    
    print("VM Network Reconfiguration Tool")
    print("=" * 80)
    
    if args.dry_run:
        print("[MODE] DRY RUN - No changes will be made")
        print()
    
    vms_to_process = []
    
    if args.all:
        vms_to_process = find_all_vms(args.vm_data_dir)
        if not vms_to_process:
            print(f"[ERROR] No VMs found in {args.vm_data_dir}")
            return 1
        print(f"[INFO] Found {len(vms_to_process)} VM(s) to reconfigure")
    else:
        if not os.path.exists(args.vm_path):
            print(f"[ERROR] VM not found: {args.vm_path}")
            return 1
        vms_to_process = [args.vm_path]
    
    success_count = 0
    error_count = 0
    
    for vmx_path in vms_to_process:
        vm_name = os.path.basename(vmx_path).replace('.vmx', '')
        
        try:
            old_config, new_config = update_vmx_network(vmx_path, dry_run=args.dry_run)
            print_config_comparison(vm_name, old_config, new_config)
            success_count += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process {vm_name}: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total VMs: {len(vms_to_process)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {error_count}")
    
    if not args.dry_run and success_count > 0:
        print("\n[SUCCESS] VM network configuration updated!")
        print("[INFO] Each VM now has a unique MAC address")
        print("[INFO] When you start these VMs, they will get unique IP addresses")
        print("\n[NEXT STEPS]")
        print("  1. Verify: python check_vm_macs.py")
        print("  2. Start VMs and check IPs: python check_vm_ips.py")
    elif args.dry_run:
        print("\n[INFO] This was a dry run. Use without --dry-run to apply changes.")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())

