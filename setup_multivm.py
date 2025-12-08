"""
Setup script for multi-VM environment
Cleans lock files and creates VM copies for multiprocessing
"""
import os
import shutil
import subprocess
import sys

def clean_lock_files(vm_dir):
    """Clean all .lck files and directories in VM directory"""
    print(f"Cleaning lock files in {vm_dir}...")
    lock_count = 0
    
    if not os.path.exists(vm_dir):
        print(f"Directory {vm_dir} does not exist")
        return
    
    for root, dirs, files in os.walk(vm_dir):
        # Check directories
        for d in dirs:
            if d.endswith('.lck'):
                lock_path = os.path.join(root, d)
                try:
                    shutil.rmtree(lock_path)
                    print(f"  Removed lock directory: {lock_path}")
                    lock_count += 1
                except Exception as e:
                    print(f"  Failed to remove {lock_path}: {e}")
        
        # Check files
        for f in files:
            if f.endswith('.lck'):
                lock_path = os.path.join(root, f)
                try:
                    os.remove(lock_path)
                    print(f"  Removed lock file: {lock_path}")
                    lock_count += 1
                except Exception as e:
                    print(f"  Failed to remove {lock_path}: {e}")
    
    print(f"Cleaned {lock_count} lock files/directories")
    
    # Kill any running VMware processes
    print("\nKilling VMware processes...")
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'vmware.exe', '/T'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        subprocess.run(['taskkill', '/F', '/IM', 'vmware-vmx.exe', '/T'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        print("VMware processes killed")
    except Exception as e:
        print(f"Error killing processes: {e}")


def copy_vm(source_vm, target_vm):
    """Copy VM directory and rename internal structure"""
    if os.path.exists(target_vm):
        print(f"{target_vm} already exists, skipping...")
        return False
    
    print(f"Copying {source_vm} to {target_vm}...")
    print("This may take a while (several GB of data)...")
    
    try:
        # Copy the entire VM directory
        shutil.copytree(source_vm, target_vm)
        print(f"Successfully copied to {target_vm}")
        
        # Get source and target VM names
        source_vm_name = os.path.basename(source_vm)
        target_vm_name = os.path.basename(target_vm)
        
        # Rename the inner directory from source_name to target_name
        # Structure: vm_data/Ubuntu1/Ubuntu0 -> vm_data/Ubuntu1/Ubuntu1
        inner_source_dir = os.path.join(target_vm, source_vm_name)
        inner_target_dir = os.path.join(target_vm, target_vm_name)
        
        if os.path.exists(inner_source_dir) and inner_source_dir != inner_target_dir:
            print(f"Renaming {inner_source_dir} to {inner_target_dir}...")
            os.rename(inner_source_dir, inner_target_dir)
            
            # Rename .vmx and related files
            for ext in ['vmx', 'vmxf', 'vmsd', 'nvram']:
                old_file = os.path.join(inner_target_dir, f"{source_vm_name}.{ext}")
                new_file = os.path.join(inner_target_dir, f"{target_vm_name}.{ext}")
                if os.path.exists(old_file):
                    print(f"Renaming {os.path.basename(old_file)} to {os.path.basename(new_file)}...")
                    os.rename(old_file, new_file)
            
            print(f"VM structure updated for {target_vm_name}")
        
        return True
    except Exception as e:
        print(f"Error copying VM: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    vm_data_dir = "./vm_data"
    
    # Clean lock files in all VMs
    print("="*60)
    print("Step 1: Cleaning lock files")
    print("="*60)
    
    for vm_name in os.listdir(vm_data_dir):
        vm_path = os.path.join(vm_data_dir, vm_name)
        if os.path.isdir(vm_path) and not vm_name.endswith('.zip'):
            clean_lock_files(vm_path)
    
    # Check how many VMs we have
    print("\n" + "="*60)
    print("Step 2: Checking existing VMs")
    print("="*60)
    
    existing_vms = []
    for vm_name in sorted(os.listdir(vm_data_dir)):
        vm_path = os.path.join(vm_data_dir, vm_name)
        if os.path.isdir(vm_path) and not vm_name.endswith('.zip'):
            existing_vms.append((vm_name, vm_path))
            print(f"  Found: {vm_name}")
    
    if not existing_vms:
        print("ERROR: No VMs found in vm_data directory!")
        return
    
    # Ask user how many VMs they need
    print("\n" + "="*60)
    print("Step 3: Creating VM copies (if needed)")
    print("="*60)
    
    num_vms_needed = int(input(f"\nHow many VMs do you need? (currently have {len(existing_vms)}): "))
    
    if num_vms_needed <= len(existing_vms):
        print(f"You already have {len(existing_vms)} VMs, no need to create more")
        print("\nExisting VMs:")
        for vm_name, vm_path in existing_vms:
            vmx_file = os.path.join(vm_path, vm_name, f"{vm_name}.vmx")
            if os.path.exists(vmx_file):
                print(f"  {vmx_file}")
        return
    
    # Create additional VM copies
    source_vm_name, source_vm_path = existing_vms[0]
    
    for i in range(len(existing_vms), num_vms_needed):
        target_vm_name = f"Ubuntu{i}"
        target_vm_path = os.path.join(vm_data_dir, target_vm_name)
        
        if copy_vm(source_vm_path, target_vm_path):
            print(f"\nNOTE: You may need to update {target_vm_name}.vmx file")
            print("      to change the VM display name and regenerate UUID")
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    
    print("\nAvailable VMs:")
    for i in range(num_vms_needed):
        vm_name = f"Ubuntu{i}"
        vmx_file = os.path.join(vm_data_dir, vm_name, vm_name, f"{vm_name}.vmx")
        if os.path.exists(vmx_file):
            print(f"  {vmx_file}")
    
    print("\nTo use multiple VMs in run_all.py:")
    print("  Modify run_all.py to use different VM paths for each process")
    print("  Or use --num_envs 1 to run single process mode")


if __name__ == "__main__":
    main()

