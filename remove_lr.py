import os
import glob

# Remove all .lr_find* files in the directory
files_to_remove = glob.glob('.lr_find*')
for f in files_to_remove:
    try:
        os.remove(f)
        print(f"Removed physically: {f}")
    except Exception as e:
        print(f"Failed to remove {f}: {e}")

# Run git rm --cached to remove them from git tracking
os.system('git rm --cached ".lr_find*"')
print("Successfully removed from git index.")
