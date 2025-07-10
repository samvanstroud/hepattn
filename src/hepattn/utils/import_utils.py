import subprocess
import sys


def check_import_safe(module_name):
    code = f"""
try:
    import {module_name}
except Exception:
    exit(1)
exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code])
    return result.returncode == 0
