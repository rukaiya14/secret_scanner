#!/usr/bin/env python3
"""
Universal Git Hook Installer for Secret/PII Detection

This script installs a pre-push Git hook that:
1. Scans code before every push
2. Detects secrets and PII
3. Blocks the push if issues found
4. Sends Slack alerts to developers
5. Works for ANY repository on the developer's machine

Usage:
    python install_git_hook.py --install-global    # Install for all repos
    python install_git_hook.py --install-local     # Install for current repo only
    python install_git_hook.py --uninstall-global  # Remove from all repos
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


class GitHookInstaller:
    """Installs universal Git hooks for secret/PII detection."""
    
    # The pre-push hook script
    PRE_PUSH_HOOK = '''#!/usr/bin/env python
"""
Universal Pre-Push Hook - Secret & PII Detection

This hook runs automatically before every git push and:
- Scans all files being pushed for secrets and PII
- Blocks the push if high-confidence findings detected
- Sends Slack alert to the developer
- Works across all repositories

Installed by: GitHub Secret Scanner
"""

import sys
import os
import subprocess
import json
import platform

# Configuration
SCANNER_PATH = r"{{scanner_path}}"
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")

# Detect Python executable
if platform.system() == "Windows":
    PYTHON_EXE = sys.executable  # Use current Python on Windows
else:
    PYTHON_EXE = "python3"  # Use python3 on Unix/Mac

def get_changed_files():
    """Get list of files being pushed."""
    try:
        # Get files changed in commits being pushed
        result = subprocess.run(
            ["git", "diff", "--name-only", "@{{u}}..HEAD"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            # No upstream branch, get all tracked files
            result = subprocess.run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                check=True
            )
        
        files = [f.strip() for f in result.stdout.split("\\n") if f.strip()]
        return files
    except Exception as e:
        print(f"⚠️  Warning: Could not get changed files: {{e}}")
        return []

def run_scanner():
    """Run the secret scanner on changed files."""
    print("\\n🔍 Scanning for secrets and PII before push...")
    print("=" * 60)
    
    try:
        # Run scanner in push mode using detected Python executable
        result = subprocess.run(
            [PYTHON_EXE, SCANNER_PATH, "--mode=push"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # Print scanner output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result.returncode
    except FileNotFoundError:
        print(f"❌ Error: Scanner not found at {{SCANNER_PATH}}")
        print("   Please reinstall the Git hook or update the scanner path.")
        return 2
    except Exception as e:
        print(f"❌ Error running scanner: {{e}}")
        return 2

def main():
    """Main hook execution."""
    print("\\n" + "=" * 60)
    print("🛡️  PRE-PUSH SECURITY CHECK")
    print("=" * 60)
    
    # Get repository info
    try:
        repo_result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=False
        )
        repo_name = repo_result.stdout.strip() if repo_result.returncode == 0 else "Unknown"
    except:
        repo_name = "Unknown"
    
    print(f"Repository: {{repo_name}}")
    print(f"Scanner: {{SCANNER_PATH}}")
    print()
    
    # Run the scanner
    exit_code = run_scanner()
    
    if exit_code == 0:
        print("\\n✅ Security check passed - push allowed")
        print("=" * 60 + "\\n")
        return 0
    elif exit_code == 1:
        print("\\n" + "=" * 60)
        print("❌ PUSH BLOCKED - Secrets or PII detected!")
        print("=" * 60)
        print()
        print("Action required:")
        print("  1. Remove all secrets and PII from your code")
        print("  2. Use environment variables or secret management")
        print("  3. Run 'python scan_secrets.py --mode=push' to verify")
        print("  4. Try pushing again")
        print()
        print("To bypass this check (NOT RECOMMENDED):")
        print("  git push --no-verify")
        print()
        return 1
    else:
        print("\\n⚠️  Scanner error - push allowed but please investigate")
        print("=" * 60 + "\\n")
        return 0  # Allow push on scanner errors

if __name__ == "__main__":
    sys.exit(main())
'''
    
    def __init__(self):
        """Initialize the installer."""
        self.scanner_path = os.path.abspath("scan_secrets.py")
        
        if not os.path.exists(self.scanner_path):
            print(f"❌ Error: scan_secrets.py not found at {self.scanner_path}")
            print("   Please run this script from the scanner directory.")
            sys.exit(1)
    
    def install_global(self):
        """Install hook globally for all repositories."""
        print("🔧 Installing universal Git hook globally...")
        print()
        
        # Get global Git hooks directory
        try:
            result = subprocess.run(
                ["git", "config", "--global", "core.hooksPath"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                hooks_dir = Path(result.stdout.strip())
            else:
                # Default global hooks directory
                home = Path.home()
                hooks_dir = home / ".git-hooks"
                
                # Set global hooks path
                subprocess.run(
                    ["git", "config", "--global", "core.hooksPath", str(hooks_dir)],
                    check=True
                )
                print(f"✅ Set global hooks path: {hooks_dir}")
        except Exception as e:
            print(f"❌ Error setting up global hooks: {e}")
            return False
        
        # Create hooks directory
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Install pre-push hook
        hook_path = hooks_dir / "pre-push"
        self._write_hook(hook_path)
        
        print()
        print("=" * 60)
        print("✅ Universal Git hook installed successfully!")
        print("=" * 60)
        print()
        print("The hook will now run automatically for ALL repositories")
        print("on this machine before every 'git push'.")
        print()
        print("Configuration:")
        print(f"  • Hooks directory: {hooks_dir}")
        print(f"  • Scanner path: {self.scanner_path}")
        print(f"  • Slack alerts: {'Enabled' if os.getenv('SLACK_WEBHOOK_URL') else 'Not configured'}")
        print()
        print("To configure Slack alerts:")
        print("  $env:SLACK_WEBHOOK_URL='your-webhook-url'  # PowerShell")
        print("  export SLACK_WEBHOOK_URL='your-webhook-url'  # Linux/Mac")
        print()
        
        return True
    
    def install_local(self):
        """Install hook for current repository only."""
        print("🔧 Installing Git hook for current repository...")
        print()
        
        # Check if we're in a Git repository
        if not os.path.exists(".git"):
            print("❌ Error: Not in a Git repository")
            print("   Please run this command from the root of a Git repository.")
            return False
        
        # Get hooks directory
        hooks_dir = Path(".git/hooks")
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Install pre-push hook
        hook_path = hooks_dir / "pre-push"
        self._write_hook(hook_path)
        
        print()
        print("=" * 60)
        print("✅ Git hook installed successfully!")
        print("=" * 60)
        print()
        print("The hook will now run automatically before every 'git push'")
        print("in this repository.")
        print()
        print("Configuration:")
        print(f"  • Hook path: {hook_path}")
        print(f"  • Scanner path: {self.scanner_path}")
        print()
        
        return True
    
    def uninstall_global(self):
        """Remove global Git hook."""
        print("🗑️  Uninstalling universal Git hook...")
        print()
        
        try:
            result = subprocess.run(
                ["git", "config", "--global", "core.hooksPath"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                hooks_dir = Path(result.stdout.strip())
                hook_path = hooks_dir / "pre-push"
                
                if hook_path.exists():
                    hook_path.unlink()
                    print(f"✅ Removed hook: {hook_path}")
                else:
                    print("ℹ️  Hook not found (already uninstalled)")
                
                # Unset global hooks path if directory is empty
                if hooks_dir.exists() and not any(hooks_dir.iterdir()):
                    subprocess.run(
                        ["git", "config", "--global", "--unset", "core.hooksPath"],
                        check=False
                    )
                    print("✅ Unset global hooks path")
            else:
                print("ℹ️  No global hooks configured")
        except Exception as e:
            print(f"❌ Error uninstalling: {e}")
            return False
        
        print()
        print("✅ Universal Git hook uninstalled")
        return True
    
    def _write_hook(self, hook_path: Path):
        """Write the hook script to file."""
        # Generate hook content with scanner path (escape backslashes for Windows)
        scanner_path_escaped = self.scanner_path.replace('\\', '\\\\')
        hook_content = self.PRE_PUSH_HOOK.replace('{{scanner_path}}', scanner_path_escaped)
        
        # Write hook file
        hook_path.write_text(hook_content, encoding='utf-8')
        
        # Make executable (Unix/Mac)
        if sys.platform != 'win32':
            os.chmod(hook_path, 0o755)
        
        print(f"✅ Created hook: {hook_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Install universal Git hook for secret/PII detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install for all repositories (recommended)
  python install_git_hook.py --install-global

  # Install for current repository only
  python install_git_hook.py --install-local

  # Remove global installation
  python install_git_hook.py --uninstall-global

The hook will automatically:
  • Scan code before every push
  • Detect secrets and PII
  • Block push if issues found
  • Send Slack alerts (if configured)
  • Work across all repositories
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--install-global',
        action='store_true',
        help='Install hook globally for all repositories'
    )
    group.add_argument(
        '--install-local',
        action='store_true',
        help='Install hook for current repository only'
    )
    group.add_argument(
        '--uninstall-global',
        action='store_true',
        help='Remove global hook installation'
    )
    
    args = parser.parse_args()
    
    installer = GitHookInstaller()
    
    if args.install_global:
        success = installer.install_global()
    elif args.install_local:
        success = installer.install_local()
    elif args.uninstall_global:
        success = installer.uninstall_global()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
