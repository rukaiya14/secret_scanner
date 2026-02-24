# Universal Deployment Blocker - Real-Time Secret & PII Detection

## Overview

This is a **universal Git hook system** that automatically scans code for secrets and PII **before every push** across **all repositories** on a developer's machine.

## Key Features

✅ **Universal Protection** - Works for ALL repositories, not just one
✅ **Real-Time Scanning** - Runs automatically before every `git push`
✅ **Deployment Blocking** - Prevents push if secrets/PII detected
✅ **Slack Alerts** - Notifies developers immediately
✅ **Zero Configuration** - Install once, works everywhere
✅ **68 Detection Rules** - Covers secrets + PII comprehensively

## How It Works

```
Developer writes code
       ↓
git add & git commit
       ↓
git push  ← Hook triggers here!
       ↓
🔍 Scanner runs automatically
       ↓
   Secrets found?
       ↓
  YES → ❌ Push BLOCKED + Slack alert
       ↓
  NO → ✅ Push allowed
```

## Installation

### Option 1: Global Installation (Recommended)

Install once for ALL repositories on your machine:

```bash
python install_git_hook.py --install-global
```

This will:
- Create `~/.git-hooks/pre-push` hook
- Configure Git to use this hook globally
- Apply to every repository you push to

### Option 2: Local Installation

Install for current repository only:

```bash
cd /path/to/your/repo
python install_git_hook.py --install-local
```

This will:
- Create `.git/hooks/pre-push` in current repo
- Only apply to this specific repository

## What Gets Detected

### Secrets (62 patterns)
- AWS Access Keys & Secret Keys
- GitHub Personal Access Tokens
- Slack API Tokens
- Stripe API Keys
- Google API Keys
- Azure Keys
- Database Connection Strings
- Private Keys (RSA, SSH, PGP)
- OAuth Tokens
- JWT Tokens
- API Keys from 50+ services

### PII (6 patterns)
- Email Addresses
- Social Security Numbers (SSN)
- Credit Card Numbers
- Phone Numbers
- IP Addresses
- Passport Numbers

## Usage Example

### Normal Push (No Secrets)

```bash
$ git push origin main

============================================================
🛡️  PRE-PUSH SECURITY CHECK
============================================================
Repository: https://github.com/user/my-project.git
Scanner: C:\...\scan_secrets.py

🔍 Scanning for secrets and PII before push...
============================================================
✅ No secrets detected

✅ Security check passed - push allowed
============================================================

Enumerating objects: 5, done.
...
```

### Blocked Push (Secrets Found)

```bash
$ git push origin main

============================================================
🛡️  PRE-PUSH SECURITY CHECK
============================================================
Repository: https://github.com/user/my-project.git
Scanner: C:\...\scan_secrets.py

🔍 Scanning for secrets and PII before push...
============================================================

🚨 SECRET DETECTED - Found 2 potential secret(s) in 1 file(s)
   High confidence: 2, Low confidence: 0

📄 config.py
   🔴 Line 15: AWS Access Key
      Secret: AKIA*************MPL
      Entropy: 3.78

   🔴 Line 42: Email Address
      Secret: john*************com
      Entropy: 3.52

============================================================
Summary: 2 high-confidence secret(s) detected
⚠️  Pipeline will FAIL - secrets must be removed before merge
============================================================

============================================================
❌ PUSH BLOCKED - Secrets or PII detected!
============================================================

Action required:
  1. Remove all secrets and PII from your code
  2. Use environment variables or secret management
  3. Run 'python scan_secrets.py --mode=push' to verify
  4. Try pushing again

To bypass this check (NOT RECOMMENDED):
  git push --no-verify

error: failed to push some refs to 'origin'
```

## Slack Integration

### Setup Slack Alerts

1. Create a Slack webhook (see SLACK_SETUP.md)
2. Set environment variable:

```powershell
# PowerShell (Windows)
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Add to PowerShell profile for persistence:
notepad $PROFILE
# Add the line above to the file
```

```bash
# Linux/Mac
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export SLACK_WEBHOOK_URL="your-url"' >> ~/.bashrc
```

3. Test it:

```bash
echo "AWS_KEY=AKIAI44QH8DHBEX7AMPLE" > test.txt
git add test.txt
git commit -m "test"
git push  # Will be blocked + Slack alert sent
```

## For Development Teams

### Distributing to All Developers

Create a setup script for your team:

```bash
# setup_security.sh
#!/bin/bash

echo "Installing universal secret scanner..."

# Clone scanner
git clone https://github.com/your-org/secret-scanner.git ~/.secret-scanner
cd ~/.secret-scanner

# Install dependencies
pip install -r requirements.txt

# Install global hook
python install_git_hook.py --install-global

# Configure Slack
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/TEAM/WEBHOOK/URL"
echo 'export SLACK_WEBHOOK_URL="..."' >> ~/.bashrc

echo "✅ Setup complete! All your git pushes will now be scanned."
```

### Onboarding New Developers

Add to your onboarding checklist:

1. Run setup script: `bash setup_security.sh`
2. Test with: `python ~/.secret-scanner/demo_scanner.py`
3. Verify Slack alerts are working
4. Done! All pushes are now protected

## Bypassing the Hook (Emergency Only)

If you absolutely must push without scanning (NOT RECOMMENDED):

```bash
git push --no-verify
```

⚠️ **Warning**: This bypasses all security checks. Only use in emergencies and immediately create a ticket to fix the issue.

## Uninstallation

### Remove Global Hook

```bash
python install_git_hook.py --uninstall-global
```

### Remove Local Hook

```bash
rm .git/hooks/pre-push
```

## Troubleshooting

### Hook not running

Check if hook is installed:
```bash
# Global
cat ~/.git-hooks/pre-push

# Local
cat .git/hooks/pre-push
```

### Scanner not found

The hook stores the absolute path to `scan_secrets.py`. If you move the scanner, reinstall the hook:

```bash
python install_git_hook.py --install-global
```

### Hook runs but doesn't detect secrets

Test the scanner directly:
```bash
python scan_secrets.py --mode=push
```

If it detects secrets directly but not via hook, check the scanner path in the hook file.

### Slack notifications not working

1. Verify webhook URL is set:
   ```bash
   echo $SLACK_WEBHOOK_URL  # Linux/Mac
   echo $env:SLACK_WEBHOOK_URL  # PowerShell
   ```

2. Test webhook directly:
   ```bash
   python slack_notifier.py
   ```

## Performance

- Typical scan: 10-15 seconds for 40-50 files
- Scales linearly with file count
- 5-second timeout per file (ReDoS protection)
- Minimal impact on developer workflow

## Security Best Practices

1. **Never bypass the hook** unless absolutely necessary
2. **Use environment variables** for secrets, never hardcode
3. **Use secret management** (AWS Secrets Manager, HashiCorp Vault, etc.)
4. **Rotate secrets immediately** if accidentally committed
5. **Review Slack alerts** promptly and take action

## Advanced Configuration

### Custom Rules

Add your own detection patterns to `secret_rules.xlsx`:

1. Open `secret_rules.xlsx`
2. Add new row with:
   - secret_type: Name of the secret
   - pattern: Regex pattern
   - description: What it detects
   - confidence_level: HIGH or LOW
3. Save file
4. Test: `python scan_secrets.py --mode=push`

### Adjusting Sensitivity

```bash
# More aggressive (catches more, more false positives)
python scan_secrets.py --mode=push --no-entropy

# Less aggressive (fewer false positives, might miss some)
python scan_secrets.py --mode=push --no-filter-comments
```

## Support

For issues:
1. Check scanner works: `python scan_secrets.py --mode=push`
2. Verify hook is installed: `cat ~/.git-hooks/pre-push`
3. Test Slack: `python slack_notifier.py`
4. Review logs in terminal output

## Files

- `install_git_hook.py` - Hook installer
- `scan_secrets.py` - Main scanner
- `slack_notifier.py` - Slack integration
- `secret_rules.xlsx` - Detection rules (68 patterns)
- `SLACK_SETUP.md` - Slack configuration guide
- `ENHANCEMENT_SUMMARY.md` - Feature documentation

---

**Your entire development team is now protected from accidentally pushing secrets or PII!** 🛡️
