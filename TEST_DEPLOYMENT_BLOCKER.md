# Testing the Universal Deployment Blocker

## Complete Test Example

Follow these steps to test the deployment blocker:

### Step 1: Verify Hook is Installed

```powershell
# Check if global hook exists
cat C:\Users\Aliasgar Ghadiali\.git-hooks\pre-push

# Should show the Python hook script
```

### Step 2: Create a Test Repository

```powershell
# Create a new test directory
cd C:\Users\Aliasgar Ghadiali\Downloads
mkdir test-secret-blocker
cd test-secret-blocker

# Initialize git
git init
git config user.name "Test User"
git config user.email "test@example.com"
```

### Step 3: Create a File with Secrets

```powershell
# Create a file with fake secrets and PII
@"
# Configuration File

# AWS Credentials (FAKE - for testing only)
AWS_ACCESS_KEY_ID=AKIAI44QH8DHBEX7AMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Database Connection
DB_HOST=database.example.com
DB_USER=admin
DB_PASSWORD=SuperSecret123!

# Contact Information (PII)
ADMIN_EMAIL=john.doe@company.com
SUPPORT_PHONE=555-123-4567
CUSTOMER_SSN=123-45-6789

# API Keys
STRIPE_KEY=sk_live_4eC39HqLyjWDarjtT1zdp7dc
GITHUB_TOKEN=ghp_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r
"@ | Out-File -FilePath config.txt -Encoding UTF8
```

### Step 4: Try to Push (Will Be Blocked)

```powershell
# Add and commit the file
git add config.txt
git commit -m "Add configuration file"

# Try to push (this will be BLOCKED)
git remote add origin https://github.com/test/test-repo.git
git push -u origin main
```

### Expected Output:

```
============================================================
🛡️  PRE-PUSH SECURITY CHECK
============================================================
Repository: https://github.com/test/test-repo.git
Scanner: C:\Users\Aliasgar Ghadiali\Downloads\Major Project\scan_secrets.py

🔍 Scanning for secrets and PII before push...
============================================================

🚨 SECRET DETECTED - Found 7 potential secret(s) in 1 file(s)
   High confidence: 7, Low confidence: 0

📄 config.txt
   🔴 Line 4: AWS Access Key
      Secret: AKIA*************MPL
      Entropy: 3.78

   🔴 Line 5: AWS Secret Key
      Secret: wJal*************KEY
      Entropy: 4.52

   🔴 Line 13: Email Address
      Secret: john*************com
      Entropy: 3.52

   🔴 Line 14: US Phone Number
      Secret: 555*******567
      Entropy: 3.09

   🔴 Line 15: US Social Security Number
      Secret: 12******789
      Entropy: 3.28

   🔴 Line 18: Stripe API Key
      Secret: sk_l*************7dc
      Entropy: 4.15

   🔴 Line 19: GitHub Token
      Secret: ghp_*************q8r
      Entropy: 4.38

============================================================
Summary: 7 high-confidence secret(s) detected
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

error: failed to push some refs to 'https://github.com/test/test-repo.git'
```

### Step 5: Fix the Issues

```powershell
# Create a clean config file
@"
# Configuration File

# AWS Credentials - Use environment variables
AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY}

# Database Connection
DB_HOST=database.example.com
DB_USER=\${DB_USER}
DB_PASSWORD=\${DB_PASSWORD}

# Contact Information
ADMIN_EMAIL=\${ADMIN_EMAIL}
SUPPORT_PHONE=\${SUPPORT_PHONE}

# API Keys
STRIPE_KEY=\${STRIPE_KEY}
GITHUB_TOKEN=\${GITHUB_TOKEN}
"@ | Out-File -FilePath config.txt -Encoding UTF8

# Commit the fix
git add config.txt
git commit -m "Fix: Use environment variables instead of hardcoded secrets"
```

### Step 6: Push Again (Will Succeed)

```powershell
git push -u origin main
```

### Expected Output:

```
============================================================
🛡️  PRE-PUSH SECURITY CHECK
============================================================
Repository: https://github.com/test/test-repo.git
Scanner: C:\Users\Aliasgar Ghadiali\Downloads\Major Project\scan_secrets.py

🔍 Scanning for secrets and PII before push...
============================================================
✅ No secrets detected

✅ Security check passed - push allowed
============================================================

Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 285 bytes | 285.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/test/test-repo.git
 * [new branch]      main -> main
```

### Step 7: Test with Different Repository

```powershell
# Go to a different repository
cd C:\Users\Aliasgar Ghadiali\Downloads\Major Project

# Create a test file with a secret
echo "API_KEY=AKIAI44QH8DHBEX7AMPLE" > test_secret.txt

# Try to commit and push
git add test_secret.txt
git commit -m "Test secret"
git push origin test-secret-detection
```

**Result**: The hook will run and block this push too! It works for ALL repositories.

### Step 8: Clean Up

```powershell
# Remove test repository
cd C:\Users\Aliasgar Ghadiali\Downloads
rm -r -force test-secret-blocker

# Remove test file from main project
cd "C:\Users\Aliasgar Ghadiali\Downloads\Major Project"
git rm test_secret.txt
git commit -m "Remove test file"
```

## Quick Test Commands

Copy and paste this entire block to test:

```powershell
# Quick test
cd C:\Users\Aliasgar Ghadiali\Downloads
mkdir quick-test
cd quick-test
git init
echo "AWS_KEY=AKIAI44QH8DHBEX7AMPLE" > secret.txt
git add secret.txt
git commit -m "test"
git remote add origin https://github.com/test/test.git
git push origin main
# Should be BLOCKED!

# Clean up
cd ..
rm -r -force quick-test
```

## Testing Slack Notifications

If you have Slack configured:

```powershell
# Set Slack webhook
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Run the test above
# You should receive a Slack message with the detected secrets
```

## Bypass Test (Emergency Only)

To verify the bypass works:

```powershell
# This will skip the hook and push anyway
git push --no-verify origin main
```

⚠️ **Warning**: Only use `--no-verify` in emergencies!

## Verification Checklist

- [ ] Hook blocks push when secrets detected
- [ ] Hook allows push when no secrets found
- [ ] Slack notification sent (if configured)
- [ ] Works in different repositories
- [ ] Shows clear error messages
- [ ] Provides actionable fix instructions

## Troubleshooting

### Hook doesn't run

```powershell
# Check if hook is installed
cat C:\Users\Aliasgar Ghadiali\.git-hooks\pre-push

# Reinstall if needed
cd "C:\Users\Aliasgar Ghadiali\Downloads\Major Project"
python install_git_hook.py --install-global
```

### Scanner not found

```powershell
# Verify scanner exists
ls "C:\Users\Aliasgar Ghadiali\Downloads\Major Project\scan_secrets.py"

# If moved, reinstall hook
python install_git_hook.py --install-global
```

### Test scanner directly

```powershell
cd "C:\Users\Aliasgar Ghadiali\Downloads\Major Project"
python scan_secrets.py --mode=push
```

---

**Your deployment blocker is working if pushes with secrets are blocked!** ✅
