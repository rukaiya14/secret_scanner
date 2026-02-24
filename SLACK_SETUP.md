# Slack Webhook Integration Setup

This guide explains how to set up Slack notifications for the GitHub Secret Scanner.

## Features

When secrets or PII are detected, the scanner will:
- 🚨 Send a Slack alert to your configured channel
- 📊 Show detailed findings with file paths and line numbers
- 🔴 Indicate severity (HIGH/LOW confidence)
- ⛔ Confirm that deployment has been blocked

## Setup Instructions

### Step 1: Create a Slack Webhook

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name your app (e.g., "Secret Scanner Alerts")
4. Select your workspace
5. Click "Incoming Webhooks" in the left sidebar
6. Toggle "Activate Incoming Webhooks" to ON
7. Click "Add New Webhook to Workspace"
8. Select the channel where you want alerts (e.g., #security-alerts)
9. Click "Allow"
10. Copy the Webhook URL (looks like: `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX`)

### Step 2: Configure the Scanner

#### Option A: Environment Variable (Recommended)

Set the `SLACK_WEBHOOK_URL` environment variable:

**Windows (PowerShell):**
```powershell
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

**Windows (CMD):**
```cmd
set SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Linux/Mac:**
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

#### Option B: GitHub Actions Secret

For CI/CD integration:

1. Go to your GitHub repository
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Name: `SLACK_WEBHOOK_URL`
5. Value: Your webhook URL
6. Click "Add secret"

Then update `.github/workflows/secret-scan.yml`:

```yaml
- name: Run secret scanner (Pull Request)
  if: github.event_name == 'pull_request'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
  run: |
    python scan_secrets.py --mode=pull_request --base-ref=${{ github.base_ref }}
```

### Step 3: Test the Integration

Create a test file with a fake secret:

```bash
echo "AWS_ACCESS_KEY_ID=AKIAI44QH8DHBEX7AMPLE" > test_secret.txt
python scan_secrets.py --mode=push
```

You should see:
1. Console output showing the detected secret
2. A message: "✅ Slack notification sent successfully"
3. A Slack message in your configured channel

### Step 4: Clean Up Test

```bash
rm test_secret.txt
```

## Slack Message Format

The Slack alert includes:

- **Header**: Severity level (🚨 CRITICAL or ⚠️ WARNING)
- **Repository**: Repository name
- **Branch**: Branch where secrets were found
- **Author**: Commit author email
- **Total Findings**: Count of secrets/PII detected
- **Details**: Up to 10 findings with:
  - File path
  - Line number
  - Secret type
  - Masked secret value
- **Action Required**: Reminder that deployment is blocked

## Troubleshooting

### "No Slack webhook URL configured"

- Make sure `SLACK_WEBHOOK_URL` environment variable is set
- Verify the URL is correct (starts with `https://hooks.slack.com/`)

### "'requests' library not installed"

```bash
pip install requests
```

### "Slack notification failed: 404"

- Your webhook URL may be invalid or revoked
- Create a new webhook in Slack and update the URL

### "Slack notification failed: 400"

- The message payload may be too large
- The scanner limits findings to 10 per message to avoid this

## Disabling Slack Notifications

To disable Slack notifications:

```bash
unset SLACK_WEBHOOK_URL  # Linux/Mac
$env:SLACK_WEBHOOK_URL=""  # Windows PowerShell
```

Or simply don't set the environment variable.

## Security Best Practices

1. **Never commit webhook URLs to git** - Always use environment variables or secrets
2. **Rotate webhooks periodically** - Create new webhooks every 90 days
3. **Limit webhook permissions** - Only post to specific channels
4. **Monitor webhook usage** - Check Slack's webhook logs regularly
5. **Use private channels** - Send alerts to private security channels, not public ones

## Testing Without Slack

You can test the scanner without Slack by not setting the webhook URL:

```bash
python scan_secrets.py --mode=push
```

The scanner will work normally and just skip Slack notifications.

## Example Slack Alert

```
🚨 CRITICAL: Secrets/PII Detected

Repository: GitHub_Secret_Scanner
Branch: main
Author: developer@example.com
Total Findings: 3

📄 config.py
  🔴 Line 42: AWS Access Key
     `AKIA*************MPL`
  🔴 Line 43: AWS Secret Key
     `wJal*************KEY`

📄 user_data.py
  🔴 Line 15: Email Address
     `john*************com`

⛔ Deployment has been blocked. Remove all secrets/PII before proceeding.
```

## Support

For issues or questions:
1. Check the main README.md
2. Review GitHub Actions logs
3. Test locally with `python scan_secrets.py --mode=push`
4. Verify Slack webhook is active in Slack's app settings
