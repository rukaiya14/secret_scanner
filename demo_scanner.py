#!/usr/bin/env python3
"""
Demo script showing the enhanced GitHub Secret Scanner capabilities
"""

import os
import subprocess

print("=" * 70)
print("GitHub Secret Scanner - Enhanced with PII Detection & Slack Alerts")
print("=" * 70)
print()

# Check if Slack is configured
slack_configured = bool(os.getenv('SLACK_WEBHOOK_URL'))
print(f"Slack Notifications: {'✅ Enabled' if slack_configured else '❌ Not configured'}")
print()

# Show total rules
print("Detection Capabilities:")
print("  • 62 Secret patterns (AWS, GitHub, Slack, Stripe, etc.)")
print("  • 6 PII patterns (Email, SSN, Credit Cards, Phone, etc.)")
print("  • Total: 68 detection rules")
print()

# Demo 1: Scan test file
print("-" * 70)
print("DEMO 1: Scanning test file with PII and secrets")
print("-" * 70)
print()

result = subprocess.run(
    ['python', 'scan_secrets.py', '--mode=push'],
    capture_output=False
)

print()
print("-" * 70)
print(f"Scanner Exit Code: {result.returncode}")
print("  • 0 = No secrets found (deployment allowed)")
print("  • 1 = Secrets found (deployment blocked)")
print("  • 2 = Scanner error")
print("-" * 70)
print()

# Show next steps
print("Next Steps:")
print("  1. Set up Slack webhook (see SLACK_SETUP.md)")
print("  2. Test with: python scan_secrets.py --mode=push")
print("  3. Integrate with GitHub Actions")
print("  4. Monitor alerts in Slack channel")
print()

print("For more information:")
print("  • README.md - Scanner basics")
print("  • SLACK_SETUP.md - Slack configuration")
print("  • ENHANCEMENT_SUMMARY.md - What's new")
print()
