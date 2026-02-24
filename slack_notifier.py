#!/usr/bin/env python3
"""
Slack Webhook Notifier for Secret Scanner

Sends alerts to Slack when secrets or PII are detected.
"""

import json
import os
from typing import List, Optional
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class Finding:
    """Represents a detected secret/PII."""
    file_path: str
    line_number: int
    matched_text: str
    secret_type: str
    confidence: str
    entropy: float
    is_comment: bool


class SlackNotifier:
    """Sends Slack notifications when secrets/PII are detected."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL (or set SLACK_WEBHOOK_URL env var)
        """
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        
        if not self.webhook_url:
            print("⚠️  No Slack webhook URL configured. Set SLACK_WEBHOOK_URL environment variable.")
            self.enabled = False
        elif not HAS_REQUESTS:
            print("⚠️  'requests' library not installed. Install with: pip install requests")
            self.enabled = False
        else:
            self.enabled = True
    
    def send_alert(self, findings: List[Finding], repo_name: str = "Repository", 
                   branch: str = "main", author: str = "Developer") -> bool:
        """
        Send Slack alert for detected secrets/PII.
        
        Args:
            findings: List of findings to report
            repo_name: Repository name
            branch: Branch name
            author: Commit author
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.enabled or not findings:
            return False
        
        # Count high confidence findings
        high_confidence = [f for f in findings if f.confidence == 'HIGH']
        low_confidence = [f for f in findings if f.confidence == 'LOW']
        
        # Build Slack message
        message = self._build_message(findings, high_confidence, low_confidence, 
                                     repo_name, branch, author)
        
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Slack notification sent successfully")
                return True
            else:
                print(f"⚠️  Slack notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"⚠️  Failed to send Slack notification: {e}")
            return False
    
    def _build_message(self, findings: List[Finding], high_confidence: List[Finding],
                      low_confidence: List[Finding], repo_name: str, 
                      branch: str, author: str) -> dict:
        """Build Slack message payload."""
        
        # Determine severity color
        if high_confidence:
            color = "danger"  # Red
            severity = "🚨 CRITICAL"
        else:
            color = "warning"  # Yellow
            severity = "⚠️  WARNING"
        
        # Build findings summary
        findings_text = f"*{len(high_confidence)} high-confidence* and *{len(low_confidence)} low-confidence* findings\n\n"
        
        # Group findings by file
        findings_by_file = {}
        for finding in findings[:10]:  # Limit to first 10 to avoid huge messages
            if finding.file_path not in findings_by_file:
                findings_by_file[finding.file_path] = []
            findings_by_file[finding.file_path].append(finding)
        
        # Add findings details
        for file_path, file_findings in findings_by_file.items():
            findings_text += f"📄 *{file_path}*\n"
            for finding in file_findings:
                icon = "🔴" if finding.confidence == 'HIGH' else "🟡"
                masked_secret = self._mask_secret(finding.matched_text)
                findings_text += f"  {icon} Line {finding.line_number}: {finding.secret_type}\n"
                findings_text += f"     `{masked_secret}`\n"
            findings_text += "\n"
        
        if len(findings) > 10:
            findings_text += f"_...and {len(findings) - 10} more findings_\n"
        
        # Build Slack message
        message = {
            "text": f"{severity}: Secrets/PII Detected in {repo_name}",
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{severity}: Secrets/PII Detected"
                            }
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Repository:*\n{repo_name}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Branch:*\n{branch}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Author:*\n{author}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Total Findings:*\n{len(findings)}"
                                }
                            ]
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": findings_text
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "⛔ *Deployment has been blocked.* Remove all secrets/PII before proceeding."
                            }
                        }
                    ]
                }
            ]
        }
        
        return message
    
    @staticmethod
    def _mask_secret(secret: str, mask_percentage: float = 0.7) -> str:
        """Mask a secret for safe display (70% masked)."""
        if not secret or len(secret) <= 4:
            return '*' * len(secret)
        
        chars_to_mask = int(len(secret) * mask_percentage)
        chars_to_keep = len(secret) - chars_to_mask
        
        # Keep first few characters for context
        prefix_len = min(4, chars_to_keep)
        suffix_len = chars_to_keep - prefix_len
        
        if suffix_len > 0:
            return secret[:prefix_len] + '*' * chars_to_mask + secret[-suffix_len:]
        else:
            return secret[:prefix_len] + '*' * chars_to_mask


# Example usage
if __name__ == '__main__':
    # Test notification
    notifier = SlackNotifier()
    
    if notifier.enabled:
        test_findings = [
            Finding(
                file_path="config.py",
                line_number=42,
                matched_text="AKIAI44QH8DHBEXAMPLE",
                secret_type="AWS Access Key",
                confidence="HIGH",
                entropy=4.2,
                is_comment=False
            ),
            Finding(
                file_path="user_data.py",
                line_number=15,
                matched_text="john.doe@example.com",
                secret_type="Email Address",
                confidence="HIGH",
                entropy=3.8,
                is_comment=False
            )
        ]
        
        notifier.send_alert(
            findings=test_findings,
            repo_name="GitHub_Secret_Scanner",
            branch="main",
            author="developer@example.com"
        )
    else:
        print("Slack notifications are disabled. Set SLACK_WEBHOOK_URL to enable.")
