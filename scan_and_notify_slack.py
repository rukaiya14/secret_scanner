#!/usr/bin/env python3
"""
Silent Secret Scanner with Slack Notifications and Explainable AI

This script runs the secret scanner silently (no terminal output) and sends
formatted notifications to Slack with explainable AI context showing WHY
each secret was detected.

Features:
- Silent mode: No terminal output (all output redirected)
- Rich Slack notifications with formatted blocks
- Explainable AI: Shows pattern matches, entropy scores, and detection reasoning
- Code context: Includes surrounding lines for each finding
- Actionable insights: Provides remediation guidance

Usage:
    python scan_and_notify_slack.py --mode push
    python scan_and_notify_slack.py --mode pull_request --base-ref main

Environment Variables:
    SLACK_WEBHOOK_URL: Your Slack webhook URL (required)
    Or pass via --webhook-url argument

Exit Codes:
    0 = Notification sent successfully (or no secrets found)
    1 = High-confidence secrets detected
    2 = Error occurred
"""

import sys
import os
import argparse
import io
import contextlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import scanner components
from scan_secrets import (
    RuleLoader, FileScanner, Validator, Finding, 
    SecretRule, ScanResult
)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("ERROR: 'requests' library not installed. Install with: pip install requests")
    sys.exit(2)


@dataclass
class ExplainableAIContext:
    """Context explaining WHY a secret was detected."""
    pattern_name: str           # Name of the regex pattern that matched
    pattern_description: str    # Human-readable description
    entropy_score: float        # Shannon entropy value
    entropy_explanation: str    # What the entropy means
    confidence_reason: str      # Why it's HIGH or LOW confidence
    code_context: List[str]     # Surrounding lines of code
    remediation: str            # How to fix this issue


class SlackNotifierWithAI:
    """Sends Slack notifications with explainable AI context."""
    
    def __init__(self, webhook_url: str):
        """
        Initialize Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
        
        if not self.webhook_url:
            raise ValueError("Slack webhook URL is required")
        
        if not HAS_REQUESTS:
            raise ImportError("'requests' library is required")
    
    def send_notification(
        self, 
        findings: List[Finding],
        scan_result: ScanResult,
        mode: str,
        repo_name: str = "Repository",
        branch: str = "main"
    ) -> bool:
        """
        Send Slack notification with explainable AI context.
        
        Args:
            findings: List of findings to report
            scan_result: Scan result with statistics
            mode: Scan mode (pull_request or push)
            repo_name: Repository name
            branch: Branch name
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not findings:
            # No secrets found - send success notification
            return self._send_success_notification(scan_result, repo_name, branch)
        
        # Build message with explainable AI
        message = self._build_message_with_ai(
            findings, scan_result, mode, repo_name, branch
        )
        
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"ERROR: Slack notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to send Slack notification: {e}")
            return False
    
    def _send_success_notification(
        self, 
        scan_result: ScanResult,
        repo_name: str,
        branch: str
    ) -> bool:
        """Send success notification when no secrets found."""
        message = {
            "text": f"✅ No Secrets Detected in {repo_name}",
            "attachments": [
                {
                    "color": "good",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": "✅ Secret Scan Passed"
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
                                    "text": f"*Files Scanned:*\n{scan_result.files_scanned}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Execution Time:*\n{scan_result.execution_time_seconds:.2f}s"
                                }
                            ]
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "🎉 *No secrets or credentials detected.* Deployment can proceed safely."
                            }
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _build_message_with_ai(
        self,
        findings: List[Finding],
        scan_result: ScanResult,
        mode: str,
        repo_name: str,
        branch: str
    ) -> dict:
        """Build Slack message with explainable AI context."""
        
        # Determine severity
        high_confidence = [f for f in findings if f.confidence == 'HIGH']
        if high_confidence:
            color = "danger"
            severity = "🚨 CRITICAL"
        else:
            color = "warning"
            severity = "⚠️  WARNING"
        
        # Build blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity}: Secrets Detected"
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
                        "text": f"*Scan Mode:*\n{mode}"
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
                    "text": f"*Summary:* {scan_result.high_confidence_count} high-confidence and {scan_result.low_confidence_count} low-confidence findings"
                }
            },
            {
                "type": "divider"
            }
        ]
        
        # Add findings with explainable AI (limit to first 5 for readability)
        findings_to_show = findings[:5]
        
        for i, finding in enumerate(findings_to_show, 1):
            # Generate explainable AI context
            ai_context = self._generate_ai_context(finding)
            
            # Mask secret
            masked_secret = self._mask_secret(finding.matched_text)
            
            # Build finding block
            confidence_icon = "🔴" if finding.confidence == 'HIGH' else "🟡"
            
            finding_text = f"*{confidence_icon} Finding #{i}: {finding.secret_type}*\n\n"
            finding_text += f"📄 *File:* `{finding.file_path}` (Line {finding.line_number})\n"
            finding_text += f"🔒 *Secret:* `{masked_secret}`\n\n"
            
            # Add explainable AI context
            finding_text += f"*🧠 Explainable AI Analysis:*\n"
            finding_text += f"• *Pattern Matched:* {ai_context.pattern_name}\n"
            finding_text += f"• *Entropy Score:* {ai_context.entropy_score:.2f} - {ai_context.entropy_explanation}\n"
            finding_text += f"• *Confidence:* {finding.confidence} - {ai_context.confidence_reason}\n\n"
            
            # Add code context
            if ai_context.code_context:
                finding_text += f"*📝 Code Context:*\n```\n"
                for line in ai_context.code_context[:3]:  # Show max 3 lines
                    finding_text += f"{line}\n"
                finding_text += "```\n\n"
            
            # Add remediation
            finding_text += f"*💡 Remediation:*\n{ai_context.remediation}"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": finding_text
                }
            })
            
            blocks.append({"type": "divider"})
        
        # Add "more findings" note if needed
        if len(findings) > 5:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_...and {len(findings) - 5} more findings. Run scanner locally for full details._"
                }
            })
        
        # Add action section
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "⛔ *Deployment has been blocked.* Remove all secrets before proceeding.\n\n"
                        "*Next Steps:*\n"
                        "1. Remove secrets from code\n"
                        "2. Rotate compromised credentials\n"
                        "3. Use environment variables or secret management\n"
                        "4. Re-run the scanner to verify"
            }
        })
        
        # Build final message
        message = {
            "text": f"{severity}: {len(findings)} Secrets Detected in {repo_name}",
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks
                }
            ]
        }
        
        return message
    
    def _generate_ai_context(self, finding: Finding) -> ExplainableAIContext:
        """Generate explainable AI context for a finding."""
        
        # Explain entropy score
        if finding.entropy >= 5.0:
            entropy_explanation = "Very high randomness (typical of real secrets)"
        elif finding.entropy >= 4.0:
            entropy_explanation = "High randomness (likely a real secret)"
        elif finding.entropy >= 3.5:
            entropy_explanation = "Moderate randomness (possible secret)"
        else:
            entropy_explanation = "Low randomness (may be a placeholder)"
        
        # Explain confidence
        if finding.confidence == 'HIGH':
            confidence_reason = "Passed all validation checks (not a placeholder, sufficient entropy)"
        else:
            confidence_reason = "Failed validation (placeholder pattern or low entropy)"
        
        # Get code context (try to read surrounding lines)
        code_context = self._get_code_context(finding.file_path, finding.line_number)
        
        # Generate remediation advice
        remediation = self._get_remediation_advice(finding.secret_type)
        
        return ExplainableAIContext(
            pattern_name=finding.secret_type,
            pattern_description=f"Detected by {finding.secret_type} pattern",
            entropy_score=finding.entropy,
            entropy_explanation=entropy_explanation,
            confidence_reason=confidence_reason,
            code_context=code_context,
            remediation=remediation
        )
    
    def _get_code_context(self, file_path: str, line_number: int, context_lines: int = 2) -> List[str]:
        """Get surrounding lines of code for context."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # Get surrounding lines
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            context = []
            for i in range(start, end):
                line_num = i + 1
                prefix = "→" if line_num == line_number else " "
                context.append(f"{prefix} {line_num:4d} | {lines[i].rstrip()}")
            
            return context
        except Exception:
            return [f"Line {line_number}: (unable to read context)"]
    
    def _get_remediation_advice(self, secret_type: str) -> str:
        """Get remediation advice based on secret type."""
        remediation_map = {
            "AWS Access Key": "1. Rotate the AWS access key immediately\n2. Use AWS Secrets Manager or IAM roles\n3. Never commit credentials to code",
            "GitHub Token": "1. Revoke the token at github.com/settings/tokens\n2. Use GitHub Actions secrets for CI/CD\n3. Use environment variables locally",
            "API Key": "1. Rotate the API key immediately\n2. Use environment variables or secret management\n3. Add .env to .gitignore",
            "Password": "1. Change the password immediately\n2. Use environment variables\n3. Consider using a password manager or vault",
            "Private Key": "1. Rotate the private key immediately\n2. Use SSH agent or key management service\n3. Never commit private keys to repositories",
            "Database URL": "1. Rotate database credentials\n2. Use environment variables\n3. Restrict database access by IP",
        }
        
        # Try to find matching advice
        for key, advice in remediation_map.items():
            if key.lower() in secret_type.lower():
                return advice
        
        # Default advice
        return "1. Remove the secret from code\n2. Rotate the credential immediately\n3. Use environment variables or secret management"
    
    @staticmethod
    def _mask_secret(secret: str, mask_percentage: float = 0.7) -> str:
        """Mask a secret for safe display (70% masked)."""
        if not secret or len(secret) <= 4:
            return '*' * len(secret)
        
        chars_to_mask = int(len(secret) * mask_percentage)
        chars_to_keep = len(secret) - chars_to_mask
        
        prefix_len = min(4, chars_to_keep)
        suffix_len = chars_to_keep - prefix_len
        
        if suffix_len > 0:
            return secret[:prefix_len] + '*' * chars_to_mask + secret[-suffix_len:]
        else:
            return secret[:prefix_len] + '*' * chars_to_mask


def run_scanner_silently(
    mode: str,
    rules_file: str,
    base_ref: Optional[str] = None,
    enable_entropy: bool = True,
    filter_comments: bool = True
) -> Tuple[List[Finding], ScanResult]:
    """
    Run the scanner silently (no terminal output).
    
    Args:
        mode: Scan mode (pull_request or push)
        rules_file: Path to rules file
        base_ref: Base branch reference
        enable_entropy: Enable entropy filtering
        filter_comments: Filter out comments
        
    Returns:
        Tuple of (findings, scan_result)
    """
    import time
    start_time = time.time()
    
    # Suppress all output
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            # Load rules
            rules = RuleLoader.load_rules(rules_file)
            
            # Discover files
            files_to_scan = FileScanner.get_files_to_scan(mode, base_ref)
            
            if not files_to_scan:
                # No files to scan
                return [], ScanResult(
                    findings=[],
                    files_scanned=0,
                    high_confidence_count=0,
                    low_confidence_count=0,
                    execution_time_seconds=time.time() - start_time
                )
            
            # Scan files
            all_findings = []
            files_scanned = 0
            
            for file_path in files_to_scan:
                try:
                    findings = FileScanner.scan_file(
                        file_path=file_path,
                        rules=rules,
                        timeout=5,
                        filter_comments=filter_comments,
                        enable_entropy=enable_entropy,
                        entropy_threshold=3.5
                    )
                    all_findings.extend(findings)
                    files_scanned += 1
                except Exception:
                    continue
            
            # Calculate statistics
            end_time = time.time()
            high_confidence_count = sum(1 for f in all_findings if f.confidence == 'HIGH')
            low_confidence_count = sum(1 for f in all_findings if f.confidence == 'LOW')
            
            scan_result = ScanResult(
                findings=all_findings,
                files_scanned=files_scanned,
                high_confidence_count=high_confidence_count,
                low_confidence_count=low_confidence_count,
                execution_time_seconds=end_time - start_time
            )
            
            return all_findings, scan_result
            
        except Exception as e:
            raise RuntimeError(f"Scanner failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Silent Secret Scanner with Slack Notifications and Explainable AI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['pull_request', 'push'],
        default='push',
        help='Scan mode: pull_request or push (default: push)'
    )
    
    parser.add_argument(
        '--rules-file',
        default='secret_rules.xlsx',
        help='Path to rules file (default: secret_rules.xlsx)'
    )
    
    parser.add_argument(
        '--base-ref',
        default=None,
        help='Base branch reference for pull request diffs'
    )
    
    parser.add_argument(
        '--webhook-url',
        default=None,
        help='Slack webhook URL (or set SLACK_WEBHOOK_URL env var)'
    )
    
    parser.add_argument(
        '--repo-name',
        default='GitHub_Secret_Scanner',
        help='Repository name for notifications'
    )
    
    parser.add_argument(
        '--branch',
        default='main',
        help='Branch name for notifications'
    )
    
    parser.add_argument(
        '--no-entropy',
        dest='enable_entropy',
        action='store_false',
        default=True,
        help='Disable entropy filtering'
    )
    
    parser.add_argument(
        '--no-filter-comments',
        dest='filter_comments',
        action='store_false',
        default=True,
        help='Include secrets in comments'
    )
    
    args = parser.parse_args()
    
    # Get webhook URL
    webhook_url = args.webhook_url or os.getenv('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("ERROR: Slack webhook URL is required. Set SLACK_WEBHOOK_URL environment variable or use --webhook-url")
        return 2
    
    try:
        # Run scanner silently
        findings, scan_result = run_scanner_silently(
            mode=args.mode,
            rules_file=args.rules_file,
            base_ref=args.base_ref,
            enable_entropy=args.enable_entropy,
            filter_comments=args.filter_comments
        )
        
        # Send Slack notification
        notifier = SlackNotifierWithAI(webhook_url)
        success = notifier.send_notification(
            findings=findings,
            scan_result=scan_result,
            mode=args.mode,
            repo_name=args.repo_name,
            branch=args.branch
        )
        
        if not success:
            print("ERROR: Failed to send Slack notification")
            return 2
        
        # Return exit code based on findings
        if scan_result.high_confidence_count > 0:
            return 1  # High-confidence secrets found
        else:
            return 0  # Success
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
