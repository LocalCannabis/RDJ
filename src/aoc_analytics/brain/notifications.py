"""
Notifications - Alert system for high-impact days.

Sends Slack and email alerts when the brain predicts significant events.
Designed to be actionable: "Do this TODAY because X is happening."

"You don't check the brain - the brain tells you what matters."
"""

import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import urllib.request
import urllib.error


@dataclass
class Alert:
    """A single alert to send."""
    priority: str  # "high", "medium", "low"
    title: str
    message: str
    date: str
    lift: float
    events: List[str]
    action_items: List[str]
    
    @property
    def emoji(self) -> str:
        if self.priority == "high":
            return "🔥"
        elif self.priority == "medium":
            return "📈"
        return "📊"
        
    def to_slack_block(self) -> dict:
        """Convert to Slack Block Kit format."""
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{self.emoji} {self.title}*\n"
                    f"_{self.date}_ | Expected lift: *{self.lift:+.0%}*\n\n"
                    f"*Why:* {', '.join(self.events)}\n\n"
                    f"*Action Items:*\n" + 
                    "\n".join(f"• {item}" for item in self.action_items)
                )
            }
        }
        
    def to_email_html(self) -> str:
        """Convert to email HTML format."""
        events_html = "<br>".join(f"• {e}" for e in self.events)
        actions_html = "<br>".join(f"• {a}" for a in self.action_items)
        
        return f"""
        <div style="border-left: 4px solid {'#ff4444' if self.priority == 'high' else '#ffaa00' if self.priority == 'medium' else '#44aaff'}; 
                    padding: 15px; margin: 10px 0; background: #f9f9f9;">
            <h3 style="margin: 0;">{self.emoji} {self.title}</h3>
            <p style="color: #666; margin: 5px 0;">{self.date} | Expected lift: <strong>{self.lift:+.0%}</strong></p>
            
            <p><strong>Why:</strong></p>
            <p style="margin-left: 20px;">{events_html}</p>
            
            <p><strong>Action Items:</strong></p>
            <p style="margin-left: 20px;">{actions_html}</p>
        </div>
        """


class NotificationService:
    """
    Sends alerts when high-impact days are detected.
    
    Supported channels:
    - Slack (via webhook)
    - Email (via SMTP)
    - Console (for testing)
    """
    
    def __init__(self):
        self.brain_dir = Path(__file__).parent / "data"
        
        # Config from environment
        self.slack_webhook = os.environ.get("AOC_SLACK_WEBHOOK")
        self.email_host = os.environ.get("AOC_EMAIL_HOST", "smtp.gmail.com")
        self.email_port = int(os.environ.get("AOC_EMAIL_PORT", "587"))
        self.email_user = os.environ.get("AOC_EMAIL_USER")
        self.email_password = os.environ.get("AOC_EMAIL_PASSWORD")
        self.email_recipients = os.environ.get("AOC_EMAIL_RECIPIENTS", "").split(",")
        
        # Thresholds (as decimals: 0.25 = 25%)
        self.high_priority_threshold = 0.25  # 25%+ lift = high priority
        self.medium_priority_threshold = 0.15  # 15%+ lift = medium
        self.min_alert_threshold = 0.10  # Below 10% = no alert
        
    def load_predictions(self) -> dict:
        """Load the brain's predictions."""
        predictions_file = self.brain_dir / "predictive_calendar.json"
        if predictions_file.exists():
            with open(predictions_file) as f:
                return json.load(f)
        return {}
        
    def load_inventory_recommendations(self) -> dict:
        """Load inventory recommendations."""
        inv_file = self.brain_dir / "inventory_recommendations.json"
        if inv_file.exists():
            with open(inv_file) as f:
                return json.load(f)
        return {}
        
    def generate_alerts(self) -> List[Alert]:
        """Generate alerts for upcoming high-impact days."""
        predictions = self.load_predictions()
        inventory = self.load_inventory_recommendations()
        
        alerts = []
        
        # Use 'forecasts' key (from predictive_calendar.json)
        upcoming = predictions.get("forecasts", predictions.get("upcoming_7_days", []))
        
        for day in upcoming:
            # Handle both formats: combined_lift_pct (%) or predicted_lift (decimal)
            lift_pct = day.get("combined_lift_pct", 0)  # As percentage (e.g., 25 for 25%)
            if lift_pct:
                lift = lift_pct / 100  # Convert to decimal
            else:
                lift = day.get("predicted_lift", 0)  # Already decimal
            
            if lift < self.min_alert_threshold:
                continue
                
            # Determine priority
            if lift >= self.high_priority_threshold:
                priority = "high"
            elif lift >= self.medium_priority_threshold:
                priority = "medium"
            else:
                priority = "low"
                
            # Get events - handle both list of strings and list of dicts
            raw_events = day.get("events", [])
            if raw_events and isinstance(raw_events[0], dict):
                # New format: list of event dicts with 'name' key
                events = [e.get("name", str(e)) for e in raw_events]
            elif raw_events:
                events = raw_events
            else:
                events = [day.get("reason", "General demand increase")]
                
            # Generate action items
            action_items = self._generate_action_items(day, inventory, lift)
            
            # Create alert
            date_str = day.get("date", "Unknown")
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                friendly_date = dt.strftime("%A, %B %d")
            except:
                friendly_date = date_str
                
            alert = Alert(
                priority=priority,
                title=f"{'🚨 HIGH DEMAND' if priority == 'high' else '📈 Busy Day'} Expected",
                message=f"Expect {lift:+.0%} sales lift",
                date=friendly_date,
                lift=lift,
                events=events,
                action_items=action_items,
            )
            alerts.append(alert)
            
        # Sort by lift (highest first), then date
        alerts.sort(key=lambda a: (-a.lift, a.date))
        
        return alerts
        
    def _generate_action_items(self, day: dict, inventory: dict, lift: float) -> List[str]:
        """Generate specific action items for a day."""
        items = []
        
        # Stock levels
        if lift >= 0.25:
            items.append("Increase all category orders by 30-40%")
            items.append("Ensure extra staff coverage")
        elif lift >= 0.15:
            items.append("Increase core category orders by 20-25%")
            items.append("Consider additional afternoon staff")
        else:
            items.append("Slight increase in orders (+10-15%)")
            
        # Event-specific items - handle both list of strings and list of dicts
        raw_events = day.get("events", [])
        for event in raw_events:
            # Get event name string
            if isinstance(event, dict):
                event_name = event.get("name", "")
            else:
                event_name = str(event)
            event_lower = event_name.lower()
            
            if "new year" in event_lower or "holiday" in event_lower:
                items.append("Stock party packs, edibles, and beverages")
            elif "concert" in event_lower:
                items.append("Stock pre-rolls and vapes (grab-and-go)")
            elif "cruise" in event_lower:
                items.append("Expect tourist traffic - highlight local brands")
            elif "canucks" in event_lower or "hockey" in event_lower:
                items.append("Peak traffic after 7pm - plan closing staff")
            elif "nfl" in event_lower or "football" in event_lower:
                items.append("Sunday afternoon rush - ensure adequate coverage")
                
        # Time-specific
        if day.get("peak_hour"):
            items.append(f"Peak expected: {day['peak_hour']}")
            
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
                
        return unique_items[:5]  # Max 5 items
        
    def send_slack(self, alerts: List[Alert]) -> bool:
        """Send alerts to Slack."""
        if not self.slack_webhook:
            print("  ⚠️  Slack webhook not configured (set AOC_SLACK_WEBHOOK)")
            return False
            
        # Build message
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🧠 AOC Brain Alert",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{len(alerts)} high-impact day(s) detected*\n"
                           f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                }
            },
            {"type": "divider"}
        ]
        
        for alert in alerts[:5]:  # Max 5 alerts
            blocks.append(alert.to_slack_block())
            blocks.append({"type": "divider"})
            
        payload = json.dumps({"blocks": blocks}).encode("utf-8")
        
        try:
            req = urllib.request.Request(
                self.slack_webhook,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            print(f"  ❌ Slack error: {e}")
            return False
            
    def send_email(self, alerts: List[Alert]) -> bool:
        """Send alerts via email."""
        if not self.email_user or not self.email_password:
            print("  ⚠️  Email not configured (set AOC_EMAIL_USER and AOC_EMAIL_PASSWORD)")
            return False
            
        if not self.email_recipients or not any(self.email_recipients):
            print("  ⚠️  No email recipients configured (set AOC_EMAIL_RECIPIENTS)")
            return False
            
        # Build HTML email
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>🧠 AOC Brain Alert</h2>
            <p><strong>{len(alerts)} high-impact day(s) detected</strong></p>
            <p style="color: #666;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <hr>
            {''.join(alert.to_email_html() for alert in alerts[:5])}
            <hr>
            <p style="color: #999; font-size: 12px;">
                This alert was generated by the AOC Analytics Brain.<br>
                To adjust alert thresholds, update the NotificationService configuration.
            </p>
        </body>
        </html>
        """
        
        # Send to each recipient
        try:
            for recipient in self.email_recipients:
                recipient = recipient.strip()
                if not recipient:
                    continue
                    
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"🧠 AOC Alert: {len(alerts)} High-Impact Day(s)"
                msg["From"] = self.email_user
                msg["To"] = recipient
                
                msg.attach(MIMEText(html_body, "html"))
                
                with smtplib.SMTP(self.email_host, self.email_port) as server:
                    server.starttls()
                    server.login(self.email_user, self.email_password)
                    server.send_message(msg)
                    
            return True
        except Exception as e:
            print(f"  ❌ Email error: {e}")
            return False
            
    def print_alerts(self, alerts: List[Alert]):
        """Print alerts to console."""
        print("=" * 70)
        print("🔔 NOTIFICATIONS")
        print("   High-impact day alerts")
        print("=" * 70)
        print()
        
        if not alerts:
            print("  ✅ No high-impact days in the next 7 days")
            print()
            return
            
        print(f"  {len(alerts)} alert(s) generated")
        print()
        
        for alert in alerts:
            print("-" * 70)
            print(f"  {alert.emoji} {alert.priority.upper()} PRIORITY")
            print(f"  {alert.title}")
            print(f"  📅 {alert.date}")
            print(f"  📊 Expected lift: {alert.lift:+.0%}")
            print()
            print("  Why:")
            for event in alert.events:
                print(f"    • {event}")
            print()
            print("  Action Items:")
            for item in alert.action_items:
                print(f"    ✓ {item}")
            print()
            
    def send_all(self, alerts: List[Alert] = None) -> dict:
        """Send alerts through all configured channels."""
        if alerts is None:
            alerts = self.generate_alerts()
            
        results = {
            "alerts_generated": len(alerts),
            "slack_sent": False,
            "email_sent": False,
            "console_printed": True,
        }
        
        if not alerts:
            print("  ℹ️  No alerts to send")
            return results
            
        # Print to console
        self.print_alerts(alerts)
        
        # Send to Slack
        print("-" * 70)
        print("  Sending notifications...")
        
        if self.slack_webhook:
            results["slack_sent"] = self.send_slack(alerts)
            print(f"  Slack: {'✅ Sent' if results['slack_sent'] else '❌ Failed'}")
        else:
            print("  Slack: ⏭️ Skipped (not configured)")
            
        # Send email
        if self.email_user and self.email_password:
            results["email_sent"] = self.send_email(alerts)
            print(f"  Email: {'✅ Sent' if results['email_sent'] else '❌ Failed'}")
        else:
            print("  Email: ⏭️ Skipped (not configured)")
            
        print()
        
        # Save alert log
        self._save_alert_log(alerts, results)
        
        return results
        
    def _save_alert_log(self, alerts: List[Alert], results: dict):
        """Save alert log for auditing."""
        log_file = self.brain_dir / "notification_log.json"
        
        # Load existing log
        existing = []
        if log_file.exists():
            try:
                with open(log_file) as f:
                    existing = json.load(f)
            except:
                existing = []
                
        # Add new entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "alerts_count": len(alerts),
            "results": results,
            "alerts": [
                {
                    "priority": a.priority,
                    "date": a.date,
                    "lift": a.lift,
                    "events": a.events,
                }
                for a in alerts
            ]
        }
        
        existing.append(entry)
        
        # Keep last 100 entries
        existing = existing[-100:]
        
        with open(log_file, "w") as f:
            json.dump(existing, f, indent=2)
            
    def configure_webhook(self, webhook_url: str):
        """Set Slack webhook URL."""
        self.slack_webhook = webhook_url
        
    def configure_email(self, host: str, port: int, user: str, password: str, recipients: List[str]):
        """Configure email settings."""
        self.email_host = host
        self.email_port = port
        self.email_user = user
        self.email_password = password
        self.email_recipients = recipients


def demo():
    """Run notification demo."""
    service = NotificationService()
    
    print("=" * 70)
    print("🔔 NOTIFICATION SERVICE DEMO")
    print("=" * 70)
    print()
    
    # Generate and show alerts
    alerts = service.generate_alerts()
    
    if alerts:
        print(f"  Generated {len(alerts)} alerts from brain predictions")
    else:
        print("  No high-impact days found - creating demo alert")
        # Create a demo alert to show the format
        alerts = [
            Alert(
                priority="high",
                title="🚨 HIGH DEMAND Expected",
                message="Demo alert",
                date="Sunday, January 5",
                lift=0.35,
                events=["NFL: Sunday Football", "Cold weather (-5°C)"],
                action_items=[
                    "Increase all category orders by 30-40%",
                    "Ensure extra staff coverage",
                    "Sunday afternoon rush - ensure adequate coverage"
                ],
            )
        ]
    
    # Print alerts
    service.print_alerts(alerts)
    
    # Show configuration status
    print("-" * 70)
    print("  CHANNEL STATUS")
    print("-" * 70)
    print(f"  Slack: {'✅ Configured' if service.slack_webhook else '⚠️ Not configured'}")
    print(f"  Email: {'✅ Configured' if service.email_user else '⚠️ Not configured'}")
    print()
    print("  To enable notifications:")
    print("    export AOC_SLACK_WEBHOOK='https://hooks.slack.com/services/...'")
    print("    export AOC_EMAIL_USER='your@email.com'")
    print("    export AOC_EMAIL_PASSWORD='app-password'")
    print("    export AOC_EMAIL_RECIPIENTS='recipient1@email.com,recipient2@email.com'")
    print()


if __name__ == "__main__":
    demo()
