#!/usr/bin/env python3
"""
Cost Meter Component for Hindi Book Translation System
Tracks translation costs and provides budget alerts
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class CostEntry:
    """Represents a single cost entry"""
    timestamp: str
    characters: int
    cost: float
    total_cost: float
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class CostMeter:
    """Track translation costs against monthly budget"""
    
    # Google Cloud Translation pricing
    COST_PER_1000_CHARS = 0.20  # $0.20 per 1000 characters
    
    def __init__(self, monthly_budget: float = 1000.0, 
                 log_file: Optional[str] = None,
                 alert_threshold: float = 0.7):
        """
        Initialize cost meter
        
        Args:
            monthly_budget: Monthly budget limit in USD (default: $1000)
            log_file: Path to cost log file (default: .cost_log.json in project root)
            alert_threshold: Percentage of budget to trigger alert (default: 0.7 = 70%)
        """
        self.monthly_budget = monthly_budget
        self.alert_threshold = alert_threshold
        self.current_cost = 0.0
        self.cost_log: List[CostEntry] = []
        
        # Set up log file path
        if log_file:
            self.log_file = Path(log_file)
        else:
            # Default to project root
            self.log_file = Path(".cost_log.json")
        
        # Load existing cost data if available
        self._load_cost_log()
        
        # Calculate current month's cost
        self._calculate_current_month_cost()
    
    def _load_cost_log(self):
        """Load cost history from log file"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.cost_log = [
                        CostEntry(**entry) for entry in data.get('entries', [])
                    ]
                    print(f"Loaded {len(self.cost_log)} cost entries from log")
            except Exception as e:
                print(f"Error loading cost log: {e}")
                self.cost_log = []
        else:
            print("No existing cost log found, starting fresh")
    
    def _save_cost_log(self):
        """Save cost history to log file"""
        try:
            # Prepare data
            data = {
                'monthly_budget': self.monthly_budget,
                'alert_threshold': self.alert_threshold,
                'last_updated': datetime.now().isoformat(),
                'entries': [entry.to_dict() for entry in self.cost_log]
            }
            
            # Write to file
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving cost log: {e}")
    
    def _calculate_current_month_cost(self):
        """Calculate total cost for current month"""
        current_month = datetime.now().strftime("%Y-%m")
        self.current_cost = 0.0
        
        for entry in self.cost_log:
            # Check if entry is from current month
            if entry.timestamp.startswith(current_month):
                self.current_cost += entry.cost
        
        print(f"Current month cost: ${self.current_cost:.2f}")
    
    def add_cost(self, character_count: int, description: str = "") -> Dict:
        """
        Add translation cost for given character count
        
        Args:
            character_count: Number of characters translated
            description: Optional description of what was translated
            
        Returns:
            Dict with cost details and any alerts
        """
        # Calculate cost
        cost = (character_count / 1000) * self.COST_PER_1000_CHARS
        self.current_cost += cost
        
        # Create cost entry
        entry = CostEntry(
            timestamp=datetime.now().isoformat(),
            characters=character_count,
            cost=cost,
            total_cost=self.current_cost,
            description=description
        )
        
        # Add to log
        self.cost_log.append(entry)
        
        # Save log
        self._save_cost_log()
        
        # Check for alerts
        alerts = self._check_alerts()
        
        return {
            'characters': character_count,
            'cost': cost,
            'total_cost': self.current_cost,
            'budget_percentage': (self.current_cost / self.monthly_budget) * 100,
            'alerts': alerts
        }
    
    def _check_alerts(self) -> List[str]:
        """Check if any budget alerts should be triggered"""
        alerts = []
        
        budget_percentage = self.current_cost / self.monthly_budget
        
        # Check threshold alert
        if budget_percentage >= self.alert_threshold:
            alerts.append(
                f"⚠️ WARNING: {budget_percentage:.1%} of monthly budget used "
                f"(${self.current_cost:.2f}/${self.monthly_budget:.2f})"
            )
        
        # Check if over budget
        if budget_percentage >= 1.0:
            alerts.append(
                f"🚨 CRITICAL: Monthly budget exceeded! "
                f"(${self.current_cost:.2f}/${self.monthly_budget:.2f})"
            )
        
        # Check rapid spending (more than 10% in last hour)
        recent_cost = self._get_recent_cost(hours=1)
        if recent_cost > (self.monthly_budget * 0.1):
            alerts.append(
                f"⚡ NOTICE: High spending rate - ${recent_cost:.2f} in last hour"
            )
        
        return alerts
    
    def _get_recent_cost(self, hours: int = 1) -> float:
        """Get cost from last N hours"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_cost = 0.0
        
        for entry in reversed(self.cost_log):
            entry_time = datetime.fromisoformat(entry.timestamp).timestamp()
            if entry_time >= cutoff_time:
                recent_cost += entry.cost
            else:
                break
        
        return recent_cost
    
    def check_threshold(self, threshold: float) -> bool:
        """
        Check if cost exceeds given threshold of budget
        
        Args:
            threshold: Percentage threshold (0.0 to 1.0)
            
        Returns:
            True if threshold exceeded
        """
        return (self.current_cost / self.monthly_budget) >= threshold
    
    def get_status(self) -> Dict:
        """Get current cost meter status"""
        budget_percentage = (self.current_cost / self.monthly_budget) * 100
        
        return {
            'current_cost': self.current_cost,
            'monthly_budget': self.monthly_budget,
            'budget_percentage': budget_percentage,
            'budget_remaining': self.monthly_budget - self.current_cost,
            'alert_threshold': self.alert_threshold * 100,
            'total_characters': sum(entry.characters for entry in self.cost_log),
            'total_translations': len(self.cost_log),
            'alerts': self._check_alerts()
        }
    
    def get_cost_breakdown(self, days: int = 30) -> Dict:
        """Get cost breakdown for last N days"""
        cutoff_time = datetime.now().timestamp() - (days * 86400)
        daily_costs = {}
        
        for entry in self.cost_log:
            entry_time = datetime.fromisoformat(entry.timestamp)
            
            if entry_time.timestamp() >= cutoff_time:
                date_key = entry_time.strftime("%Y-%m-%d")
                if date_key not in daily_costs:
                    daily_costs[date_key] = {
                        'cost': 0.0,
                        'characters': 0,
                        'translations': 0
                    }
                
                daily_costs[date_key]['cost'] += entry.cost
                daily_costs[date_key]['characters'] += entry.characters
                daily_costs[date_key]['translations'] += 1
        
        return daily_costs
    
    def estimate_book_cost(self, total_characters: int) -> Dict:
        """Estimate cost for translating entire book"""
        estimated_cost = (total_characters / 1000) * self.COST_PER_1000_CHARS
        
        return {
            'total_characters': total_characters,
            'estimated_cost': estimated_cost,
            'budget_impact': (estimated_cost / self.monthly_budget) * 100,
            'can_afford': estimated_cost <= (self.monthly_budget - self.current_cost),
            'pages_estimate': total_characters / 2000  # Rough estimate: 2000 chars/page
        }
    
    def reset_monthly_cost(self):
        """Reset cost tracking for new month (keeps history)"""
        self.current_cost = 0.0
        self._calculate_current_month_cost()
        print(f"Monthly cost reset. Current month cost: ${self.current_cost:.2f}")


# Test function
def test_cost_meter():
    """Test the cost meter functionality"""
    print("Testing Cost Meter...")
    print("=" * 50)
    
    # Initialize with $100 budget for testing
    meter = CostMeter(monthly_budget=100.0, alert_threshold=0.7)
    
    # Test 1: Add some costs
    print("\n1. Adding translation costs:")
    
    # Simulate translating 50,000 characters
    result = meter.add_cost(50000, "Chapter 1 translation")
    print(f"   Cost for 50,000 chars: ${result['cost']:.2f}")
    print(f"   Total cost: ${result['total_cost']:.2f} ({result['budget_percentage']:.1f}%)")
    
    # Simulate translating 100,000 more characters
    result = meter.add_cost(100000, "Chapter 2-3 translation")
    print(f"   Cost for 100,000 chars: ${result['cost']:.2f}")
    print(f"   Total cost: ${result['total_cost']:.2f} ({result['budget_percentage']:.1f}%)")
    
    # Test 2: Check alerts
    print("\n2. Checking for alerts:")
    for alert in result['alerts']:
        print(f"   {alert}")
    
    # Test 3: Get status
    print("\n3. Cost meter status:")
    status = meter.get_status()
    print(f"   Current cost: ${status['current_cost']:.2f}")
    print(f"   Budget remaining: ${status['budget_remaining']:.2f}")
    print(f"   Total characters: {status['total_characters']:,}")
    
    # Test 4: Estimate book cost
    print("\n4. Estimating full book cost:")
    book_estimate = meter.estimate_book_cost(500000)  # 500k chars for full book
    print(f"   Estimated cost: ${book_estimate['estimated_cost']:.2f}")
    print(f"   Can afford: {book_estimate['can_afford']}")
    print(f"   Estimated pages: {book_estimate['pages_estimate']:.0f}")
    
    # Test 5: Check if threshold exceeded
    print("\n5. Threshold checks:")
    print(f"   70% threshold exceeded: {meter.check_threshold(0.7)}")
    print(f"   50% threshold exceeded: {meter.check_threshold(0.5)}")
    
    print("\n" + "=" * 50)
    print("Cost meter test complete!")


if __name__ == "__main__":
    test_cost_meter() 