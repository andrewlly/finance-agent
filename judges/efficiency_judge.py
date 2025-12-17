from .base import BaseJudge

class EfficiencyJudge(BaseJudge):
    def evaluate(self, question: dict, prediction: dict) -> dict:
        """
        Evaluates efficiency based on ROI: 
        - 50% Weight: Speedup vs. Human 
        - 50% Weight: Cost Savings vs. Human
        """
        agent_time = prediction.get('duration', 0.0)
        
        agent_cost = 0.0
        logs = prediction.get('logs', {})
        if isinstance(logs, dict):
            agent_cost = logs.get('total_cost', 0.0)

        expert_mins = question.get('expert_time', 5.0)
        HUMAN_HOURLY_RATE = 100.0
        human_cost_per_min = HUMAN_HOURLY_RATE / 60.0
        
        expert_cost = expert_mins * human_cost_per_min
        expert_sec = expert_mins * 60

        if agent_time <= 0: agent_time = 1.0 
        speedup_factor = expert_sec / agent_time
        
        time_score = min(100.0, (speedup_factor / 10.0) * 100.0)

        cost_delta = expert_cost - agent_cost
        
        if expert_cost > 0:
            savings_pct = (cost_delta / expert_cost) * 100.0
        else:
            savings_pct = 0.0
            
        base_savings = max(0.0, savings_pct)

        econ_score = min(100.0, (base_savings / 90.0) * 100.0)

        final_score = (0.5 * time_score) + (0.5 * econ_score)

        return {
            "score": round(final_score, 2),
            "reason": f"Speed Score: {time_score:.1f} ({speedup_factor:.1f}x speedup). Econ Score: {econ_score:.1f} ({savings_pct:.1f}% savings).",
            "metadata": {
                "agent_time_s": round(agent_time, 2),
                "expert_time_s": expert_sec,
                "speedup_factor": round(speedup_factor, 1),
                "agent_cost": round(agent_cost, 4),
                "human_cost": round(expert_cost, 2),
                "savings_pct": round(savings_pct, 1),
                "net_savings": round(cost_delta, 4)
            }
        }

    def render(self, result: dict) -> str:
        meta = result['metadata']
        score = result['score']
        
        if score >= 90: status = "ELITE"
        elif score >= 70: status = "GOOD"
        elif score >= 50: status = "OK"
        else: status = "POOR"

        if meta['savings_pct'] > 99:
            econ_status = "Free Labor (<1% of human cost)"
        elif meta['savings_pct'] > 90:
            econ_status = "Massive Savings (>90%)"
        else:
            econ_status = f"{meta['savings_pct']}% Cheaper"

        return f"""
EFFICIENCY REPORT
=================
STATUS:     {status} ({score:.1f}/100)
SPEED:      {meta['speedup_factor']}x human speed ({meta['agent_time_s']}s vs {meta['expert_time_s']}s)
ECONOMICS:  {econ_status}
            • Human Cost: ${meta['human_cost']:.2f}
            • Agent Cost: ${meta['agent_cost']:.4f}
            • Net Savings: ${meta['net_savings']:.2f} per run
=================
""".strip()