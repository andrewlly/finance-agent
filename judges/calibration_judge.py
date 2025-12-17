from .base import BaseJudge

class CalibrationJudge(BaseJudge):
    def evaluate(self, question: dict, prediction: dict, fact_score: float) -> dict:
        """
        Evaluates how well the agent's confidence matches its actual performance.
        
        Args:
            prediction (dict): Must contain 'logs' with 'final_confidence'.
            fact_score (float): The 0-100 score from the FactJudge.
            
        Returns:
            dict: Score (0-100) where 100 is perfectly calibrated.
        """
        logs = prediction.get('logs', {})
        
        # 1. Get Agent's Self-Reported Confidence (0-100)
        # If missing, we assume 100% confidence (risky default) or 50% (neutral).
        # Assuming 100% punishes "silent failures" more, which is good for testing.
        raw_conf = logs.get('final_confidence', 100.0)
        
        confidence = max(0.0, min(100.0, float(raw_conf))) / 100.0
        
        accuracy = max(0.0, min(100.0, float(fact_score))) / 100.0

        brier_error = (confidence - accuracy) ** 2
        
        final_score = 100.0 * (1.0 - brier_error)

        if accuracy > 0.9:
            if confidence > 0.9: status = "Perfect"
            else: status = "Underconfident"
        else: 
            if confidence < 0.2: status = "Good Caution (Known Unknown)"
            elif confidence > 0.8: status = "Overconfident"
            else: status = "Weak Calibration"

        return {
            "score": round(final_score, 1),
            "reason": f"{status}. Conf: {confidence:.0%} vs Acc: {accuracy:.0%}.",
            "metadata": {
                "confidence_percent": round(confidence * 100, 1),
                "accuracy_percent": round(accuracy * 100, 1),
                "brier_error": round(brier_error, 4),
                "status": status
            }
        }

    def render(self, result: dict) -> str:
        meta = result['metadata']
        return f"""
CALIBRATION REPORT
==================
SCORE:      {result['score']}/100
STATUS:     {meta['status']}
METRICS:    Agent Confidence: {meta['confidence_percent']}%
            Actual Accuracy:  {meta['accuracy_percent']}%
==================
""".strip()