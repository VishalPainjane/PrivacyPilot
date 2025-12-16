"""
Simplified GDPR Compliance Scorer

Scores privacy policies against 7 core GDPR principles with clear ratings.
"""

from typing import Dict, List, Any
import re


class GDPRScorer:
    """Score privacy policies against GDPR principles"""
    
    # GDPR Core Principles with scoring keywords
    GDPR_PRINCIPLES = {
        "Lawfulness & Transparency": {
            "description": "Clear legal basis and transparent processing",
            "keywords": [
                "legal", "basis", "lawful", "legitimate", "consent", "agree", "permission",
                "contract", "transparent", "clear", "inform", "notice", "disclose", 
                "explain", "understand", "comply", "law", "regulation", "gdpr"
            ],
            "weight": 1.0
        },
        "Purpose Limitation": {
            "description": "Data collected for specific, explicit purposes",
            "keywords": [
                "purpose", "reason", "why we collect", "use", "process", "processing",
                "collect", "gather", "obtain", "specific", "explicit", "intended",
                "limited", "described", "outlined"
            ],
            "weight": 1.0
        },
        "Data Minimization": {
            "description": "Only necessary data is collected",
            "keywords": [
                "necessary", "need", "require", "essential", "relevant", "adequate",
                "minimum", "limited", "only collect", "types of", "information we collect",
                "data we collect", "personal information", "personal data"
            ],
            "weight": 0.8
        },
        "Accuracy": {
            "description": "Data kept accurate and up-to-date",
            "keywords": [
                "accurate", "correct", "up-to-date", "update", "modify", "change",
                "rectify", "amend", "edit", "maintain", "verify", "ensure accuracy"
            ],
            "weight": 0.7
        },
        "Storage Limitation": {
            "description": "Data retained only as long as necessary",
            "keywords": [
                "retention", "retain", "keep", "store", "storage", "how long",
                "period", "duration", "delete", "remove", "erase", "dispose",
                "as long as", "until", "archive"
            ],
            "weight": 0.9
        },
        "Security": {
            "description": "Appropriate security measures in place",
            "keywords": [
                "security", "secure", "protect", "protection", "safeguard", "safety",
                "encrypt", "encryption", "ssl", "https", "firewall", "access control",
                "technical", "organizational", "measures", "prevent", "unauthorized"
            ],
            "weight": 1.0
        },
        "User Rights": {
            "description": "Access, rectification, erasure, and portability rights",
            "keywords": [
                "rights", "right", "access", "view", "download", "export", "copy",
                "delete", "deletion", "erasure", "remove", "rectification", "correct",
                "portability", "withdraw", "opt-out", "unsubscribe", "object",
                "request", "control", "manage", "choice"
            ],
            "weight": 1.0
        }
    }
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def score_principle(self, principle_name: str, principle_data: Dict, text: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Score a single GDPR principle with flexible keyword matching
        
        Returns:
            {
                'principle': str,
                'score': float (0-100),
                'grade': str (A-F),
                'found': bool,
                'evidence': List[str],
                'assessment': str
            }
        """
        keywords = principle_data["keywords"]
        
        # Count keyword matches with word boundary matching
        text_lower = text.lower()
        matches = []
        evidence_snippets = []
        match_positions = []
        
        for keyword in keywords:
            # Use word boundary regex for better matching
            pattern = r'\b' + re.escape(keyword) + r'\w*'
            found = re.search(pattern, text_lower)
            
            if found:
                matches.append(keyword)
                # Find evidence snippets (context around match)
                idx = found.start()
                match_positions.append(idx)
                start = max(0, idx - 80)
                end = min(len(text), idx + len(keyword) + 120)
                snippet = text[start:end].strip()
                # Clean up snippet
                snippet = ' '.join(snippet.split())  # Normalize whitespace
                if len(snippet) > 20:  # Only add meaningful snippets
                    evidence_snippets.append(snippet)
        
        # Calculate score based on keyword coverage
        unique_matches = len(set(matches))
        total_keyword_mentions = len(matches)
        coverage_ratio = unique_matches / len(keywords) if keywords else 0
        
        # Score calculation - more strict requirements
        # - Find at least 30% of keywords → minimum 45 points
        # - Find 60% of keywords → 75 points
        # - Find 85%+ of keywords → 90+ points
        # Bonus for multiple mentions (shows thorough coverage)
        repetition_factor = min(1.1, total_keyword_mentions / unique_matches if unique_matches > 0 else 1)
        
        if coverage_ratio >= 0.85:
            base_score = 88 + (coverage_ratio - 0.85) * 80  # 88-100
        elif coverage_ratio >= 0.6:
            base_score = 70 + (coverage_ratio - 0.6) * 72  # 70-88
        elif coverage_ratio >= 0.3:
            base_score = 45 + (coverage_ratio - 0.3) * 83.3  # 45-70
        elif coverage_ratio >= 0.1:
            base_score = 20 + (coverage_ratio - 0.1) * 125  # 20-45
        else:
            base_score = coverage_ratio * 200  # 0-20
        
        # Apply repetition bonus (max 10% boost)
        base_score = min(100, base_score * repetition_factor)
        
        # Bonus for evidence spread (keywords found in multiple locations)
        if len(match_positions) > 1:
            spread = max(match_positions) - min(match_positions)
            if spread > 1000:  # Keywords across document
                base_score = min(100, base_score + 5)
        
        # Apply weight
        final_score = min(100, base_score * principle_data['weight'])
        
        # Determine grade - stricter thresholds
        if final_score >= 90:
            grade = 'A'
            assessment = 'Excellent - Comprehensive GDPR compliance'
        elif final_score >= 75:
            grade = 'B'
            assessment = 'Good - Strong coverage with minor gaps'
        elif final_score >= 60:
            grade = 'C'
            assessment = 'Fair - Basic compliance, improvements needed'
        elif final_score >= 45:
            grade = 'D'
            assessment = 'Weak - Significant gaps in coverage'
        else:
            grade = 'F'
            assessment = 'Insufficient - Major GDPR requirements missing'
        
        return {
            'principle': principle_name,
            'description': principle_data['description'],
            'score': round(final_score, 1),
            'grade': grade,
            'found': len(matches) > 0,
            'keywords_found': list(set(matches))[:8],  # Top 8 unique matches
            'evidence': evidence_snippets[:4],  # Top 4 evidence snippets
            'assessment': assessment,
            'coverage': f"{unique_matches}/{len(keywords)}"
        }
    
    def score_policy(self, text: str, retrieved_chunks: List[Dict] = None) -> Dict[str, Any]:
        """
        Score entire privacy policy against GDPR
        
        Args:
            text: Full privacy policy text
            retrieved_chunks: Optional retrieved evidence chunks
        
        Returns:
            {
                'overall_score': float,
                'overall_grade': str,
                'principle_scores': List[Dict],
                'summary': str,
                'strengths': List[str],
                'weaknesses': List[str]
            }
        """
        # Extract evidence text from chunks
        evidence = []
        if retrieved_chunks:
            evidence = [chunk.get('text', '') for chunk in retrieved_chunks]
        
        # Score each principle
        principle_scores = []
        total_score = 0
        
        for principle_name, principle_data in self.GDPR_PRINCIPLES.items():
            score_result = self.score_principle(principle_name, principle_data, text, evidence)
            principle_scores.append(score_result)
            total_score += score_result['score']
        
        # Calculate overall score
        overall_score = total_score / len(self.GDPR_PRINCIPLES)
        
        # Overall grade with more reasonable thresholds
        if overall_score >= 80:
            overall_grade = 'A'
            summary = '🟢 Excellent GDPR Compliance - Policy demonstrates strong adherence to GDPR principles'
        elif overall_score >= 65:
            overall_grade = 'B'
            summary = '🟡 Good GDPR Compliance - Policy covers most GDPR requirements with minor gaps'
        elif overall_score >= 50:
            overall_grade = 'C'
            summary = '🟠 Fair GDPR Compliance - Policy addresses core requirements but has notable gaps'
        elif overall_score >= 35:
            overall_grade = 'D'
            summary = '🔴 Weak GDPR Compliance - Policy has significant gaps in GDPR coverage'
        else:
            overall_grade = 'F'
            summary = '⛔ Insufficient GDPR Compliance - Policy lacks essential GDPR requirements'
        
        # Identify strengths (scores >= 70)
        strengths = [
            f"{s['principle']} ({s['grade']}): {s['coverage']} keywords"
            for s in principle_scores if s['score'] >= 70
        ]
        
        # Identify weaknesses (scores < 50)
        weaknesses = [
            f"{s['principle']} ({s['grade']}): Only {s['coverage']} keywords found"
            for s in principle_scores if s['score'] < 50
        ]
        
        return {
            'overall_score': round(overall_score, 1),
            'overall_grade': overall_grade,
            'principle_scores': principle_scores,
            'summary': summary,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': self._generate_recommendations(principle_scores)
        }
    
    def _generate_recommendations(self, principle_scores: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on scores"""
        recommendations = []
        
        for score in principle_scores:
            if score['score'] < 60:
                if score['principle'] == "Lawfulness & Transparency":
                    recommendations.append("📋 Clearly state the legal basis for processing personal data")
                elif score['principle'] == "Purpose Limitation":
                    recommendations.append("🎯 Explicitly define specific purposes for data collection")
                elif score['principle'] == "Data Minimization":
                    recommendations.append("📊 Specify that only necessary data is collected")
                elif score['principle'] == "Accuracy":
                    recommendations.append("✏️ Describe procedures for keeping data accurate and up-to-date")
                elif score['principle'] == "Storage Limitation":
                    recommendations.append("⏰ Define clear data retention periods and deletion policies")
                elif score['principle'] == "Security":
                    recommendations.append("🔒 Detail technical and organizational security measures")
                elif score['principle'] == "User Rights":
                    recommendations.append("👤 Clearly explain user rights (access, deletion, portability)")
        
        return recommendations
