"""Test GDPR Scorer with sample text"""
import sys
sys.path.insert(0, '.')

# Import only what we need to avoid dependency issues
from scrape.scrape_v2 import scrape

# Import GDPRScorer directly to avoid pipeline __init__ loading embedder
import re
from typing import Dict, List, Any


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
    
    def score_principle(self, principle_name: str, principle_data: Dict, text: str) -> Dict[str, Any]:
        keywords = principle_data["keywords"]
        text_lower = text.lower()
        matches = []
        evidence_snippets = []
        
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\w*'
            found = re.search(pattern, text_lower)
            
            if found:
                matches.append(keyword)
                idx = found.start()
                start = max(0, idx - 80)
                end = min(len(text), idx + len(keyword) + 120)
                snippet = text[start:end].strip()
                snippet = ' '.join(snippet.split())
                if len(snippet) > 20:
                    evidence_snippets.append(snippet)
        
        unique_matches = len(set(matches))
        coverage_ratio = unique_matches / len(keywords) if keywords else 0
        
        if coverage_ratio >= 0.8:
            base_score = 85 + (coverage_ratio - 0.8) * 75
        elif coverage_ratio >= 0.5:
            base_score = 65 + (coverage_ratio - 0.5) * 66.7
        elif coverage_ratio >= 0.2:
            base_score = 40 + (coverage_ratio - 0.2) * 83.3
        else:
            base_score = coverage_ratio * 200
        
        final_score = min(100, base_score * principle_data['weight'])
        
        if final_score >= 85:
            grade, assessment = 'A', 'Excellent'
        elif final_score >= 70:
            grade, assessment = 'B', 'Good'
        elif final_score >= 55:
            grade, assessment = 'C', 'Fair'
        elif final_score >= 40:
            grade, assessment = 'D', 'Weak'
        else:
            grade, assessment = 'F', 'Insufficient'
        
        return {
            'principle': principle_name,
            'score': round(final_score, 1),
            'grade': grade,
            'keywords_found': list(set(matches))[:8],
            'evidence': evidence_snippets[:4],
            'assessment': assessment,
            'coverage': f"{unique_matches}/{len(keywords)}"
        }
    
    def score_policy(self, text: str) -> Dict[str, Any]:
        principle_scores = []
        total_score = 0
        
        for principle_name, principle_data in self.GDPR_PRINCIPLES.items():
            score_result = self.score_principle(principle_name, principle_data, text)
            principle_scores.append(score_result)
            total_score += score_result['score']
        
        overall_score = total_score / len(self.GDPR_PRINCIPLES)
        
        if overall_score >= 80:
            overall_grade, summary = 'A', 'Excellent GDPR Compliance'
        elif overall_score >= 65:
            overall_grade, summary = 'B', 'Good GDPR Compliance'
        elif overall_score >= 50:
            overall_grade, summary = 'C', 'Fair GDPR Compliance'
        elif overall_score >= 35:
            overall_grade, summary = 'D', 'Weak GDPR Compliance'
        else:
            overall_grade, summary = 'F', 'Insufficient GDPR Compliance'
        
        return {
            'overall_score': overall_score,
            'overall_grade': overall_grade,
            'summary': summary,
            'principle_scores': principle_scores
        }


def test_scorer():
    print("Scraping Reddit privacy policy...")
    result = scrape("https://www.reddit.com/policies/privacy-policy")
    
    text = result.get('text', '')
    print(f"Scraped {len(text):,} characters from {len(result.get('sources', []))} pages\n")
    
    # Score it
    scorer = GDPRScorer()
    score_result = scorer.score_policy(text)
    
    print("="*70)
    print("GDPR COMPLIANCE SCORE")
    print("="*70)
    print(f"Overall Score: {score_result['overall_score']:.1f}/100")
    print(f"Grade: {score_result['overall_grade']}")
    print(f"\n{score_result['summary']}\n")
    
    print("\nPRINCIPLE SCORES:")
    print("-"*70)
    for principle in score_result['principle_scores']:
        print(f"\n{principle['principle']}: {principle['score']:.1f}/100 ({principle['grade']})")
        print(f"  Coverage: {principle['coverage']}")
        print(f"  Keywords found: {', '.join(principle['keywords_found'][:5])}")
        if principle['evidence']:
            print(f"  Evidence: {principle['evidence'][0][:150]}...")
    
    print("\n" + "="*70)
    if score_result['strengths']:
        print("\nSTRENGTHS:")
        for s in score_result['strengths']:
            print(f"  + {s}")
    
    if score_result['weaknesses']:
        print("\nWEAKNESSES:")
        for w in score_result['weaknesses']:
            print(f"  - {w}")

if __name__ == "__main__":
    test_scorer()
