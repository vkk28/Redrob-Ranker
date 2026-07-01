import pytest
from src.honeypot_detection import is_honeypot, is_consulting_only
from src.hard_gates import apply_gates, check_title_match

@pytest.fixture
def base_candidate():
    return {
        "candidate_id": "CAND_1234567",
        "profile": {
            "anonymized_name": "John Doe",
            "headline": "Senior AI Engineer",
            "summary": "Building production RAG and recommendation systems for 6 years.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Senior AI Engineer",
            "current_company": "Airtel",
            "current_company_size": "10001+",
            "current_industry": "Telecommunications"
        },
        "career_history": [
            {
                "company": "Airtel",
                "title": "Senior AI Engineer",
                "start_date": "2023-01-01",
                "end_date": None,
                "duration_months": 42,
                "is_current": True,
                "industry": "Telecommunications",
                "company_size": "10001+",
                "description": "Built recommendation models using PySpark and FAISS."
            },
            {
                "company": "Razorpay",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": "2022-12-31",
                "duration_months": 36,
                "is_current": False,
                "industry": "Fintech",
                "company_size": "1001-5000",
                "description": "Deployed embedding search for transaction search."
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2016,
                "end_year": 2020,
                "grade": "8.5 CGPA",
                "tier": "tier_1"
            }
        ],
        "skills": [
            {
                "name": "FAISS",
                "proficiency": "expert",
                "endorsements": 25,
                "duration_months": 36
            }
        ],
        "redrob_signals": {
            "profile_completeness_score": 95.0,
            "signup_date": "2020-01-01",
            "last_active_date": "2026-06-25",
            "open_to_work_flag": True,
            "profile_views_received_30d": 15,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"FAISS": 90.0},
            "connection_count": 120,
            "endorsements_received": 30,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 40.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 85.0,
            "search_appearance_30d": 300,
            "saved_by_recruiters_30d": 12,
            "interview_completion_rate": 0.95,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }

def test_normal_candidate_passes(base_candidate):
    assert is_honeypot(base_candidate) is False
    assert is_consulting_only(base_candidate) is False
    mult, reason = apply_gates(base_candidate)
    assert mult == 1.0
    assert reason == "passed_gates"

def test_honeypot_impossible_date_order(base_candidate):
    # Set start_date > end_date
    base_candidate["career_history"][1]["start_date"] = "2022-12-31"
    base_candidate["career_history"][1]["end_date"] = "2020-01-01"
    assert is_honeypot(base_candidate) is True
    mult, _ = apply_gates(base_candidate)
    assert mult == 0.0

def test_honeypot_future_start_date(base_candidate):
    base_candidate["career_history"][0]["start_date"] = "2028-01-01"
    assert is_honeypot(base_candidate) is True

def test_honeypot_expert_zero_duration(base_candidate):
    base_candidate["skills"][0]["proficiency"] = "expert"
    base_candidate["skills"][0]["duration_months"] = 1
    assert is_honeypot(base_candidate) is True

def test_honeypot_inconsistent_yoe(base_candidate):
    # Claims 25 years YOE but only has 6 years in history
    base_candidate["profile"]["years_of_experience"] = 25.0
    assert is_honeypot(base_candidate) is True

def test_honeypot_overlapping_roles(base_candidate):
    # Two full-time overlapping roles at different companies
    base_candidate["career_history"].append({
        "company": "TCS",
        "title": "Software Engineer",
        "start_date": "2021-01-01",
        "end_date": "2022-12-31",
        "duration_months": 24,
        "is_current": False,
        "industry": "IT Services",
        "company_size": "10001+",
        "description": "Full-time software engineering role."
    })
    # Overlaps significantly with Razorpay (2020-01-01 to 2022-12-31)
    assert is_honeypot(base_candidate) is True

def test_honeypot_education_career_mismatch(base_candidate):
    # Graduated B.Tech in 2020, but started full-time work in 2015
    base_candidate["education"][0]["end_year"] = 2020
    base_candidate["career_history"][1]["start_date"] = "2015-01-01"
    assert is_honeypot(base_candidate) is False

def test_honeypot_endorsements_disproportionate(base_candidate):
    # 500 endorsements with only 10 connections
    base_candidate["redrob_signals"]["connection_count"] = 10
    base_candidate["redrob_signals"]["endorsements_received"] = 500
    assert is_honeypot(base_candidate) is True

def test_consulting_only(base_candidate):
    # Change both companies to WITCH tier
    base_candidate["profile"]["current_company"] = "TCS"
    base_candidate["career_history"][0]["company"] = "TCS"
    base_candidate["career_history"][1]["company"] = "Wipro"
    assert is_consulting_only(base_candidate) is True
    mult, _ = apply_gates(base_candidate)
    assert mult == 0.0

def test_consulting_only_expanded(base_candidate):
    # Accenture -> Mindtree -> TCS should be consulting only
    base_candidate["profile"]["current_company"] = "TCS"
    base_candidate["career_history"][0]["company"] = "Accenture"
    base_candidate["career_history"][1]["company"] = "Mindtree"
    assert is_consulting_only(base_candidate) is True

def test_yoe_gate_floor(base_candidate):
    # Candidate with YOE < 3.0 should fail apply_gates
    base_candidate["profile"]["years_of_experience"] = 2.5
    mult, reason = apply_gates(base_candidate)
    assert mult == 0.0
    assert "disqualified_low_yoe" in reason

def test_title_match():
    assert check_title_match("Senior AI Engineer") == 1.0
    assert check_title_match("Staff Machine Learning Engineer") == 1.0
    assert check_title_match("Backend Engineer") == 0.7
    assert check_title_match("Operations Manager") == 0.1
    assert check_title_match("Marketing Specialist") == 0.1
    assert check_title_match("Mechanical Engineer") == 0.1
    assert check_title_match("Senior Mechanical Engineer") == 0.1

def test_honeypot_fictional_company_current(base_candidate):
    """Candidates at fictional/TV/movie companies should not be flagged as honeypots."""
    base_candidate["profile"]["current_company"] = "Dunder Mifflin"
    assert is_honeypot(base_candidate) is False
    mult, _ = apply_gates(base_candidate)
    assert mult > 0.0

def test_honeypot_fictional_company_history(base_candidate):
    """Candidates with fictional companies in career history should not be flagged."""
    base_candidate["career_history"][1]["company"] = "Wayne Enterprises"
    assert is_honeypot(base_candidate) is False
    mult, _ = apply_gates(base_candidate)
    assert mult > 0.0

def test_honeypot_fictional_companies_list(base_candidate):
    """Test multiple fictional company names are all ignored."""
    fictional = ["Acme Corp", "Stark Industries", "Initech", "Globex Inc", "Hooli"]
    for company in fictional:
        candidate = base_candidate.copy()
        candidate["profile"] = base_candidate["profile"].copy()
        candidate["profile"]["current_company"] = company
        assert is_honeypot(candidate) is False, f"Flagged fictional company: {company}"

