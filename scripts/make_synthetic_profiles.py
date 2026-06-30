import json
from pathlib import Path
import os

def create_synthetic_profiles(out_path: Path):
    profiles = [
        # 1. A true Tier-5 candidate: Shipped recommendation systems, but no trendy buzzwords
        {
            "candidate_id": "CAND_9000001",
            "profile": {
                "anonymized_name": "Tier5 Dev",
                "headline": "Software Engineer | Large scale backend",
                "summary": "Experienced software developer with 7 years working on large scale backend systems and data indexing pipelines.",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": "Software Engineer",
                "current_company": "Flipkart",
                "current_company_size": "5001-10000",
                "current_industry": "E-commerce"
            },
            "career_history": [
                {
                    "company": "Flipkart",
                    "title": "Software Engineer",
                    "start_date": "2022-01-01",
                    "end_date": None,
                    "duration_months": 54,
                    "is_current": True,
                    "industry": "E-commerce",
                    "company_size": "5001-10000",
                    "description": "Designed and shipped the product recommendation system. Built collaborative filtering algorithms, Elasticsearch matching, and ranking logic processing millions of daily queries. Handled system scalability and matching latency."
                }
            ],
            "education": [
                {
                    "institution": "Pune University",
                    "degree": "B.E.",
                    "field_of_study": "Computer Science",
                    "start_year": 2015,
                    "end_year": 2019,
                    "tier": "tier_3"
                }
            ],
            "skills": [
                {"name": "Elasticsearch", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
                {"name": "Java", "proficiency": "advanced", "endorsements": 50, "duration_months": 84},
                {"name": "Python", "proficiency": "intermediate", "endorsements": 20, "duration_months": 36}
            ],
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "signup_date": "2020-01-01",
                "last_active_date": "2026-06-28",
                "open_to_work_flag": True,
                "profile_views_received_30d": 12,
                "applications_submitted_30d": 3,
                "recruiter_response_rate": 0.85,
                "avg_response_time_hours": 6.0,
                "skill_assessment_scores": {"Elasticsearch": 85.0},
                "connection_count": 200,
                "endorsements_received": 110,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 20.0, "max": 35.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": False,
                "github_activity_score": 45.0,
                "search_appearance_30d": 150,
                "saved_by_recruiters_30d": 8,
                "interview_completion_rate": 1.0,
                "offer_acceptance_rate": 0.8,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        },
        # 2. A keyword-stuffer: Marketing manager claiming expert search engine and AI skills
        {
            "candidate_id": "CAND_9000002",
            "profile": {
                "anonymized_name": "Buzzword King",
                "headline": "Marketing Manager with RAG, FAISS, Milvus, Vector Databases, LLM experience",
                "summary": "Marketing professional specializing in SEO, vector search, RAG systems, Pinecone, and LLM optimization.",
                "location": "Noida",
                "country": "India",
                "years_of_experience": 8.0,
                "current_title": "Marketing Manager",
                "current_company": "WebTech Solutions",
                "current_company_size": "51-200",
                "current_industry": "Marketing Services"
            },
            "career_history": [
                {
                    "company": "WebTech Solutions",
                    "title": "Marketing Manager",
                    "start_date": "2021-01-01",
                    "end_date": None,
                    "duration_months": 66,
                    "is_current": True,
                    "industry": "Marketing Services",
                    "company_size": "51-200",
                    "description": "Led digital marketing and content strategies. Used SEO tools to increase site traffic. Read about RAG systems and Pinecone databases for blog writing."
                }
            ],
            "education": [
                {
                    "institution": "Amity University",
                    "degree": "BBA",
                    "field_of_study": "Marketing",
                    "start_year": 2014,
                    "end_year": 2017,
                    "tier": "tier_3"
                }
            ],
            "skills": [
                {"name": "RAG", "proficiency": "expert", "endorsements": 5, "duration_months": 2},
                {"name": "FAISS", "proficiency": "expert", "endorsements": 10, "duration_months": 1},
                {"name": "Pinecone", "proficiency": "expert", "endorsements": 8, "duration_months": 1},
                {"name": "Marketing", "proficiency": "expert", "endorsements": 90, "duration_months": 96}
            ],
            "redrob_signals": {
                "profile_completeness_score": 85.0,
                "signup_date": "2021-01-01",
                "last_active_date": "2026-06-25",
                "open_to_work_flag": True,
                "profile_views_received_30d": 50,
                "applications_submitted_30d": 12,
                "recruiter_response_rate": 0.9,
                "avg_response_time_hours": 4.0,
                "skill_assessment_scores": {},
                "connection_count": 500,
                "endorsements_received": 113,
                "notice_period_days": 15,
                "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
                "preferred_work_mode": "remote",
                "willing_to_relocate": False,
                "github_activity_score": -1.0,
                "search_appearance_30d": 200,
                "saved_by_recruiters_30d": 2,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.7,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        },
        # 3. A honeypot: Impossible skill duration (Expert skill with 0 duration_months)
        {
            "candidate_id": "CAND_9000003",
            "profile": {
                "anonymized_name": "Trap Candidate",
                "headline": "Senior AI Engineer",
                "summary": "AI professional specializing in deep learning and NLP.",
                "location": "Delhi NCR",
                "country": "India",
                "years_of_experience": 6.0,
                "current_title": "Senior AI Engineer",
                "current_company": "TechCorp",
                "current_company_size": "201-500",
                "current_industry": "Software"
            },
            "career_history": [
                {
                    "company": "TechCorp",
                    "title": "Senior AI Engineer",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 78,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "Shipped ML systems using deep learning."
                }
            ],
            "education": [],
            "skills": [
                {"name": "PyTorch", "proficiency": "expert", "endorsements": 300, "duration_months": 0}
            ],
            "redrob_signals": {
                "profile_completeness_score": 75.0,
                "signup_date": "2020-01-01",
                "last_active_date": "2026-06-25",
                "open_to_work_flag": True,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 0,
                "recruiter_response_rate": 0.5,
                "avg_response_time_hours": 24.0,
                "skill_assessment_scores": {},
                "connection_count": 20,
                "endorsements_received": 300,
                "notice_period_days": 60,
                "expected_salary_range_inr_lpa": {"min": 25.0, "max": 40.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 10.0,
                "search_appearance_30d": 50,
                "saved_by_recruiters_30d": 1,
                "interview_completion_rate": 0.8,
                "offer_acceptance_rate": -1.0,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": False
            }
        },
        # 4. A WITCH-tier-only consulting career
        {
            "candidate_id": "CAND_9000004",
            "profile": {
                "anonymized_name": "Consulting Only",
                "headline": "Senior Tech Consultant",
                "summary": "Technical consultant with 8 years of experience working across TCS and Wipro.",
                "location": "Hyderabad",
                "country": "India",
                "years_of_experience": 8.0,
                "current_title": "Senior Consultant",
                "current_company": "TCS",
                "current_company_size": "10001+",
                "current_industry": "IT Services"
            },
            "career_history": [
                {
                    "company": "TCS",
                    "title": "Senior Consultant",
                    "start_date": "2022-01-01",
                    "end_date": None,
                    "duration_months": 54,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Provided development consulting for overseas client in telecom."
                },
                {
                    "company": "Wipro",
                    "title": "Consultant",
                    "start_date": "2018-06-01",
                    "end_date": "2021-12-31",
                    "duration_months": 42,
                    "is_current": False,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Developed web applications using Java and Python."
                }
            ],
            "education": [
                {
                    "institution": "JNTU",
                    "degree": "B.Tech",
                    "field_of_study": "ECE",
                    "start_year": 2014,
                    "end_year": 2018,
                    "tier": "tier_3"
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 60},
                {"name": "Java", "proficiency": "advanced", "endorsements": 15, "duration_months": 48}
            ],
            "redrob_signals": {
                "profile_completeness_score": 85.0,
                "signup_date": "2019-01-01",
                "last_active_date": "2026-06-25",
                "open_to_work_flag": True,
                "profile_views_received_30d": 5,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.6,
                "avg_response_time_hours": 12.0,
                "skill_assessment_scores": {},
                "connection_count": 80,
                "endorsements_received": 25,
                "notice_period_days": 90,
                "expected_salary_range_inr_lpa": {"min": 10.0, "max": 18.0},
                "preferred_work_mode": "flexible",
                "willing_to_relocate": True,
                "github_activity_score": 12.0,
                "search_appearance_30d": 40,
                "saved_by_recruiters_30d": 0,
                "interview_completion_rate": 1.0,
                "offer_acceptance_rate": 1.0,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        }
    ]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
        
    print(f"Generated synthetic profiles at {out_path}")

if __name__ == '__main__':
    out = Path("/Users/varunkashyap/Desktop/Redrob-Ranker/data/synthetic/synthetic_candidates.json")
    create_synthetic_profiles(out)
