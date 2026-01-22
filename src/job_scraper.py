"""
Job Scraper Module

Provides IT job listings data for the RAG chatbot demo.
Includes mock Copenhagen job data from top Danish tech companies.

Key Functions:
- get_copenhagen_it_jobs(): Returns formatted job listings text
- LinkedInJobScraper: Class for future LinkedIn integration (placeholder)

Companies Featured:
- Novo Nordisk, Maersk, Spotify, LEGO, Microsoft
- Danske Bank, Pleo, Unity, Trustpilot, Zendesk, Lunar, Too Good To Go

Note: Currently uses mock data. Real scraping would require API access.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import time
import re


@dataclass
class JobListing:
    """Represents a job listing."""
    title: str
    company: str
    location: str
    description: str
    url: str
    posted_date: str
    job_type: str  # Full-time, Part-time, Contract, etc.
    experience_level: str  # Entry, Mid, Senior
    salary: Optional[str] = None
    skills: Optional[List[str]] = None
    
    def to_text(self) -> str:
        """Convert job listing to text format for RAG ingestion."""
        skills_str = ", ".join(self.skills) if self.skills else "Not specified"
        salary_str = self.salary if self.salary else "Not specified"
        
        return f"""
JOB LISTING
===========
Title: {self.title}
Company: {self.company}
Location: {self.location}
Job Type: {self.job_type}
Experience Level: {self.experience_level}
Salary: {salary_str}
Posted: {self.posted_date}
Required Skills: {skills_str}

Description:
{self.description}

Apply here: {self.url}
---
"""


class LinkedInJobScraper:
    """
    Scraper for LinkedIn job listings.
    Note: LinkedIn has rate limits and may block scrapers.
    For production, consider using LinkedIn's official API.
    """
    
    BASE_URL = "https://www.linkedin.com/jobs/search"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def search_jobs(
        self,
        keywords: str = "software developer",
        location: str = "Copenhagen, Denmark",
        num_jobs: int = 25
    ) -> List[JobListing]:
        """
        Search for jobs on LinkedIn.
        
        Note: This may not work reliably due to LinkedIn's anti-scraping measures.
        Use get_sample_jobs() for testing instead.
        """
        jobs = []
        
        params = {
            'keywords': keywords,
            'location': location,
            'trk': 'public_jobs_jobs-search-bar_search-submit',
            'position': 1,
            'pageNum': 0,
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find job cards
            job_cards = soup.find_all('div', class_='base-card')
            
            for card in job_cards[:num_jobs]:
                try:
                    title_elem = card.find('h3', class_='base-search-card__title')
                    company_elem = card.find('h4', class_='base-search-card__subtitle')
                    location_elem = card.find('span', class_='job-search-card__location')
                    link_elem = card.find('a', class_='base-card__full-link')
                    
                    if title_elem and company_elem:
                        job = JobListing(
                            title=title_elem.get_text(strip=True),
                            company=company_elem.get_text(strip=True),
                            location=location_elem.get_text(strip=True) if location_elem else location,
                            description="See full job posting for details.",
                            url=link_elem['href'] if link_elem else "",
                            posted_date="Recently posted",
                            job_type="Full-time",
                            experience_level="Mid-level",
                        )
                        jobs.append(job)
                        
                except Exception as e:
                    continue
                    
            # Add delay to be respectful
            time.sleep(1)
            
        except Exception as e:
            print(f"LinkedIn scraping failed: {e}")
            print("Using sample data instead...")
            return self.get_sample_jobs()
        
        if not jobs:
            return self.get_sample_jobs()
            
        return jobs
    
    def get_sample_jobs(self) -> List[JobListing]:
        """
        Returns realistic sample job data for Copenhagen IT market.
        Use this for testing or when scraping fails.
        """
        sample_jobs = [
            JobListing(
                title="Senior Software Engineer",
                company="Novo Nordisk",
                location="Copenhagen, Denmark",
                description="""We are looking for a Senior Software Engineer to join our Digital Health team.

Responsibilities:
- Design and develop scalable backend services using Python and Go
- Lead technical discussions and code reviews
- Mentor junior developers
- Work with cloud technologies (AWS, Azure)
- Collaborate with cross-functional teams

Requirements:
- 5+ years of software development experience
- Strong proficiency in Python, Go, or Java
- Experience with microservices architecture
- Knowledge of CI/CD pipelines
- Excellent communication skills in English

We offer:
- Competitive salary (650,000 - 850,000 DKK annually)
- Flexible working hours
- Pension scheme
- Health insurance
- Professional development opportunities""",
                url="https://www.linkedin.com/jobs/novo-nordisk-senior-software-engineer",
                posted_date="2 days ago",
                job_type="Full-time",
                experience_level="Senior",
                salary="650,000 - 850,000 DKK/year",
                skills=["Python", "Go", "AWS", "Microservices", "CI/CD"]
            ),
            JobListing(
                title="Full Stack Developer",
                company="Maersk",
                location="Copenhagen, Denmark",
                description="""Join Maersk's technology team to build the future of global logistics.

What you'll do:
- Develop web applications using React and Node.js
- Build RESTful APIs and GraphQL services
- Work with containerized applications (Docker, Kubernetes)
- Participate in agile development processes
- Write clean, maintainable code

What we're looking for:
- 3+ years of full stack development experience
- Proficiency in JavaScript/TypeScript, React, Node.js
- Experience with SQL and NoSQL databases
- Familiarity with cloud platforms
- Good problem-solving skills

Benefits:
- Salary range: 550,000 - 700,000 DKK
- Remote work options (2-3 days/week)
- International work environment
- Learning and development budget
- Annual bonus program""",
                url="https://www.linkedin.com/jobs/maersk-full-stack-developer",
                posted_date="1 week ago",
                job_type="Full-time",
                experience_level="Mid-level",
                salary="550,000 - 700,000 DKK/year",
                skills=["React", "Node.js", "TypeScript", "Docker", "Kubernetes", "PostgreSQL"]
            ),
            JobListing(
                title="Machine Learning Engineer",
                company="Spotify",
                location="Copenhagen, Denmark",
                description="""Spotify is looking for a Machine Learning Engineer to work on recommendation systems.

Your role:
- Develop and improve ML models for music recommendations
- Work with large-scale data processing pipelines
- Collaborate with researchers and product teams
- Deploy models to production using MLOps practices
- Analyze model performance and iterate

Requirements:
- MSc or PhD in Computer Science, Statistics, or related field
- 3+ years of ML engineering experience
- Strong Python skills and familiarity with ML frameworks (PyTorch, TensorFlow)
- Experience with big data technologies (Spark, Beam)
- Knowledge of recommendation systems is a plus

What we offer:
- Competitive compensation: 700,000 - 900,000 DKK
- Stock options
- Flexible public holidays
- Parental leave (6 months)
- Free Spotify Premium for life""",
                url="https://www.linkedin.com/jobs/spotify-ml-engineer",
                posted_date="3 days ago",
                job_type="Full-time",
                experience_level="Senior",
                salary="700,000 - 900,000 DKK/year",
                skills=["Python", "PyTorch", "TensorFlow", "Spark", "MLOps", "Recommendation Systems"]
            ),
            JobListing(
                title="DevOps Engineer",
                company="Zendesk",
                location="Copenhagen, Denmark",
                description="""We're hiring a DevOps Engineer to strengthen our platform team.

Responsibilities:
- Manage and improve CI/CD pipelines
- Maintain infrastructure as code (Terraform, CloudFormation)
- Monitor and optimize system performance
- Implement security best practices
- Support development teams with tooling

Qualifications:
- 4+ years of DevOps/SRE experience
- Strong knowledge of AWS or GCP
- Experience with Kubernetes and container orchestration
- Proficiency in scripting (Bash, Python)
- Understanding of networking and security

Perks:
- Base salary: 600,000 - 750,000 DKK
- Remote-first culture
- Home office setup allowance
- Wellness programs
- Team events and offsites""",
                url="https://www.linkedin.com/jobs/zendesk-devops-engineer",
                posted_date="5 days ago",
                job_type="Full-time",
                experience_level="Mid-level",
                salary="600,000 - 750,000 DKK/year",
                skills=["AWS", "Kubernetes", "Terraform", "CI/CD", "Python", "Docker"]
            ),
            JobListing(
                title="Junior Software Developer",
                company="Trustpilot",
                location="Copenhagen, Denmark",
                description="""Start your career at Trustpilot as a Junior Software Developer!

What you'll learn and do:
- Write code in C# and JavaScript
- Build features for our review platform
- Work alongside experienced developers
- Participate in code reviews
- Learn agile methodologies

We're looking for:
- BSc in Computer Science or related field (or equivalent experience)
- Basic programming knowledge in any language
- Enthusiasm to learn and grow
- Good communication skills
- Ability to work in a team

We provide:
- Competitive entry-level salary: 420,000 - 500,000 DKK
- Mentorship program
- Learning budget (15,000 DKK/year)
- Modern office in central Copenhagen
- Friday bars and social events""",
                url="https://www.linkedin.com/jobs/trustpilot-junior-developer",
                posted_date="1 day ago",
                job_type="Full-time",
                experience_level="Entry-level",
                salary="420,000 - 500,000 DKK/year",
                skills=["C#", "JavaScript", ".NET", "SQL", "Git"]
            ),
            JobListing(
                title="Data Engineer",
                company="LEGO Group",
                location="Copenhagen, Denmark",
                description="""Build data pipelines that power creativity at LEGO!

Your mission:
- Design and implement data pipelines using modern tools
- Work with cloud data platforms (Azure, Databricks)
- Ensure data quality and governance
- Collaborate with data scientists and analysts
- Optimize data infrastructure for performance

Requirements:
- 3-5 years of data engineering experience
- Proficiency in Python and SQL
- Experience with data warehousing concepts
- Knowledge of ETL/ELT processes
- Familiarity with Azure or AWS data services

LEGO benefits:
- Salary: 580,000 - 720,000 DKK
- Generous LEGO set discount (50%)
- Flexible working arrangements
- Excellent pension scheme (12% employer contribution)
- Family-friendly workplace""",
                url="https://www.linkedin.com/jobs/lego-data-engineer",
                posted_date="4 days ago",
                job_type="Full-time",
                experience_level="Mid-level",
                salary="580,000 - 720,000 DKK/year",
                skills=["Python", "SQL", "Azure", "Databricks", "ETL", "Data Warehousing"]
            ),
            JobListing(
                title="Frontend Developer (React)",
                company="Pleo",
                location="Copenhagen, Denmark",
                description="""Join Pleo, one of Denmark's hottest fintech startups!

The role:
- Build beautiful, responsive user interfaces
- Work with React, TypeScript, and modern CSS
- Collaborate with designers on UX improvements
- Write unit and integration tests
- Contribute to our component library

What you need:
- 2-4 years of frontend development experience
- Expert knowledge of React and TypeScript
- Understanding of web accessibility (WCAG)
- Experience with state management (Redux, Zustand)
- Eye for design and attention to detail

Why Pleo:
- Salary: 500,000 - 650,000 DKK
- Pleo card with monthly spending allowance
- Stock options
- Unlimited paid vacation (yes, really!)
- Work from anywhere policy""",
                url="https://www.linkedin.com/jobs/pleo-frontend-developer",
                posted_date="6 days ago",
                job_type="Full-time",
                experience_level="Mid-level",
                salary="500,000 - 650,000 DKK/year",
                skills=["React", "TypeScript", "CSS", "Redux", "Testing", "Web Accessibility"]
            ),
            JobListing(
                title="IT Security Specialist",
                company="Danske Bank",
                location="Copenhagen, Denmark",
                description="""Protect one of Scandinavia's largest banks as an IT Security Specialist.

Key responsibilities:
- Monitor and respond to security incidents
- Conduct vulnerability assessments
- Implement security controls and policies
- Work with SIEM tools and threat intelligence
- Train employees on security awareness

Required qualifications:
- 4+ years in IT security
- Knowledge of security frameworks (ISO 27001, NIST)
- Experience with security tools (Splunk, CrowdStrike, etc.)
- Relevant certifications (CISSP, CEH, or similar)
- Understanding of financial sector regulations

What we offer:
- Competitive salary: 650,000 - 800,000 DKK
- Comprehensive health insurance
- Pension (up to 15% contribution)
- Professional development and certifications paid
- Hybrid work model""",
                url="https://www.linkedin.com/jobs/danske-bank-security-specialist",
                posted_date="1 week ago",
                job_type="Full-time",
                experience_level="Senior",
                salary="650,000 - 800,000 DKK/year",
                skills=["SIEM", "Incident Response", "Vulnerability Assessment", "ISO 27001", "CISSP"]
            ),
            JobListing(
                title="Backend Developer (Python)",
                company="Unity Technologies",
                location="Copenhagen, Denmark",
                description="""Work on game engine backend services at Unity!

What you'll work on:
- Develop microservices for Unity's cloud platform
- Build APIs used by millions of game developers
- Optimize performance for high-traffic systems
- Write comprehensive tests and documentation
- Participate in on-call rotation

We need:
- 3+ years of backend development experience
- Strong Python skills (FastAPI, Django preferred)
- Experience with distributed systems
- Knowledge of message queues (Kafka, RabbitMQ)
- Familiarity with game development is a bonus

Unity perks:
- Salary range: 600,000 - 780,000 DKK
- Free games and Unity Pro license
- Ergonomic workspace setup
- Regular game jams
- Relocation assistance available""",
                url="https://www.linkedin.com/jobs/unity-backend-developer",
                posted_date="2 days ago",
                job_type="Full-time",
                experience_level="Mid-level",
                salary="600,000 - 780,000 DKK/year",
                skills=["Python", "FastAPI", "Django", "Kafka", "Microservices", "PostgreSQL"]
            ),
            JobListing(
                title="Cloud Solutions Architect",
                company="Microsoft Denmark",
                location="Copenhagen, Denmark",
                description="""Help enterprises transform with Azure as a Cloud Solutions Architect.

Your impact:
- Design cloud architectures for enterprise customers
- Lead technical discussions and workshops
- Create proof-of-concept implementations
- Stay current with Azure services and features
- Build relationships with key stakeholders

Qualifications:
- 7+ years of IT experience, 3+ in cloud architecture
- Deep knowledge of Azure services
- Experience with hybrid cloud solutions
- Strong presentation and communication skills
- Azure certifications (Solutions Architect Expert preferred)

Microsoft benefits:
- Base salary: 800,000 - 1,000,000 DKK
- Annual bonus (15-25%)
- Stock awards
- Comprehensive health and wellness programs
- Generous parental leave
- Employee discounts on Microsoft products""",
                url="https://www.linkedin.com/jobs/microsoft-cloud-architect",
                posted_date="3 days ago",
                job_type="Full-time",
                experience_level="Senior",
                salary="800,000 - 1,000,000 DKK/year",
                skills=["Azure", "Cloud Architecture", "Enterprise Solutions", "Hybrid Cloud", "DevOps"]
            ),
            JobListing(
                title="QA Engineer",
                company="Templafy",
                location="Copenhagen, Denmark",
                description="""Ensure quality at Templafy, a leading document management platform.

Responsibilities:
- Design and execute test plans
- Write automated tests (Selenium, Cypress)
- Perform API testing
- Collaborate with developers on quality improvements
- Report and track bugs

Requirements:
- 2-3 years of QA experience
- Experience with test automation frameworks
- Knowledge of API testing tools (Postman, REST Assured)
- Understanding of agile testing practices
- Attention to detail

Benefits:
- Salary: 480,000 - 580,000 DKK
- Stock options
- Flexible hours
- Team lunches
- Modern office with rooftop terrace""",
                url="https://www.linkedin.com/jobs/templafy-qa-engineer",
                posted_date="5 days ago",
                job_type="Full-time",
                experience_level="Mid-level",
                salary="480,000 - 580,000 DKK/year",
                skills=["Selenium", "Cypress", "API Testing", "Postman", "Test Automation"]
            ),
            JobListing(
                title="Site Reliability Engineer",
                company="Lunar",
                location="Copenhagen, Denmark",
                description="""Join Denmark's leading digital bank as an SRE!

What you'll do:
- Ensure high availability of banking services
- Build and improve monitoring and alerting
- Automate operational tasks
- Participate in incident management
- Drive reliability improvements

What we look for:
- 4+ years in SRE or DevOps roles
- Experience with Kubernetes and container platforms
- Strong scripting abilities (Go, Python)
- Knowledge of observability tools (Prometheus, Grafana)
- Experience with financial services is a plus

Lunar offers:
- Salary: 650,000 - 800,000 DKK
- Equity in a growing fintech
- Premium health insurance
- Latest tech equipment
- Downtown Copenhagen office""",
                url="https://www.linkedin.com/jobs/lunar-sre",
                posted_date="1 day ago",
                job_type="Full-time",
                experience_level="Senior",
                salary="650,000 - 800,000 DKK/year",
                skills=["Kubernetes", "Go", "Python", "Prometheus", "Grafana", "SRE"]
            ),
        ]
        
        return sample_jobs


def format_jobs_for_ingestion(jobs: List[JobListing]) -> str:
    """Format all jobs into a single text document for RAG ingestion."""
    header = f"""
COPENHAGEN IT & SOFTWARE JOB LISTINGS
======================================
Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Total Jobs: {len(jobs)}

This document contains current job listings for IT and Software positions
in Copenhagen, Denmark. Use this information to help job seekers find
relevant opportunities.

AVAILABLE POSITIONS:
"""
    
    job_texts = [job.to_text() for job in jobs]
    
    summary = f"""

SUMMARY STATISTICS
==================
Total Positions: {len(jobs)}
Companies Hiring: {len(set(job.company for job in jobs))}

Experience Levels:
- Entry-level: {sum(1 for job in jobs if job.experience_level == "Entry-level")}
- Mid-level: {sum(1 for job in jobs if job.experience_level == "Mid-level")}
- Senior: {sum(1 for job in jobs if job.experience_level == "Senior")}

Top Skills in Demand:
"""
    
    # Count skills
    skill_counts = {}
    for job in jobs:
        if job.skills:
            for skill in job.skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for skill, count in top_skills:
        summary += f"- {skill}: {count} jobs\n"
    
    return header + "\n".join(job_texts) + summary


def get_copenhagen_it_jobs() -> str:
    """
    Main function to get Copenhagen IT jobs for the chatbot.
    Returns formatted text ready for RAG ingestion.
    """
    scraper = LinkedInJobScraper()
    
    # Try to scrape, fall back to sample data
    jobs = scraper.get_sample_jobs()  # Using sample data for reliability
    
    return format_jobs_for_ingestion(jobs)


if __name__ == "__main__":
    # Test the module
    print("Fetching Copenhagen IT jobs...")
    jobs_text = get_copenhagen_it_jobs()
    print(jobs_text[:2000])
    print(f"\n... and {len(jobs_text) - 2000} more characters")
