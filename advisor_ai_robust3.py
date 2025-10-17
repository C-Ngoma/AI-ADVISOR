# advisor_ai_ml_enhanced.py
# Enhanced AI Advisor with machine learning components and AI-powered career advice

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import textwrap
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_PATHS = [Path('ai_job_dataset.csv'), Path('ai_job_dataset1.csv')]

REQUIRED_COLUMNS = {
    'job_title', 'salary_usd', 'required_skills', 'education_required',
    'years_experience', 'posting_date', 'remote_ratio', 'company_size', 'company_location'
}

class CareerAdvisor:
    """AI-powered career advice generator"""
    
    def __init__(self):
        self.skill_gap_advice = {
            'technical': [
                "Consider learning Python and data analysis libraries like Pandas and NumPy",
                "Build projects using machine learning frameworks like TensorFlow or PyTorch",
                "Practice data visualization with tools like Matplotlib and Tableau",
                "Learn cloud platforms like AWS, Azure, or GCP for AI deployment"
            ],
            'soft_skills': [
                "Develop your communication skills for explaining technical concepts",
                "Practice problem-solving with real-world case studies",
                "Improve collaboration skills through team projects",
                "Build your professional network through LinkedIn and tech communities"
            ],
            'certifications': [
                "Consider Google's Professional Machine Learning Engineer certification",
                "AWS Certified Machine Learning Specialty could boost your profile",
                "Microsoft Azure AI Engineer Associate is highly valued",
                "Deep Learning AI specialization on Coursera"
            ]
        }
    
    def generate_career_advice(self, profile, recommendations, all_jobs):
        """Generate personalized career advice based on profile and recommendations"""
        advice_sections = []
        
        # Analyze skill gaps
        skill_gaps = self._analyze_skill_gaps(profile, recommendations, all_jobs)
        if skill_gaps:
            advice_sections.append(self._format_skill_gap_advice(skill_gaps))
        
        # Career path analysis
        career_path = self._analyze_career_path(profile, recommendations)
        if career_path:
            advice_sections.append(career_path)
        
        # Market insights
        market_insights = self._generate_market_insights(recommendations, all_jobs)
        if market_insights:
            advice_sections.append(market_insights)
        
        # Action plan
        action_plan = self._create_action_plan(profile, recommendations)
        advice_sections.append(action_plan)
        
        return "\n\n".join(advice_sections)
    
    def _analyze_skill_gaps(self, profile, recommendations, all_jobs):
        """Identify missing skills for top recommended jobs"""
        if recommendations.empty:
            return None
            
        user_skills = set(skill.lower().strip() for skill in profile.get('skills', []))
        top_jobs_skills = set()
        
        # Get skills from top recommended jobs
        for _, job in recommendations.head(3).iterrows():
            job_title = job['job_title']
            job_skills = all_jobs[all_jobs['job_title'] == job_title]['required_skills']
            for skills_str in job_skills:
                if pd.notna(skills_str):
                    skills = tokenize_skills(skills_str)
                    top_jobs_skills.update(skills)
        
        missing_skills = top_jobs_skills - user_skills
        return list(missing_skills)[:10]  # Return top 10 missing skills
    
    def _format_skill_gap_advice(self, missing_skills):
        """Format skill gap analysis into actionable advice"""
        advice = "🔍 **SKILL GAP ANALYSIS**\n"
        advice += "To increase your competitiveness for recommended roles, consider developing these skills:\n\n"
        
        for i, skill in enumerate(missing_skills[:5], 1):
            advice += f"{i}. **{skill.title()}** - "
            if any(tech in skill.lower() for tech in ['python', 'tensorflow', 'pytorch', 'sql']):
                advice += "Technical skill - practice through online courses and projects\n"
            elif any(soft in skill.lower() for soft in ['communication', 'leadership', 'team']):
                advice += "Soft skill - develop through practice and real-world experience\n"
            else:
                advice += "Industry skill - learn through specialized training\n"
        
        advice += "\n💡 **Quick Wins**: Focus on 2-3 high-demand skills first"
        return advice
    
    def _analyze_career_path(self, profile, recommendations):
        """Analyze potential career progression paths"""
        if recommendations.empty:
            return None
            
        user_exp = profile.get('years_experience', 0)
        user_degree = profile.get('degree', '').lower()
        
        advice = "🚀 **CAREER PATH ANALYSIS**\n"
        
        # Experience-based guidance
        if user_exp < 2:
            advice += "As an early-career professional:\n"
            advice += "• Focus on building foundational skills and practical experience\n"
            advice += "• Consider internships or junior roles to gain industry exposure\n"
            advice += "• Build a portfolio of projects to demonstrate your capabilities\n"
        elif user_exp < 5:
            advice += "As a mid-level professional:\n"
            advice += "• Specialize in high-demand AI domains\n"
            advice += "• Consider leadership or mentorship opportunities\n"
            advice += "• Build expertise in specific industry applications\n"
        else:
            advice += "As an experienced professional:\n"
            advice += "• Leverage your experience for senior or specialized roles\n"
            advice += "• Consider architect or lead positions\n"
            advice += "• Explore consulting or strategic roles\n"
        
        # Degree-based guidance
        if 'phd' in user_degree:
            advice += "\n• Your PhD positions you well for research and advanced technical roles\n"
        elif 'master' in user_degree:
            advice += "\n• Your Master's degree is excellent for specialized technical positions\n"
        elif 'bachelor' in user_degree:
            advice += "\n• Your Bachelor's degree provides solid foundation - consider specialization\n"
        
        return advice
    
    def _generate_market_insights(self, recommendations, all_jobs):
        """Provide market insights based on job data"""
        if recommendations.empty:
            return None
            
        advice = "📊 **MARKET INSIGHTS**\n"
        
        # Salary insights
        if 'median_salary_usd' in recommendations.columns:
            avg_salary = recommendations['median_salary_usd'].mean()
            advice += f"• Average salary for your matches: ${avg_salary:,.0f} USD\n"
        
        # Demand insights
        if 'demand_score' in recommendations.columns:
            avg_demand = recommendations['demand_score'].mean()
            if avg_demand > 2.0:
                advice += "• High market demand for your skill profile - strong negotiating position\n"
            elif avg_demand > 1.0:
                advice += "• Moderate market demand - focus on differentiating your skills\n"
            else:
                advice += "• Competitive market - emphasize unique value proposition\n"
        
        # Success probability insights
        if 'success_probability' in recommendations.columns:
            avg_success = recommendations['success_probability'].mean()
            advice += f"• Average application success probability: {avg_success:.1%}\n"
            
            if avg_success > 0.7:
                advice += "• Excellent match with current opportunities!\n"
            elif avg_success > 0.5:
                advice += "• Good alignment - consider applying to multiple positions\n"
            else:
                advice += "• Consider skill development to improve match rates\n"
        
        return advice
    
    def _create_action_plan(self, profile, recommendations):
        """Create personalized action plan"""
        advice = "🎯 **RECOMMENDED ACTION PLAN**\n"
        
        if not recommendations.empty:
            top_job = recommendations.iloc[0]['job_title']
            advice += f"1. **Primary Target**: {top_job}\n"
            advice += "2. **Immediate Steps**:\n"
            advice += "   • Update your resume with relevant keywords from job descriptions\n"
            advice += "   • Tailor your LinkedIn profile to highlight matching skills\n"
            advice += "   • Prepare for technical interviews in your target domain\n"
            advice += "   • Network with professionals in your desired companies\n\n"
            
            advice += "3. **30-Day Plan**:\n"
            advice += "   • Apply to 3-5 positions weekly\n"
            advice += "   • Complete one skill-enhancement course\n"
            advice += "   • Build or update one portfolio project\n"
            advice += "   • Attend 2 virtual networking events\n"
        else:
            advice += "1. **Skill Development Phase**:\n"
            advice += "   • Focus on building core AI/ML skills\n"
            advice += "   • Work on practical projects for your portfolio\n"
            advice += "   • Obtain relevant certifications\n"
            advice += "   • Connect with mentors in the field\n"
        
        advice += "\n4. **Long-term Strategy**:\n"
        advice += "   • Continuously update skills with emerging technologies\n"
        advice += "   • Build professional brand through content and contributions\n"
        advice += "   • Seek international opportunities for global experience\n"
        advice += "   • Consider advanced education for career advancement\n"
        
        return advice

class AIModel:
    def __init__(self):
        self.skill_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.job_title_encoder = LabelEncoder()
        self.degree_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.is_trained = False
        self.job_features = None
        self.career_advisor = CareerAdvisor()
        
    def prepare_features(self, jobs_df):
        """Prepare features for ML training"""
        df = jobs_df.copy()
        
        # Skill embeddings using TF-IDF
        df['skills_clean'] = df['required_skills'].fillna('').apply(self.clean_text)
        if len(df) > 0:
            skill_vectors = self.skill_vectorizer.fit_transform(df['skills_clean'])
            skill_columns = [f'skill_{i}' for i in range(skill_vectors.shape[1])]
            skill_df = pd.DataFrame(skill_vectors.toarray(), columns=skill_columns)
        else:
            skill_df = pd.DataFrame()
            
        # Encode categorical variables
        df['job_title_encoded'] = self.job_title_encoder.fit_transform(df['job_title'].fillna('unknown'))
        df['degree_encoded'] = self.degree_encoder.fit_transform(
            df['education_required'].fillna('').apply(self.normalize_degree)
        )
        df['location_encoded'] = self.location_encoder.fit_transform(df['company_location'].fillna('unknown'))
        
        # Numerical features
        numerical_features = ['salary_usd', 'years_experience', 'remote_ratio']
        for feat in numerical_features:
            if feat in df.columns:
                df[f'{feat}_norm'] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
            else:
                df[f'{feat}_norm'] = 0
                
        # Combine all features
        feature_columns = ['job_title_encoded', 'degree_encoded', 'location_encoded', 
                          'salary_usd_norm', 'years_experience_norm', 'remote_ratio_norm']
        
        if not skill_df.empty:
            self.job_features = pd.concat([df[feature_columns], skill_df], axis=1)
        else:
            self.job_features = df[feature_columns]
            
        # Scale features and apply clustering
        if len(self.job_features) > 0:
            self.job_features_scaled = self.scaler.fit_transform(self.job_features)
            df['job_cluster'] = self.kmeans.fit_predict(self.job_features_scaled)
            self.is_trained = True
            
        return df
    
    def predict_job_success_probability(self, user_profile, job_features):
        """Predict probability of job success using learned patterns"""
        if not self.is_trained:
            return 0.5  # Default probability if model not trained
            
        # Simple similarity-based prediction (can be enhanced with proper classifier)
        user_vector = self.create_user_vector(user_profile)
        if user_vector is None:
            return 0.5
            
        # Calculate similarity between user and job
        similarities = cosine_similarity([user_vector], job_features)
        success_prob = np.mean(similarities) * 0.8 + 0.2  # Normalize to 0.2-1.0 range
        
        return min(max(success_prob, 0.1), 0.95)
    
    def create_user_vector(self, user_profile):
        """Create feature vector for user profile"""
        try:
            # Encode user features to match job feature space
            user_skills = self.clean_text(' '.join(user_profile.get('skills', [])))
            skill_vector = self.skill_vectorizer.transform([user_skills]).toarray()[0]
            
            # Encode other features
            user_degree = self.normalize_degree(user_profile.get('degree', ''))
            try:
                degree_encoded = self.degree_encoder.transform([user_degree])[0]
            except:
                degree_encoded = 0
                
            # Create full user vector
            user_vector = np.concatenate([
                [0],  # job_title_encoded (placeholder)
                [degree_encoded],
                [0],  # location_encoded (placeholder)
                [user_profile.get('years_experience', 0)],
                [0],  # salary placeholder
                [50],  # remote_ratio preference
                skill_vector
            ])
            
            return user_vector
        except Exception as e:
            print(f"Error creating user vector: {e}")
            return None
    
    def clean_text(self, text):
        """Clean text for NLP processing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_degree(self, deg):
        """Normalize degree levels"""
        deg = str(deg).lower()
        if 'phd' in deg or 'doctor' in deg:
            return 'phd'
        if 'master' in deg or 'msc' in deg or 'm.sc' in deg:
            return 'master'
        if 'bachelor' in deg or 'bsc' in deg or 'b.sc' in deg:
            return 'bachelor'
        if 'diploma' in deg or 'associate' in deg or 'cert' in deg:
            return 'associate'
        return 'other'

def tokenize_skills(s):
    if pd.isna(s):
        return []
    toks = re.split(r'[,\|/;]+', str(s).lower())
    return [t.strip() for t in toks if t.strip()]

def load_jobs(paths=DATA_PATHS):
    """Load and validate datasets. Returns (jobs_df, warnings)."""
    frames = []
    warnings = []
    if not paths:
        raise FileNotFoundError("No dataset paths provided.")
    for p in paths:
        if not Path(p).exists():
            warnings.append(f"Dataset missing: {p}")
            continue
        try:
            df = pd.read_csv(p, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding='latin-1', on_bad_lines='skip')
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No datasets found. Ensure ai_job_dataset.csv and ai_job_dataset1.csv are present."
        )
    jobs = pd.concat(frames, ignore_index=True)
    # Standardize columns
    jobs.columns = [c.strip().lower() for c in jobs.columns]
    # Coerce types
    if 'salary_usd' in jobs.columns:
        jobs['salary_usd'] = pd.to_numeric(jobs['salary_usd'], errors='coerce')
    for col in ['posting_date', 'application_deadline']:
        if col in jobs.columns:
            jobs[col] = pd.to_datetime(jobs[col], errors='coerce')
    # Strip string columns
    for col in ['job_title', 'required_skills', 'education_required', 'industry', 
                'company_location', 'employee_residence', 'company_size']:
        if col in jobs.columns:
            jobs[col] = jobs[col].astype(str).str.strip()
    # Validate required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in jobs.columns]
    if missing_cols:
        warnings.append(f"Dataset missing columns: {', '.join(missing_cols)}. Some features will be disabled.")
    return jobs, warnings

def build_demand_analysis(jobs):
    """Enhanced demand analysis with trend detection"""
    now = pd.Timestamp('today')
    one_year_ago = now - pd.DateOffset(years=1)
    
    if 'posting_date' in jobs.columns:
        recent = jobs[jobs['posting_date'] >= one_year_ago].copy()
        if recent.empty:
            recent = jobs.copy()
    else:
        recent = jobs.copy()
    
    # Time series analysis for trend detection
    if 'posting_date' in recent.columns and len(recent) > 0:
        recent['posting_month'] = recent['posting_date'].dt.to_period('M')
        monthly_trends = recent.groupby(['job_title', 'posting_month']).size().unstack(fill_value=0)
        # Simple trend: compare last 3 months vs previous 3 months
        if len(monthly_trends.columns) >= 6:
            recent_months = monthly_trends.iloc[:, -3:].sum(axis=1)
            previous_months = monthly_trends.iloc[:, -6:-3].sum(axis=1)
            growth_rate = (recent_months - previous_months) / (previous_months + 1)
        else:
            growth_rate = pd.Series(0, index=monthly_trends.index)
    else:
        growth_rate = pd.Series(dtype=float)
    
    # Enhanced demand scoring with ML clusters
    remote = pd.to_numeric(recent.get('remote_ratio', 0), errors='coerce').fillna(0).astype(float) / 100.0
    size_map = {'S':1, 'M':1.25, 'L':1.5}
    size_weight = recent.get('company_size', pd.Series(['S']*len(recent)))
    size_weight = size_weight.map(size_map).fillna(1.0)
    
    recent = recent.assign(_w = 1.0 + 0.5*remote + 0.25*size_weight)
    demand = recent.groupby('job_title')['_w'].sum().rename('demand_score')
    
    # Add growth rate to demand score
    demand_df = demand.reset_index()
    if not growth_rate.empty:
        growth_df = growth_rate.rename('growth_rate').reset_index()
        demand_df = demand_df.merge(growth_df, on='job_title', how='left')
        demand_df['growth_rate'] = demand_df['growth_rate'].fillna(0)
        demand_df['demand_score'] = demand_df['demand_score'] * (1 + demand_df['growth_rate'] * 0.5)
    
    pay = jobs.groupby('job_title')['salary_usd'].median().rename('median_salary_usd') if 'salary_usd' in jobs.columns else pd.Series(dtype=float)
    counts = jobs.groupby('job_title')['job_title'].count().rename('postings_count')
    
    agg = pd.concat([demand_df.set_index('job_title')['demand_score'], pay, counts], axis=1).reset_index().fillna(0)
    return agg

def validate_profile(profile):
    """Return (is_valid, errors)."""
    errors = []
    degree = str(profile.get('degree', '')).strip()
    years = profile.get('years_experience', None)
    skills = profile.get('skills', [])
    if not degree:
        errors.append("Missing highest qualification.")
    try:
        years_val = float(years)
        if years_val < 0 or years_val > 60:
            errors.append("Years of experience must be between 0 and 60.")
    except (TypeError, ValueError):
        errors.append("Years of experience must be a number.")
    if not skills or not any(s.strip() for s in skills):
        errors.append("At least one skill is required.")
    return (len(errors) == 0), errors

def ml_enhanced_ranking(jobs_filtered, agg_scores, ai_model, user_profile, preference='balanced'):
    """Enhanced ranking using ML predictions"""
    merged = jobs_filtered.merge(agg_scores, on='job_title', how='left')
    
    for col in ['demand_score', 'median_salary_usd', 'postings_count']:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    
    # Calculate success probabilities using ML model
    success_probs = []
    for idx, job in merged.iterrows():
        prob = ai_model.predict_job_success_probability(user_profile, 
                                                      ai_model.job_features_scaled[idx:idx+1] 
                                                      if idx < len(ai_model.job_features_scaled) else None)
        success_probs.append(prob)
    
    merged['success_probability'] = success_probs
    
    def norm(s):
        s = s.fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series(0.0, index=s.index)
        return (s - s.min())/(s.max()-s.min())
    
    # Normalize features
    merged['n_demand'] = norm(merged['demand_score'])
    merged['n_pay'] = norm(merged['median_salary_usd'])
    merged['n_success'] = norm(merged['success_probability'])
    merged['n_skill'] = norm(merged.get('_skill_match', pd.Series(0.0, index=merged.index)))
    
    # Dynamic weights based on ML insights
    if preference == 'easiest':
        w = dict(n_success=0.4, n_demand=0.3, n_skill=0.2, n_pay=0.1)
    elif preference == 'highest_pay':
        w = dict(n_pay=0.4, n_success=0.3, n_skill=0.2, n_demand=0.1)
    else:  # balanced
        w = dict(n_success=0.35, n_pay=0.25, n_demand=0.25, n_skill=0.15)
    
    merged['score'] = sum(w[k]*merged[k] for k in w)
    
    summary = (merged.groupby('job_title')
               .agg(
                   score=('score','mean'),
                   success_probability=('success_probability','mean'),
                   median_salary_usd=('median_salary_usd','first'),
                   demand_score=('demand_score','first'),
                   postings=('postings_count','first'),
                   sample_skills=('required_skills', lambda x: Counter([s for row in x for s in tokenize_skills(row)]).most_common(8)),
                   typical_degree=('education_required', lambda x: Counter([ai_model.normalize_degree(v) for v in x]).most_common(1)[0][0] if len(x)>0 else 'other'),
               )
               .reset_index()
               .sort_values('score', ascending=False))
    
    return summary

def recommend_jobs(profile, preference='balanced', top_n=5):
    """Main entrypoint with ML enhancements."""
    ok, val_errors = validate_profile(profile)
    if not ok:
        reasons = ["JOB NOT AVAILABLE"] + val_errors
        return pd.DataFrame(columns=['job_title','reason','median_salary_usd','demand_score','postings','score','success_probability']), reasons, "Unable to generate career advice due to profile validation errors."
    
    try:
        jobs, warnings = load_jobs()
    except FileNotFoundError as e:
        return pd.DataFrame(), ["JOB NOT AVAILABLE", str(e)], "Career advice unavailable due to missing data."
    
    # Initialize and train AI model
    ai_model = AIModel()
    jobs_with_features = ai_model.prepare_features(jobs)
    
    agg = build_demand_analysis(jobs_with_features)
    
    # Use original filtering logic (can be enhanced later)
    try:
        from advisor_ai_robust import filter_with_diagnostics
        filt, diag = filter_with_diagnostics(jobs_with_features, profile)
    except ImportError:
        # Fallback filtering if import fails
        filt = jobs_with_features.copy()
        diag = ["Using basic filtering"]
    
    if filt.empty:
        return pd.DataFrame(), ["JOB NOT AVAILABLE"] + diag, ai_model.career_advisor.generate_career_advice(profile, pd.DataFrame(), jobs_with_features)
    
    # Use ML-enhanced ranking
    ranked = ml_enhanced_ranking(filt, agg, ai_model, profile, preference=preference)
    
    if ranked.empty:
        return pd.DataFrame(), ["JOB NOT AVAILABLE", "No roles could be ranked due to insufficient data."], ai_model.career_advisor.generate_career_advice(profile, pd.DataFrame(), jobs_with_features)
    
    def fmt_reason(row):
        skills = ', '.join([s for s,_ in row['sample_skills'][:3]]) if isinstance(row['sample_skills'], list) else ''
        prob_percent = row['success_probability'] * 100
        return textwrap.shorten(
            f"Success probability: {prob_percent:.1f}%; Matches your skills; Demand score: {row['demand_score']:.2f}", 
            width=180
        )
    
    ranked['reason'] = ranked.apply(fmt_reason, axis=1)
    cols = ['job_title','reason','median_salary_usd','demand_score','postings','score','success_probability']
    out = ranked.head(top_n)[cols]
    
    msgs = []
    if warnings:
        msgs += [f"Warning: {w}" for w in warnings]
    msgs += diag
    msgs.append("ML-enhanced recommendations active with success probability scoring.")
    
    # Generate career advice
    career_advice = ai_model.career_advisor.generate_career_advice(profile, out, jobs_with_features)
    
    return out, msgs, career_advice

def cli():
    print("AdvisorAI — ML-Enhanced International AI Job Recommender")
    print("Now with AI-powered success probability predictions and career guidance!")
    print("=" * 80)
    
    try:
        degree = input('Enter your highest qualification (e.g., Bachelor in IT): ').strip()
        years = input('Enter your years of experience (0 if none): ').strip()
        skills = input('Enter your core skills, comma-separated (e.g., Python, SQL, TensorFlow): ').strip()
        prefs = input('Optional: preferred industries, comma-separated (press Enter to skip): ').strip()
        
        profile = {
            'degree': degree,
            'years_experience': float(years) if years else 0,
            'skills': [s.strip() for s in skills.split(',')] if skills else [],
            'preferred_industries': [p.strip() for p in prefs.split(',')] if prefs else []
        }
        
    except Exception as e:
        print("JOB NOT AVAILABLE")
        print(f"Input error: {e}")
        sys.exit(1)
    
    all_results = {}
    
    for pref in ['balanced', 'easiest', 'highest_pay']:
        results, messages, career_advice = recommend_jobs(profile, preference=pref, top_n=5)
        all_results[pref] = (results, messages, career_advice)
        
        print(f"\n{'='*60}")
        print(f"TOP RECOMMENDATIONS ({pref.upper()} APPROACH)")
        print(f"{'='*60}")
        
        if results.empty:
            print("JOB NOT AVAILABLE")
            for m in messages:
                print(f"- {m}")
        else:
            print(results.to_string(index=False))
            if messages:
                print("\n📝 **SYSTEM NOTES**:")
                for m in messages:
                    print(f"- {m}")
    
    # Display comprehensive career advice at the end
    print(f"\n{'='*80}")
    print("🎓 **AI CAREER ADVISOR - COMPREHENSIVE GUIDANCE**")
    print(f"{'='*80}")
    
    # Use the career advice from balanced approach (most comprehensive)
    balanced_advice = all_results['balanced'][2]
    print(balanced_advice)
    
    print(f"\n{'='*80}")
    print("💡 **NEXT STEPS**:")
    print("1. Review your top matches above")
    print("2. Follow the action plan in the career guidance")
    print("3. Focus on skill development areas identified")
    print("4. Start applying to positions that match your profile")
    print("5. Track your progress and adjust your strategy as needed")
    print(f"{'='*80}")

if __name__ == '__main__':
    cli()