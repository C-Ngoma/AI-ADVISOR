
# advisor_ai_robust.py
# Robust AdvisorAI with validation, diagnostics, and user-facing error reasons.

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import textwrap
import sys

DATA_PATHS = [Path('ai_job_dataset.csv'), Path('ai_job_dataset1.csv')]

REQUIRED_COLUMNS = {
    'job_title', 'salary_usd', 'required_skills', 'education_required',
    'years_experience', 'posting_date', 'remote_ratio', 'company_size', 'company_location'
}

def tokenize_skills(s):
    if pd.isna(s):
        return []
    toks = re.split(r'[,\|/;]+', str(s).lower())
    return [t.strip() for t in toks if t.strip()]

def normalize_degree(deg):
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

def degree_rank(deg_norm):
    order = {'phd':4, 'master':3, 'bachelor':2, 'associate':1, 'other':0}
    return order.get(deg_norm, 0)

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
            "No datasets found. Ensure ai_job_dataset.csv and ai_job_dataset1.csv are present in the working directory or ./data."
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
    for col in ['job_title', 'required_skills', 'education_required', 'industry', 'company_location', 'employee_residence', 'company_size']:
        if col in jobs.columns:
            jobs[col] = jobs[col].astype(str).str.strip()
    # Validate required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in jobs.columns]
    if missing_cols:
        warnings.append(f"Dataset missing columns: {', '.join(missing_cols)}. Some features will be disabled.")
    return jobs, warnings

def build_demand_and_pay(jobs):
    now = pd.Timestamp('today')
    one_year_ago = now - pd.DateOffset(years=1)
    if 'posting_date' in jobs.columns:
        recent = jobs[jobs['posting_date'] >= one_year_ago].copy()
        if recent.empty:
            recent = jobs.copy()
    else:
        recent = jobs.copy()
    # Remote and size weights with null-safe handling
    remote = pd.to_numeric(recent.get('remote_ratio', 0), errors='coerce').fillna(0).astype(float) / 100.0
    size_map = {'S':1, 'M':1.25, 'L':1.5}
    size_weight = recent.get('company_size', pd.Series(['S']*len(recent)))
    size_weight = size_weight.map(size_map).fillna(1.0)
    recent = recent.assign(_w = 1.0 + 0.5*remote + 0.25*size_weight)
    demand = recent.groupby('job_title')['_w'].sum().rename('demand_score')
    pay = jobs.groupby('job_title')['salary_usd'].median().rename('median_salary_usd') if 'salary_usd' in jobs.columns else pd.Series(dtype=float)
    counts = (jobs.groupby('job_title')['job_title'].count().rename('postings_count'))
    agg = pd.concat([demand, pay, counts], axis=1).reset_index().fillna(0)
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

def filter_with_diagnostics(jobs, profile):
    """Apply sequential filters and collect diagnostics explaining why results may be empty."""
    diag = []
    df = jobs.copy()

    # Degree filter
    user_deg = normalize_degree(profile.get('degree', 'other'))
    if 'education_required' in df.columns:
        df['_req_degree_norm'] = df['education_required'].apply(normalize_degree)
        before = len(df)
        df = df[df['_req_degree_norm'].apply(degree_rank) <= degree_rank(user_deg)]
        diag.append(f"Degree filter: kept {len(df)}/{before} rows (user degree: {user_deg}).")
        if df.empty:
            return df, diag + ["No roles match your degree level. Try adding a higher qualification or selecting roles with lower requirements."]

    # Years experience filter
    if 'years_experience' in df.columns:
        years = profile.get('years_experience', 0)
        before = len(df)
        df = df[pd.to_numeric(df['years_experience'], errors='coerce').fillna(0) <= float(years)]
        diag.append(f"Experience filter: kept {len(df)}/{before} rows (user years: {years}).")
        if df.empty:
            return df, diag + ["No roles fit your experience level. Consider internships or junior roles and resubmit."]

    # Skills
    user_skills = [s.lower().strip() for s in profile.get('skills', []) if s and s.strip()]
    if 'required_skills' in df.columns:
        df['_req_skills_list'] = df['required_skills'].apply(tokenize_skills)
        skill_set = set(user_skills)
        def jaccard(req):
            if not skill_set: return 0.0
            r = set([x for x in req if x])
            if not r: return 0.0
            inter = len(r & skill_set)
            union = len(r | skill_set)
            return inter/union
        df['_skill_match'] = df['_req_skills_list'].apply(jaccard)
        before = len(df)
        if skill_set:
            df = df[df['_skill_match'] > 0]
        diag.append(f"Skill overlap filter: kept {len(df)}/{before} rows (skills provided: {', '.join(user_skills) if user_skills else 'none'}).")
        if df.empty:
            return df, diag + ["No roles share skills with your profile. Add more relevant skills or broaden your list."]

    # International targeting
    if 'company_location' in df.columns and 'remote_ratio' in df.columns:
        before = len(df)
        df = df[(df['company_location'].str.lower() != 'south africa') | (pd.to_numeric(df['remote_ratio'], errors='coerce').fillna(0) >= 50)]
        diag.append(f"International/remote filter: kept {len(df)}/{before} rows.")
        if df.empty:
            return df, diag + ["All matching roles are local on-site. Add remote-friendly skills or allow international relocation."]

    return df, diag

def rank_roles(jobs_filtered, agg_scores, preference='balanced'):
    merged = jobs_filtered.merge(agg_scores, on='job_title', how='left')
    for col in ['demand_score', 'median_salary_usd', 'postings_count']:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    def norm(s):
        s = s.fillna(0).astype(float)
        if s.max() == s.min():
            return pd.Series(0.0, index=s.index)
        return (s - s.min())/(s.max()-s.min())
    merged['n_demand'] = norm(merged['demand_score'])
    merged['n_pay'] = norm(merged['median_salary_usd'])
    merged['n_skill'] = norm(merged.get('_skill_match', pd.Series(0.0, index=merged.index)))
    merged['n_industry'] = norm(merged.get('_industry_pref', pd.Series(0.0, index=merged.index)))
    if preference == 'easiest':
        w = dict(n_demand=0.5, n_skill=0.25, n_industry=0.1, n_pay=0.15)
    elif preference == 'highest_pay':
        w = dict(n_pay=0.5, n_skill=0.25, n_demand=0.2, n_industry=0.05)
    else:
        w = dict(n_pay=0.35, n_demand=0.35, n_skill=0.25, n_industry=0.05)
    merged['score'] = sum(w[k]*merged[k] for k in w)
    summary = (merged.groupby('job_title')
               .agg(
                   score=('score','mean'),
                   median_salary_usd=('median_salary_usd','first'),
                   demand_score=('demand_score','first'),
                   postings=('postings_count','first'),
                   sample_skills=('required_skills', lambda x: Counter([s for row in x for s in tokenize_skills(row)]).most_common(8)),
                   typical_degree=('education_required', lambda x: Counter([normalize_degree(v) for v in x]).most_common(1)[0][0] if len(x)>0 else 'other'),
               )
               .reset_index()
               .sort_values('score', ascending=False))
    return summary

def recommend_jobs(profile, preference='balanced', top_n=5):
    """Main entrypoint. Returns (result_df, messages). If no results, returns empty df and reason messages including 'JOB NOT AVAILABLE'."""
    ok, val_errors = validate_profile(profile)
    if not ok:
        reasons = ["JOB NOT AVAILABLE"] + val_errors
        return pd.DataFrame(columns=['job_title','reason','median_salary_usd','demand_score','postings','score']), reasons
    try:
        jobs, warnings = load_jobs()
    except FileNotFoundError as e:
        return pd.DataFrame(), ["JOB NOT AVAILABLE", str(e)]
    agg = build_demand_and_pay(jobs)
    filt, diag = filter_with_diagnostics(jobs, profile)
    if filt.empty:
        return pd.DataFrame(), ["JOB NOT AVAILABLE"] + diag
    ranked = rank_roles(filt, agg, preference=preference)
    if ranked.empty:
        return pd.DataFrame(), ["JOB NOT AVAILABLE", "No roles could be ranked due to insufficient data."]
    def fmt_reason(row):
        skills = ', '.join([s for s,_ in row['sample_skills'][:5]]) if isinstance(row['sample_skills'], list) else ''
        return textwrap.shorten(f"Matches your skills; demand score={row['demand_score']:.2f}; typical degree={row['typical_degree']}; common skills: {skills}", width=180)
    ranked['reason'] = ranked.apply(fmt_reason, axis=1)
    cols = ['job_title','reason','median_salary_usd','demand_score','postings','score']
    out = ranked.head(top_n)[cols]
    msgs = []
    if warnings:
        msgs += [f"Warning: {w}" for w in warnings]
    msgs += diag
    return out, msgs

def cli():
    print("AdvisorAI — International AI Job Recommender (Robust Mode)")
    try:
        degree = input('Enter your highest qualification (e.g., Bachelor in IT): ').strip()
        years = input('Enter your years of experience (0 if none): ').strip()
        skills = input('Enter your core skills, comma-separated (e.g., Python, SQL, TensorFlow): ').strip()
        prefs = input('Optional: preferred industries, comma-separated (press Enter to skip): ').strip()
        profile = {
            'degree': degree,
            'years_experience': float(years) if years else None,
            'skills': [s.strip() for s in skills.split(',')] if skills else [],
            'preferred_industries': [p.strip() for p in prefs.split(',')] if prefs else []
        }
    except Exception as e:
        print("JOB NOT AVAILABLE")
        print(f"Input error: {e}")
        sys.exit(1)
    for pref in ['balanced', 'easiest', 'highest_pay']:
        results, messages = recommend_jobs(profile, preference=pref, top_n=5)
        print(f"\nTop recommendations ({pref}):")
        if results.empty:
            print("JOB NOT AVAILABLE")
            for m in messages:
                print(f"- {m}")
        else:
            print(results.to_string(index=False))
            if messages:
                print("\nNotes:")
                for m in messages:
                    print(f"- {m}")

if __name__ == '__main__':
    cli()
