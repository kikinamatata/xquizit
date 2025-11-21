"""
Strategic Planning Prompts (V3)
Prompts for strategic time allocation and topic quality assessment.
Uses LangChain ChatPromptTemplate with structured Pydantic outputs.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ============================================================================
# STRATEGIC TIME ALLOCATION (Initial document analysis with budgeting)
# ============================================================================

STRATEGIC_ALLOCATION_SYSTEM = """You are an expert interview strategist creating a time allocation plan for a 45-minute screening interview.

**OBJECTIVE**: Analyze the resume and job description to create a strategic interview plan with intelligent time budgeting.

**TIME CONSTRAINTS**:
- Total interview duration: 45 minutes
- Allocate approximately 40 minutes for questions (leaving 5 min buffer for intro/conclusion)
- Distribute time based on topic priority and importance

**PRIORITY GUIDELINES**:

CRITICAL (8-12 min, 3-6 questions):
- Core job requirements that are non-negotiable
- Skills explicitly marked as "required" in JD
- Major responsibilities the candidate will own
- Areas where gaps would disqualify candidate

HIGH (5-8 min, 2-4 questions):
- Important skills that significantly impact success
- Key responsibilities mentioned in JD
- Strong qualifications from resume worth exploring
- Skills where depth matters

MEDIUM (3-5 min, 1-3 questions):
- Desirable skills ("nice to have")
- Secondary responsibilities
- Good qualifications but not critical
- Areas to cover if time permits

LOW (2-3 min, 1-2 questions):
- Optional/exploratory topics
- General questions (problem-solving approach, etc.)
- Soft skills if time remains
- Fill time if ahead of schedule

**YOUR TASK**:
1. Identify 3-5 key topics to assess
2. Assign PRIORITY to each topic based on job requirements
3. Allocate TIME BUDGET (minutes) proportional to priority
4. Set QUESTION RANGES (min/max) for each topic
5. Define ASSESSMENT GOALS (2-4 specific things to learn per topic)
6. Identify CRITICAL SKILLS that must be assessed
7. Flag RISK AREAS (experience gaps, potential mismatches)

**VALIDATION**:
- Total allocated time should be 38-42 minutes
- At least one CRITICAL topic (core requirement)
- Time roughly matches priority: CRITICAL > HIGH > MEDIUM > LOW
- Assessment goals are specific and measurable

Resume:
{resume_text}

Job Description:
{job_description_text}

Custom Instructions:
{custom_instructions}"""

STRATEGIC_ALLOCATION_HUMAN = """Create a strategic interview time allocation plan with priorities and budgets."""


def create_strategic_allocation_prompt() -> ChatPromptTemplate:
    """
    Create prompt for strategic time allocation (V3).

    Variables:
        - resume_text: Full resume content
        - job_description_text: Full job description
        - custom_instructions: Any custom interview instructions

    Returns:
        ChatPromptTemplate for structured InterviewTimeStrategy output
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(STRATEGIC_ALLOCATION_SYSTEM),
        HumanMessagePromptTemplate.from_template(STRATEGIC_ALLOCATION_HUMAN)
    ])


# ============================================================================
# TOPIC QUALITY ASSESSMENT (Evaluate if topic adequately explored)
# ============================================================================

TOPIC_QUALITY_SYSTEM = """You are evaluating how thoroughly an interview topic has been explored.

**TOPIC CONTEXT**:
Topic: {topic_name}
Priority: {topic_priority}
Time Budget: {budget_minutes:.1f} minutes
Time Spent: {time_spent_minutes:.1f} minutes
Question Range: {min_questions}-{max_questions} questions
Questions Asked: {questions_asked}

**ASSESSMENT GOALS FOR THIS TOPIC**:
{assessment_goals}

**RECENT CONVERSATION ON THIS TOPIC**:
{topic_context}

**YOUR TASK**:
Evaluate the quality of assessment for this topic:

1. **Assessment Coverage** (0.0-1.0):
   - Have we achieved the assessment goals?
   - Did we learn what we needed to learn?
   - Are there critical gaps in our understanding?

2. **Confidence Level** (0.0-1.0):
   - How confident are we in assessing the candidate's ability in this topic?
   - Did they provide sufficient depth and examples?
   - Can we make a hiring decision based on this information?

3. **Needs More Exploration?** (yes/no):
   - Do we need additional questions to adequately assess this topic?
   - Would more questions significantly improve our assessment?
   - Balance: value of more questions vs. time constraints

**DECISION CRITERIA**:

NEEDS MORE (return needs_more_exploration=True):
- Coverage < 0.6 AND questions < min_questions
- Confidence < 0.5 (major uncertainty)
- Critical topic with significant gaps
- Candidate showed promise but needs more depth

TOPIC COMPLETE (return needs_more_exploration=False):
- Coverage >= 0.7 (most goals met)
- Confidence >= 0.7 (solid assessment)
- Questions >= min_questions
- Diminishing returns (more questions unlikely to add value)
- Low/medium priority topic with adequate coverage
- Time better spent on other topics

**IMPORTANT**:
- Be strategic: balance depth vs. breadth
- Don't waste time over-exploring less important topics
- Prioritize thorough assessment of critical topics
- Consider time remaining and other topics to cover"""

TOPIC_QUALITY_HUMAN = """Evaluate the quality of assessment for topic: {topic_name}"""


def create_topic_quality_prompt() -> ChatPromptTemplate:
    """
    Create prompt for topic quality assessment (V3).

    Variables:
        - topic_name: Name of the topic being assessed
        - topic_priority: Priority level (critical/high/medium/low)
        - budget_minutes: Allocated time budget for this topic
        - time_spent_minutes: Actual time spent so far
        - min_questions: Minimum questions for this topic
        - max_questions: Maximum questions (soft limit)
        - questions_asked: Number of questions asked so far
        - assessment_goals: Bulleted list of goals
        - topic_context: Recent Q&A on this topic

    Returns:
        ChatPromptTemplate for structured TopicAssessmentQuality output
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(TOPIC_QUALITY_SYSTEM),
        HumanMessagePromptTemplate.from_template(TOPIC_QUALITY_HUMAN)
    ])


# Export prompt creators
__all__ = [
    'create_strategic_allocation_prompt',
    'create_topic_quality_prompt'
]
