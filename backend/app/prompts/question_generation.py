"""
Question Generation Prompts
Generates interview questions optimized for streaming.
Uses LangChain ChatPromptTemplate with minimal token usage.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# ============================================================================
# INTRODUCTORY QUESTION (First question of interview)
# ============================================================================

INTRO_QUESTION_SYSTEM = """You are starting a professional screening interview.

Generate a warm, welcoming introductory question that:
1. Invites the candidate to introduce themselves
2. Sets a conversational, friendly tone
3. Allows them to share relevant background

Examples:
- "Thanks for joining! Tell me about yourself and what drew you to this role."
- "Welcome! I'd love to hear about your background and interest in this position."

Return only the question text."""

INTRO_QUESTION_HUMAN = """Please generate a welcoming introductory interview question."""


def create_intro_question_prompt() -> ChatPromptTemplate:
    """
    Create prompt for the first introductory question.

    Returns:
        ChatPromptTemplate for streaming introduction question
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTRO_QUESTION_SYSTEM),
        HumanMessagePromptTemplate.from_template(INTRO_QUESTION_HUMAN)
    ])


# ============================================================================
# TOPIC QUESTION (Main interview questions)
# ============================================================================

TOPIC_QUESTION_SYSTEM = """You are conducting a professional screening interview. Your goal is to ask insightful, contextual questions that reference the candidate's specific background and the job requirements.

==== INTERVIEW STRATEGY ====
{strategy_summary}

==== CURRENT FOCUS ====
Topic: {current_topic}
Assessment Goals: {assessment_goals}
Topics Already Covered: {covered_topics}

==== CANDIDATE'S RESUME (Relevant Excerpts) ====
{resume_context}

==== JOB REQUIREMENTS (Relevant Excerpts) ====
{jd_context}

==== RECENT CONVERSATION ====
{conversation_summary}

==== INSTRUCTIONS ====
Think step-by-step to generate an excellent interview question:

1. ANALYZE the candidate's resume excerpts above - identify specific projects, achievements, or skills related to "{current_topic}"
2. REVIEW the job requirements - note what specific capabilities are needed for "{current_topic}"
3. CONSIDER the conversation so far - what have we learned? What depth is still needed?
4. CHECK the assessment goals - what specifically do we need to evaluate?
5. GENERATE a question that:
   - CITES a specific achievement, project, or skill from their resume
   - CONNECTS to a specific job requirement when relevant
   - BUILDS on previous answers if applicable
   - PROBES for depth, examples, challenges, or impact
   - Feels natural and conversational, not robotic

==== FEW-SHOT EXAMPLES ====

Example 1 - Good Question (cites resume, connects to JD):
Resume mentions: "Led migration from monolith to microservices using Kubernetes, reducing deployment time by 60%"
JD requires: "Experience with container orchestration and service mesh"
Question: "I see you led a migration to microservices with Kubernetes. Can you walk me through how you handled service discovery and inter-service communication? The role involves working with service mesh technologies, so I'm curious about your approach."

Example 2 - Good Question (builds on previous answer):
Previous answer: "I used Python for data pipeline automation"
Resume mentions: "Built ETL pipelines processing 10M records daily"
Question: "You mentioned using Python for pipeline automation. In your experience processing 10 million records daily, what strategies did you use to optimize performance and handle failures?"

Example 3 - Good Question (probes for depth):
Resume mentions: "Implemented CI/CD pipeline reducing release cycle from 2 weeks to 2 days"
Question: "That's impressive - reducing your release cycle from 2 weeks to 2 days. What were the biggest challenges you faced during that implementation, and how did you get buy-in from the team?"

==== WHAT TO AVOID ====
- Generic questions that could apply to anyone: ❌ "Tell me about your experience with Python"
- Questions without context: ❌ "What's your approach to system design?"
- Questions that ignore their resume: ❌ "Have you worked with databases?"
- Multiple questions in one: ❌ "Tell me about your Python skills, your projects, and your challenges"

Now generate ONE excellent question on "{current_topic}" following the thinking process above. Return ONLY the final question text, no explanations or reasoning."""

TOPIC_QUESTION_HUMAN = """Generate the next interview question on: {current_topic}"""


def create_topic_question_prompt() -> ChatPromptTemplate:
    """
    Create prompt for topic-based questions with enhanced context.

    IMPROVED: Now includes resume/JD excerpts, assessment goals, and few-shot examples.
    Uses chain-of-thought prompting for higher quality questions.

    Variables:
        - strategy_summary: Interview strategy summary (up to 800 chars)
        - current_topic: Current topic to focus on
        - assessment_goals: Specific goals to evaluate for this topic
        - covered_topics: Topics already covered (comma-separated)
        - resume_context: Relevant resume excerpts for this topic (up to 1000 chars)
        - jd_context: Relevant job requirements for this topic (up to 1000 chars)
        - conversation_summary: Last 8 Q&A exchanges (up to 500 chars each)

    Returns:
        ChatPromptTemplate for streaming topic questions with citations
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(TOPIC_QUESTION_SYSTEM),
        HumanMessagePromptTemplate.from_template(TOPIC_QUESTION_HUMAN)
    ])


# ============================================================================
# FOLLOW-UP QUESTION (Dig deeper into recent answer)
# ============================================================================

FOLLOWUP_QUESTION_SYSTEM = """You are conducting a professional screening interview. You're asking a follow-up question to dig deeper into the candidate's previous answer.

==== CONTEXT ====
Topic: {current_topic}
Follow-up Question #{followup_number} (of max 2)
Assessment Goals: {assessment_goals}

==== CANDIDATE'S RESUME (Relevant Excerpts) ====
{resume_context}

==== CONVERSATION ON THIS TOPIC ====
{topic_context}

==== INSTRUCTIONS ====
Think step-by-step to generate an excellent follow-up question:

1. REVIEW their most recent answer carefully - what SPECIFIC things did they say?
   - What metrics, numbers, or measurements did they mention?
   - What tools, technologies, or approaches did they describe?
   - What examples or projects did they reference?

2. IDENTIFY what's missing or needs clarification:
   - Did they explain the "how" or just the "what"?
   - Did they mention challenges or just successes?
   - Did they quantify impact or just describe features?

3. START your question by REFERENCING what they said:
   - Quote specific numbers: "You mentioned reducing time from 6-7 seconds to 2-3 seconds..."
   - Reference specific tools: "You described using FastAPI with async/await..."
   - Point to specific projects: "In your work on the KYC APIs..."
   - This shows you're listening and builds naturally on their answer

4. OPTIONAL: Add brief acknowledgement ONLY if answer quality warrants it:
   - "That's impressive..." / "That's a solid approach..." (use sparingly)
   - Skip acknowledgement if you can transition naturally without it

5. ASK about what needs depth:
   - Probe for details they didn't cover
   - Ask about challenges, edge cases, or trade-offs
   - Request specific examples or metrics
   - DO NOT ask for information they already provided!

==== FEW-SHOT EXAMPLES WITH REFERENCING ====

Example 1 - References specific metrics:
Candidate said: "I used FastAPI with async/await to call multiple APIs concurrently. This reduced response time from 6-7 seconds to around 2-3 seconds when verifying borrower identity across government databases and credit bureaus."
✅ GOOD Follow-up: "You mentioned reducing response time from 6-7 seconds to 2-3 seconds with concurrent API calls. How did you handle error scenarios when one of those external services failed or had high latency?"

Example 2 - References specific technology choice:
Candidate said: "I chose PostgreSQL for the main database and Redis for caching frequently accessed data"
✅ GOOD Follow-up: "You chose PostgreSQL with Redis caching - what criteria led you to that combination over alternatives like MongoDB or in-memory databases?"

Example 3 - References specific project and probes for depth:
Candidate said: "I built a real-time notification system using WebSockets that pushed updates to thousands of connected clients"
✅ GOOD Follow-up: "In your WebSocket-based notification system handling thousands of clients - how did you manage connection scaling and ensure message delivery reliability?"

Example 4 - Builds on detailed technical answer:
Candidate said: "We implemented circuit breakers with fallback mechanisms. If an external API failed, we'd return cached data or a default response to avoid blocking the user experience."
✅ GOOD Follow-up: "That's a solid approach with circuit breakers and fallbacks. How did you decide on the thresholds - like how many failures before opening the circuit, or how long to keep it open?"

==== WHAT TO AVOID - ANTI-PATTERNS ====

❌ BAD: Asking for information already provided:
Candidate said: "I used FastAPI because of async/await support and built-in data validation with Pydantic. The main challenge was handling concurrent API calls..."
Follow-up: "Could you elaborate on the challenges you faced?" ← They JUST explained challenges!

❌ BAD: Vague follow-up without referencing:
Candidate gave detailed answer about architecture decisions
Follow-up: "Can you tell me more about that?" ← More about WHAT? Be specific!

❌ BAD: Completely ignoring what they said:
Candidate explained technical implementation with FastAPI
Follow-up: "Tell me about your leadership experience" ← Total topic change, ignores their answer

❌ BAD: Repeating back what they said:
Candidate: "I reduced deployment time by 60% using Kubernetes"
Follow-up: "So you used Kubernetes?" ← They literally just said that!

✅ GOOD: Reference specifics and ask about what's missing:
Candidate: "I reduced deployment time by 60% using Kubernetes"
Follow-up: "That's significant - 60% reduction in deployment time. What were the biggest obstacles you encountered during the Kubernetes migration, and how did you minimize downtime?"

==== REMEMBER ====
- START with what they said (quote metrics, tools, projects)
- BUILD on their answer, don't repeat the question
- ASK about gaps, not information already provided
- Acknowledgement is optional - use only when genuinely warranted

Now generate ONE excellent follow-up question based on their recent answer. Return ONLY the final question text, no explanations."""

FOLLOWUP_QUESTION_HUMAN = """Generate a follow-up question based on the conversation above."""


def create_followup_question_prompt() -> ChatPromptTemplate:
    """
    Create prompt for follow-up questions with enhanced context.

    IMPROVED V2: Now emphasizes referencing specific details from candidate's answer.
    Includes explicit anti-patterns and examples showing proper referencing.
    User messages are preserved fully (no truncation) to enable accurate referencing.

    Variables:
        - current_topic: Current topic
        - followup_number: Which follow-up (1 or 2)
        - assessment_goals: Specific goals to evaluate for this topic
        - resume_context: Relevant resume excerpts for this topic (up to 1000 chars)
        - topic_context: Last 6 Q&A exchanges on this topic (FULL user messages preserved)

    Returns:
        ChatPromptTemplate for streaming follow-up questions that reference specifics
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(FOLLOWUP_QUESTION_SYSTEM),
        HumanMessagePromptTemplate.from_template(FOLLOWUP_QUESTION_HUMAN)
    ])


__all__ = [
    'create_intro_question_prompt',
    'create_topic_question_prompt',
    'create_followup_question_prompt'
]
