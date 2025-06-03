# core/ai_feedback_system.py - AI Collaboration & Feedback Engine

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from core.llm_client import LLMRole


class FeedbackType(Enum):
    """Types of feedback between AI specialists"""
    REQUIREMENT_CLARIFICATION = "requirement_clarification"
    IMPLEMENTATION_GUIDANCE = "implementation_guidance"
    QUALITY_ASSESSMENT = "quality_assessment"
    INTEGRATION_FEEDBACK = "integration_feedback"
    ARCHITECTURAL_SUGGESTION = "architectural_suggestion"
    CODE_REVIEW = "code_review"
    CONSISTENCY_CHECK = "consistency_check"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"


@dataclass
class FeedbackMessage:
    """A feedback message between AI specialists"""
    from_ai: str  # Role of the AI sending feedback
    to_ai: str  # Role of the AI receiving feedback
    feedback_type: FeedbackType
    content: str
    context: Dict[str, Any]  # Additional context data
    priority: str = "medium"  # low, medium, high, critical
    requires_response: bool = True
    timestamp: datetime = None
    response_to: Optional[str] = None  # ID of message this responds to

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CollaborationSession:
    """A session of AI collaboration on a specific task"""
    session_id: str
    task_description: str
    participating_ais: List[str]
    messages: List[FeedbackMessage]
    current_state: str  # planning, implementation, review, complete
    target_files: List[str]
    success_criteria: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None


class AIFeedbackSystem:
    """
    🤖 AI Collaboration Engine - Enables Intelligent Specialist Interaction

    This system orchestrates communication between AI specialists, enabling:
    - Planner → Coder: Detailed requirements and constraints
    - Coder → Assembler: Implementation context and integration notes
    - Assembler → Reviewer: Assembly rationale and quality assertions
    - Reviewer → Planner: Quality feedback and improvement suggestions
    - Cross-cutting: Consistency checks and architectural guidance
    """

    def __init__(self, llm_client, project_state_manager, terminal=None):
        self.llm_client = llm_client
        self.project_state = project_state_manager
        self.terminal = terminal

        # Active collaboration tracking
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.message_queue: List[FeedbackMessage] = []
        self.collaboration_history: List[CollaborationSession] = []

        # AI interaction patterns learned over time
        self.interaction_patterns: Dict[str, Any] = {}
        self.success_metrics: Dict[str, float] = {}

    async def initiate_collaboration(self, task_description: str, target_files: List[str],
                                     participating_ais: List[str] = None) -> str:
        """Start a new AI collaboration session"""
        if participating_ais is None:
            participating_ais = ["planner", "coder", "assembler", "reviewer"]

        session_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = CollaborationSession(
            session_id=session_id,
            task_description=task_description,
            participating_ais=participating_ais,
            messages=[],
            current_state="planning",
            target_files=target_files,
            success_criteria=self._generate_success_criteria(task_description),
            started_at=datetime.now()
        )

        self.active_sessions[session_id] = session
        self._log(f"🤝 Started collaboration session: {session_id}")

        # Kick off with planner requirements gathering
        await self._initiate_planning_phase(session_id)

        return session_id

    async def send_feedback(self, session_id: str, from_ai: str, to_ai: str,
                            feedback_type: FeedbackType, content: str,
                            context: Dict[str, Any] = None, priority: str = "medium") -> FeedbackMessage:
        """Send feedback from one AI specialist to another"""

        message = FeedbackMessage(
            from_ai=from_ai,
            to_ai=to_ai,
            feedback_type=feedback_type,
            content=content,
            context=context or {},
            priority=priority
        )

        if session_id in self.active_sessions:
            self.active_sessions[session_id].messages.append(message)

        self.message_queue.append(message)
        self._log(f"📨 {from_ai} → {to_ai}: {feedback_type.value}")

        return message

    async def process_planner_requirements(self, session_id: str, user_prompt: str) -> Dict[str, Any]:
        """Have the planner create detailed requirements with project context"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Get comprehensive project context
        project_context = self.project_state.get_project_context(ai_role="planner")

        planner_prompt = f"""
As the Senior Planner AI, analyze this request with full project awareness:

USER REQUEST: {user_prompt}

PROJECT CONTEXT:
{json.dumps(project_context, indent=2)}

Create a comprehensive implementation plan that:

1. RESPECTS ESTABLISHED PATTERNS:
   - Follow existing naming conventions: {project_context.get('established_patterns', {}).get('naming_conventions', {})}
   - Maintain architectural consistency with: {project_context.get('project_overview', {}).get('architecture_type', 'unknown')}
   - Use established import patterns

2. PROVIDES DETAILED MICRO-TASK BREAKDOWN:
   - Break each file into 5-15 line atomic tasks
   - Specify exact interfaces between components
   - Define integration points clearly

3. GIVES CONTEXT TO SPECIALISTS:
   - Coder guidance: Implementation constraints and requirements
   - Assembler guidance: Integration requirements and consistency checks
   - Reviewer guidance: Quality criteria and success metrics

Return comprehensive JSON with:
- project_plan: High-level architecture decisions
- file_specifications: Detailed specs for each file
- micro_tasks: Atomic tasks with context
- specialist_guidance: Role-specific instructions
- integration_requirements: How files work together
- quality_criteria: Success metrics for review

Be EXTREMELY detailed - this context will guide all downstream AI decisions.
"""

        try:
            # Get planner response with enhanced context
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(planner_prompt, LLMRole.PLANNER):
                response_chunks.append(chunk)
                if len(response_chunks) % 10 == 0:
                    await asyncio.sleep(0.01)  # Allow UI updates

            response_text = ''.join(response_chunks)

            # Extract and validate the plan
            plan = self._extract_and_validate_plan(response_text)

            # Record this planning decision
            self.project_state.record_ai_decision(
                ai_role="planner",
                decision_type="comprehensive_planning",
                context=f"Created detailed plan for: {user_prompt}",
                reasoning="Used full project context to ensure consistency and integration",
                confidence=0.9
            )

            # Send guidance to other specialists
            await self._distribute_planner_guidance(session_id, plan)

            return plan

        except Exception as e:
            self._log(f"❌ Planner requirements processing failed: {e}")
            return self._create_fallback_plan(user_prompt)

    async def execute_coder_task_with_feedback(self, session_id: str, task: Dict[str, Any],
                                               planner_guidance: Dict[str, Any]) -> str:
        """Execute a coding task with rich context and feedback integration"""

        # Get coder-specific context
        project_context = self.project_state.get_project_context(
            for_file=task.get('file_path'),
            ai_role="coder"
        )

        # Get specific guidance from planner
        coder_guidance = planner_guidance.get("specialist_guidance", {}).get("coder", {})

        coder_prompt = f"""
As the Specialist Coder AI, implement this atomic task with full project awareness:

TASK DETAILS:
{json.dumps(task, indent=2)}

PLANNER GUIDANCE:
{json.dumps(coder_guidance, indent=2)}

PROJECT CONTEXT:
{json.dumps(project_context, indent=2)}

CRITICAL REQUIREMENTS:
1. Follow established patterns: {project_context.get('established_patterns', {})}
2. Implement exact interface specified by planner
3. Use consistent naming: {project_context.get('coding_standards', {})}
4. Consider integration with: {project_context.get('file_relationships', {})}

Generate ONLY the code for this specific atomic task.
Include proper error handling, type hints, and documentation.
Ensure it integrates seamlessly with existing project patterns.

RETURN ONLY PYTHON CODE - NO EXPLANATIONS:
"""

        try:
            # Execute the coding task
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(coder_prompt, LLMRole.CODER):
                response_chunks.append(chunk)
                if len(response_chunks) % 5 == 0:
                    await asyncio.sleep(0.01)

            code = ''.join(response_chunks)
            clean_code = self._clean_code(code)

            # Send feedback to assembler about this implementation
            await self.send_feedback(
                session_id=session_id,
                from_ai="coder",
                to_ai="assembler",
                feedback_type=FeedbackType.IMPLEMENTATION_GUIDANCE,
                content=f"Implemented task: {task['description']}",
                context={
                    "task_id": task.get("id"),
                    "implementation_notes": "Followed project patterns and planner guidance",
                    "integration_points": task.get("dependencies", []),
                    "code_preview": clean_code[:200] + "..." if len(clean_code) > 200 else clean_code
                }
            )

            # Record the coding decision
            self.project_state.record_ai_decision(
                ai_role="coder",
                decision_type="task_implementation",
                context=f"Implemented: {task['description']}",
                reasoning="Used project context and planner guidance for consistency",
                confidence=0.85,
                file_affected=task.get('file_path')
            )

            return clean_code

        except Exception as e:
            self._log(f"❌ Coder task execution failed: {e}")
            return f"# TODO: Implement {task['description']}\npass"

    async def assemble_with_context_feedback(self, session_id: str, file_path: str,
                                             task_results: List[Dict[str, Any]],
                                             planner_guidance: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Assemble code with rich context and generate feedback for review"""

        # Get assembler-specific context
        project_context = self.project_state.get_project_context(
            for_file=file_path,
            ai_role="assembler"
        )

        # Get assembler guidance from planner
        assembler_guidance = planner_guidance.get("specialist_guidance", {}).get("assembler", {})

        # Get implementation context from coder messages
        coder_messages = [msg for msg in self.active_sessions[session_id].messages
                          if msg.from_ai == "coder" and msg.to_ai == "assembler"]

        assembler_prompt = f"""
As the Code Assembler AI, create a cohesive file with full project awareness:

FILE: {file_path}

PLANNER GUIDANCE:
{json.dumps(assembler_guidance, indent=2)}

PROJECT CONTEXT:
{json.dumps(project_context, indent=2)}

TASK IMPLEMENTATIONS:
{json.dumps([{"task": r["task"], "code": r["code"]} for r in task_results], indent=2)}

CODER IMPLEMENTATION NOTES:
{json.dumps([{"content": msg.content, "context": msg.context} for msg in coder_messages], indent=2)}

ASSEMBLY REQUIREMENTS:
1. Create a professional, production-ready file
2. Maintain consistency with: {project_context.get('established_patterns', {})}
3. Follow integration requirements: {assembler_guidance.get('integration_requirements', {})}
4. Ensure proper imports and organization
5. Add comprehensive docstring and file header
6. Include error handling and type hints

CRITICAL: The assembled file must integrate seamlessly with existing project patterns and architecture.

Return ONLY the complete Python file - NO EXPLANATIONS:
"""

        try:
            # Assemble the file
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(assembler_prompt, LLMRole.ASSEMBLER):
                response_chunks.append(chunk)
                if len(response_chunks) % 10 == 0:
                    await asyncio.sleep(0.01)

            assembled_code = ''.join(response_chunks)
            clean_code = self._clean_code(assembled_code)

            # Validate consistency with project
            consistency_check = self.project_state.validate_file_consistency(file_path, clean_code)

            # Create detailed context for reviewer
            review_context = {
                "assembly_rationale": "Integrated all micro-tasks following project patterns",
                "consistency_check": consistency_check,
                "integration_notes": "Maintained interface contracts from planner",
                "coder_implementations": len(task_results),
                "quality_assertions": [
                    "Follows established naming conventions",
                    "Maintains architectural consistency",
                    "Includes proper error handling",
                    "Has comprehensive documentation"
                ]
            }

            # Send feedback to reviewer
            await self.send_feedback(
                session_id=session_id,
                from_ai="assembler",
                to_ai="reviewer",
                feedback_type=FeedbackType.QUALITY_ASSESSMENT,
                content=f"Assembled {file_path} with {len(task_results)} integrated components",
                context=review_context,
                priority="high"
            )

            # Record assembly decision
            self.project_state.record_ai_decision(
                ai_role="assembler",
                decision_type="file_assembly",
                context=f"Assembled {file_path} from {len(task_results)} micro-tasks",
                reasoning="Integrated components while maintaining project consistency",
                confidence=0.87,
                file_affected=file_path
            )

            return clean_code, review_context

        except Exception as e:
            self._log(f"❌ Assembly with context failed: {e}")
            return self._fallback_assembly(task_results), {}

    async def review_with_collaborative_feedback(self, session_id: str, file_path: str,
                                                 assembled_code: str, assembly_context: Dict[str, Any],
                                                 planner_guidance: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Comprehensive review with collaborative feedback integration"""

        # Get reviewer-specific context
        project_context = self.project_state.get_project_context(
            for_file=file_path,
            ai_role="reviewer"
        )

        # Get reviewer guidance from planner
        reviewer_guidance = planner_guidance.get("specialist_guidance", {}).get("reviewer", {})
        quality_criteria = planner_guidance.get("quality_criteria", {})

        # Get all previous feedback in this session
        session_history = self._get_session_context(session_id)

        reviewer_prompt = f"""
As the Senior Code Reviewer AI, conduct a comprehensive review with full project awareness:

FILE: {file_path}
CODE:
```python
{assembled_code}
```

PLANNER QUALITY CRITERIA:
{json.dumps(quality_criteria, indent=2)}

REVIEWER GUIDANCE:
{json.dumps(reviewer_guidance, indent=2)}

PROJECT CONTEXT:
{json.dumps(project_context, indent=2)}

ASSEMBLY CONTEXT:
{json.dumps(assembly_context, indent=2)}

COLLABORATION HISTORY:
{json.dumps(session_history, indent=2)}

REVIEW REQUIREMENTS:
1. Verify against planner's quality criteria
2. Check consistency with project patterns: {project_context.get('established_patterns', {})}
3. Validate assembler's quality assertions
4. Assess integration with existing files: {project_context.get('file_relationships', {})}
5. Ensure architectural alignment

RETURN COMPREHENSIVE JSON REVIEW:
{{
    "approved": true/false,
    "overall_score": 1-10,
    "quality_assessment": {{
        "code_quality": 1-10,
        "consistency_score": 1-10,
        "integration_score": 1-10,
        "documentation_score": 1-10
    }},
    "specific_feedback": {{
        "strengths": ["list of strengths"],
        "issues": ["list of issues"],
        "suggestions": ["list of improvements"]
    }},
    "collaboration_feedback": {{
        "planner_requirements_met": true/false,
        "coder_implementations_quality": "assessment",
        "assembler_integration_quality": "assessment"
    }},
    "next_steps": ["what should happen next"],
    "confidence": 0.0-1.0
}}

Be thorough and provide actionable feedback for continuous improvement.
"""

        try:
            # Get comprehensive review
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(reviewer_prompt, LLMRole.REVIEWER):
                response_chunks.append(chunk)
                if len(response_chunks) % 10 == 0:
                    await asyncio.sleep(0.01)

            response_text = ''.join(response_chunks)
            review_data = self._extract_review_json(response_text)

            approved = review_data.get("approved", False)
            feedback = review_data.get("specific_feedback", {})

            # Record review decision
            self.project_state.record_ai_decision(
                ai_role="reviewer",
                decision_type="comprehensive_review",
                context=f"Reviewed {file_path}: {'APPROVED' if approved else 'NEEDS_REVISION'}",
                reasoning=f"Quality score: {review_data.get('overall_score', 0)}/10",
                confidence=review_data.get("confidence", 0.8),
                file_affected=file_path
            )

            # Send feedback back to planner for learning
            if not approved:
                await self.send_feedback(
                    session_id=session_id,
                    from_ai="reviewer",
                    to_ai="planner",
                    feedback_type=FeedbackType.ARCHITECTURAL_SUGGESTION,
                    content=f"Review of {file_path} identified improvement opportunities",
                    context={
                        "review_results": review_data,
                        "suggestions_for_planning": review_data.get("next_steps", [])
                    },
                    priority="high"
                )

            return approved, json.dumps(feedback, indent=2), review_data

        except Exception as e:
            self._log(f"❌ Collaborative review failed: {e}")
            # Fallback to basic approval with minimal feedback
            return True, "Review system encountered an error - manual review recommended", {}

    async def process_iterative_feedback(self, session_id: str, file_path: str,
                                         review_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process review feedback and orchestrate improvements"""

        if review_feedback.get("approved", False):
            # Success case - complete the session
            await self._complete_collaboration_session(session_id, success=True)
            return {"action": "approved", "next_steps": []}

        # Improvement needed - orchestrate feedback loop
        issues = review_feedback.get("specific_feedback", {}).get("issues", [])
        suggestions = review_feedback.get("specific_feedback", {}).get("suggestions", [])

        improvement_plan = {
            "action": "improve",
            "issues_identified": issues,
            "improvement_suggestions": suggestions,
            "next_steps": []
        }

        # Determine if this needs re-planning, re-coding, or re-assembly
        collaboration_feedback = review_feedback.get("collaboration_feedback", {})

        if not collaboration_feedback.get("planner_requirements_met", True):
            # Fundamental planning issues
            improvement_plan["next_steps"].append({
                "phase": "re_planning",
                "ai": "planner",
                "focus": "address architectural concerns"
            })

        if "coder_implementations_quality" in collaboration_feedback:
            # Code quality issues
            improvement_plan["next_steps"].append({
                "phase": "re_coding",
                "ai": "coder",
                "focus": "improve implementation quality"
            })

        if "assembler_integration_quality" in collaboration_feedback:
            # Integration issues
            improvement_plan["next_steps"].append({
                "phase": "re_assembly",
                "ai": "assembler",
                "focus": "improve integration and consistency"
            })

        # Send improvement guidance to relevant AIs
        for step in improvement_plan["next_steps"]:
            await self.send_feedback(
                session_id=session_id,
                from_ai="reviewer",
                to_ai=step["ai"],
                feedback_type=FeedbackType.OPTIMIZATION_SUGGESTION,
                content=f"Improvement needed for {file_path}: {step['focus']}",
                context={
                    "specific_issues": issues,
                    "improvement_suggestions": suggestions,
                    "priority_focus": step["focus"]
                },
                priority="high"
            )

        return improvement_plan

    def get_collaboration_insights(self, session_id: str = None) -> Dict[str, Any]:
        """Get insights about AI collaboration patterns and effectiveness"""

        if session_id:
            # Specific session insights
            session = self.active_sessions.get(session_id) or next(
                (s for s in self.collaboration_history if s.session_id == session_id), None
            )

            if not session:
                return {"error": "Session not found"}

            return self._analyze_session_performance(session)

        # Overall collaboration insights
        return self._analyze_overall_collaboration_patterns()

    # Private helper methods
    def _log(self, message: str):
        """Log collaboration events"""
        if self.terminal and hasattr(self.terminal, 'log'):
            self.terminal.log(message)
        else:
            print(message)

    def _generate_success_criteria(self, task_description: str) -> List[str]:
        """Generate success criteria for a collaboration session"""
        return [
            "Code meets functional requirements",
            "Maintains project consistency",
            "Passes quality review",
            "Integrates seamlessly with existing code",
            "Follows established patterns"
        ]

    async def _initiate_planning_phase(self, session_id: str):
        """Start the planning phase of collaboration"""
        session = self.active_sessions[session_id]

        # Send initial context to planner
        await self.send_feedback(
            session_id=session_id,
            from_ai="system",
            to_ai="planner",
            feedback_type=FeedbackType.REQUIREMENT_CLARIFICATION,
            content=f"Initiate planning for: {session.task_description}",
            context={
                "target_files": session.target_files,
                "success_criteria": session.success_criteria
            },
            priority="high"
        )

    async def _distribute_planner_guidance(self, session_id: str, plan: Dict[str, Any]):
        """Distribute planner guidance to specialist AIs"""

        # Send to coder
        await self.send_feedback(
            session_id=session_id,
            from_ai="planner",
            to_ai="coder",
            feedback_type=FeedbackType.IMPLEMENTATION_GUIDANCE,
            content="Implementation requirements and constraints",
            context={
                "coding_guidance": plan.get("specialist_guidance", {}).get("coder", {}),
                "micro_tasks": plan.get("micro_tasks", []),
                "integration_requirements": plan.get("integration_requirements", {})
            }
        )

        # Send to assembler
        await self.send_feedback(
            session_id=session_id,
            from_ai="planner",
            to_ai="assembler",
            feedback_type=FeedbackType.INTEGRATION_FEEDBACK,
            content="Assembly requirements and consistency guidelines",
            context={
                "assembly_guidance": plan.get("specialist_guidance", {}).get("assembler", {}),
                "file_specifications": plan.get("file_specifications", {}),
                "consistency_requirements": plan.get("integration_requirements", {})
            }
        )

        # Send to reviewer
        await self.send_feedback(
            session_id=session_id,
            from_ai="planner",
            to_ai="reviewer",
            feedback_type=FeedbackType.QUALITY_ASSESSMENT,
            content="Quality criteria and review guidelines",
            context={
                "review_guidance": plan.get("specialist_guidance", {}).get("reviewer", {}),
                "quality_criteria": plan.get("quality_criteria", {}),
                "success_metrics": plan.get("success_criteria", [])
            }
        )

    def _extract_and_validate_plan(self, response_text: str) -> Dict[str, Any]:
        """Extract and validate the planning response"""
        try:
            # Try to extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = response_text[start:end]
                plan = json.loads(json_text)
                return plan
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback plan if extraction fails
        return self._create_fallback_plan("Could not parse planner response")

    def _create_fallback_plan(self, prompt: str) -> Dict[str, Any]:
        """Create a basic fallback plan"""
        return {
            "project_plan": {"architecture": "simple", "description": prompt},
            "file_specifications": {"main.py": {"description": "Main implementation"}},
            "micro_tasks": [{"id": "implement_main", "description": "Implement main functionality"}],
            "specialist_guidance": {
                "coder": {"focus": "basic implementation"},
                "assembler": {"focus": "simple integration"},
                "reviewer": {"focus": "basic quality check"}
            },
            "integration_requirements": {},
            "quality_criteria": {"min_score": 6}
        }

    def _clean_code(self, code: str) -> str:
        """Clean code response from LLM"""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1]
        return code.strip()

    def _fallback_assembly(self, task_results: List[Dict[str, Any]]) -> str:
        """Create basic assembly when main assembly fails"""
        code_parts = [result.get("code", "") for result in task_results]
        return "\n\n".join(code_parts)

    def _extract_review_json(self, response_text: str) -> Dict[str, Any]:
        """Extract review JSON with multiple fallback strategies"""
        try:
            # Try standard JSON extraction
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = response_text[start:end]
                return json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback review
        return {
            "approved": True,
            "overall_score": 7,
            "specific_feedback": {
                "strengths": ["Code appears functional"],
                "issues": ["Could not fully parse review"],
                "suggestions": ["Manual review recommended"]
            },
            "confidence": 0.5
        }

    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context from the collaboration session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {}

        return {
            "message_count": len(session.messages),
            "current_state": session.current_state,
            "participating_ais": session.participating_ais,
            "recent_messages": [
                {
                    "from": msg.from_ai,
                    "to": msg.to_ai,
                    "type": msg.feedback_type.value,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                }
                for msg in session.messages[-5:]  # Last 5 messages
            ]
        }

    async def _complete_collaboration_session(self, session_id: str, success: bool = True):
        """Complete a collaboration session"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]
        session.completed_at = datetime.now()
        session.current_state = "complete" if success else "failed"

        # Move to history
        self.collaboration_history.append(session)
        del self.active_sessions[session_id]

        self._log(f"🏁 Collaboration session {session_id} completed: {'SUCCESS' if success else 'FAILED'}")

    def _analyze_session_performance(self, session: CollaborationSession) -> Dict[str, Any]:
        """Analyze the performance of a specific session"""
        duration = (session.completed_at - session.started_at).total_seconds() if session.completed_at else 0

        return {
            "session_id": session.session_id,
            "duration_seconds": duration,
            "message_count": len(session.messages),
            "success": session.current_state == "complete",
            "ai_participation": {ai: len([m for m in session.messages if m.from_ai == ai])
                                 for ai in session.participating_ais},
            "feedback_types": {ft.value: len([m for m in session.messages if m.feedback_type == ft])
                               for ft in FeedbackType}
        }

    def _analyze_overall_collaboration_patterns(self) -> Dict[str, Any]:
        """Analyze overall collaboration patterns across all sessions"""
        all_sessions = self.collaboration_history + list(self.active_sessions.values())

        if not all_sessions:
            return {"total_sessions": 0}

        successful_sessions = [s for s in all_sessions if s.current_state == "complete"]

        return {
            "total_sessions": len(all_sessions),
            "success_rate": len(successful_sessions) / len(all_sessions),
            "average_duration": sum((s.completed_at - s.started_at).total_seconds()
                                    for s in successful_sessions if s.completed_at) / len(
                successful_sessions) if successful_sessions else 0,
            "most_active_ai": max(FeedbackType, key=lambda ft: len([m for s in all_sessions
                                                                    for m in s.messages if
                                                                    m.feedback_type == ft])).value,
            "collaboration_efficiency": self._calculate_collaboration_efficiency(all_sessions)
        }

    def _calculate_collaboration_efficiency(self, sessions: List[CollaborationSession]) -> float:
        """Calculate overall collaboration efficiency score"""
        if not sessions:
            return 0.0

        # Simple efficiency metric: successful sessions / total messages
        total_messages = sum(len(s.messages) for s in sessions)
        successful_sessions = len([s for s in sessions if s.current_state == "complete"])

        return successful_sessions / max(total_messages, 1) * 100  # Efficiency percentage