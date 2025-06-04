# core/enhanced_micro_task_engine.py - Streamlined for RESULTS

import json
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.llm_client import LLMRole


@dataclass
class SimpleTaskSpec:
    """Streamlined task specification - focused on results"""
    id: str
    description: str
    expected_lines: int
    context: str
    exact_requirements: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "expected_lines": self.expected_lines,
            "context": self.context,
            "exact_requirements": self.exact_requirements,
            "type": "code_chunk"
        }


class StreamlinedMicroTaskEngine:
    """
    🚀 Streamlined Micro-Task Engine - RESULTS FOCUSED

    Creates 3-5 logical code chunks with:
    - Domain-aware context
    - Professional-grade prompts
    - Fast execution
    - Real results
    """

    def __init__(self, llm_client, project_state_manager, domain_context_manager):
        self.llm_client = llm_client
        self.project_state = project_state_manager
        self.domain_context = domain_context_manager

    async def create_smart_tasks(self, file_path: str, file_spec: Dict[str, Any],
                                 project_context: Dict[str, Any]) -> List[SimpleTaskSpec]:
        """Create 3-5 logical code chunks with professional context"""

        try:
            # Detect domain
            domain = self._detect_domain_fast(file_path, file_spec, project_context)

            # Get domain-specific task template
            if "gui" in domain or "calculator" in file_spec.get('description', '').lower():
                return await self._create_gui_tasks(file_path, file_spec, project_context)
            elif "api" in domain:
                return await self._create_api_tasks(file_path, file_spec, project_context)
            elif "cli" in domain:
                return await self._create_cli_tasks(file_path, file_spec, project_context)
            else:
                return await self._create_generic_tasks(file_path, file_spec, project_context)

        except Exception as e:
            print(f"[ERROR] Task creation failed: {e}")
            return self._create_fallback_tasks(file_path, file_spec)

    def _detect_domain_fast(self, file_path: str, file_spec: Dict, project_context: Dict) -> str:
        """Fast domain detection"""
        description = (file_spec.get('description', '') + ' ' + file_path).lower()

        if any(word in description for word in ['gui', 'calculator', 'pyside', 'tkinter', 'qt']):
            return "gui"
        elif any(word in description for word in ['api', 'flask', 'fastapi', 'django']):
            return "api"
        elif any(word in description for word in ['cli', 'command', 'script']):
            return "cli"
        else:
            return "generic"

    async def _create_gui_tasks(self, file_path: str, file_spec: Dict, project_context: Dict) -> List[SimpleTaskSpec]:
        """Create GUI-specific tasks (3-4 chunks)"""

        description = file_spec.get('description', 'GUI application')

        tasks = [
            SimpleTaskSpec(
                id="chunk_1_imports_setup",
                description="Import required GUI modules and create main application class",
                expected_lines=25,
                context=f"Creating {description}",
                exact_requirements="""
- Import PySide6 or tkinter modules
- Create main application class with proper inheritance
- Set up window properties (title, size, theme)
- Initialize layout structure
- Include proper error handling and logging
"""
            ),

            SimpleTaskSpec(
                id="chunk_2_ui_components",
                description="Create all UI components and layout",
                expected_lines=35,
                context=f"Building UI for {description}",
                exact_requirements="""
- Create all necessary widgets/components
- Set up proper layout management
- Apply styling and themes
- Configure widget properties and connections
- Ensure responsive design principles
"""
            ),

            SimpleTaskSpec(
                id="chunk_3_event_logic",
                description="Implement event handlers and business logic",
                expected_lines=30,
                context=f"Adding functionality to {description}",
                exact_requirements="""
- Implement all event handlers and callbacks
- Add business logic and calculations
- Include input validation and error handling
- Ensure proper state management
- Add keyboard shortcuts if applicable
"""
            ),

            SimpleTaskSpec(
                id="chunk_4_main_execution",
                description="Create main execution block with proper initialization",
                expected_lines=15,
                context=f"Finalizing {description}",
                exact_requirements="""
- Create main execution block with if __name__ == '__main__'
- Proper application initialization and error handling
- Resource cleanup and graceful shutdown
- Cross-platform compatibility considerations
"""
            )
        ]

        return tasks

    async def _create_api_tasks(self, file_path: str, file_spec: Dict, project_context: Dict) -> List[SimpleTaskSpec]:
        """Create API-specific tasks"""

        description = file_spec.get('description', 'API application')

        tasks = [
            SimpleTaskSpec(
                id="chunk_1_setup",
                description="Import modules and create API application setup",
                expected_lines=20,
                context=f"Setting up {description}",
                exact_requirements="""
- Import Flask/FastAPI/Django modules
- Create application instance with configuration
- Set up error handling and logging
- Configure CORS and security settings
"""
            ),

            SimpleTaskSpec(
                id="chunk_2_routes",
                description="Implement API routes and endpoints",
                expected_lines=40,
                context=f"Creating API endpoints for {description}",
                exact_requirements="""
- Define all API routes with proper HTTP methods
- Implement request validation and parsing
- Add authentication and authorization
- Include comprehensive error handling
- Proper response formatting
"""
            ),

            SimpleTaskSpec(
                id="chunk_3_main",
                description="Create main execution with proper server setup",
                expected_lines=15,
                context=f"Finalizing {description}",
                exact_requirements="""
- Main execution block with server configuration
- Environment-based configuration
- Graceful shutdown handling
- Production-ready settings
"""
            )
        ]

        return tasks

    async def _create_cli_tasks(self, file_path: str, file_spec: Dict, project_context: Dict) -> List[SimpleTaskSpec]:
        """Create CLI-specific tasks"""

        description = file_spec.get('description', 'CLI application')

        tasks = [
            SimpleTaskSpec(
                id="chunk_1_setup",
                description="Import modules and set up argument parsing",
                expected_lines=25,
                context=f"Setting up {description}",
                exact_requirements="""
- Import argparse/click modules
- Set up command-line argument parsing
- Configure logging and error handling
- Define help text and usage information
"""
            ),

            SimpleTaskSpec(
                id="chunk_2_commands",
                description="Implement core command functionality",
                expected_lines=35,
                context=f"Building commands for {description}",
                exact_requirements="""
- Implement all command handlers
- Add input validation and error handling
- Include progress indicators where appropriate
- Proper output formatting and user feedback
"""
            ),

            SimpleTaskSpec(
                id="chunk_3_main",
                description="Create main execution with command routing",
                expected_lines=20,
                context=f"Finalizing {description}",
                exact_requirements="""
- Main execution block with command routing
- Global error handling and cleanup
- Exit code management
- Cross-platform compatibility
"""
            )
        ]

        return tasks

    async def _create_generic_tasks(self, file_path: str, file_spec: Dict, project_context: Dict) -> List[
        SimpleTaskSpec]:
        """Create generic tasks for any file"""

        description = file_spec.get('description', 'Python module')

        tasks = [
            SimpleTaskSpec(
                id="chunk_1_setup",
                description="Import modules and set up basic structure",
                expected_lines=20,
                context=f"Setting up {description}",
                exact_requirements="""
- Import all required modules
- Set up logging and configuration
- Define constants and global variables
- Include proper docstrings and type hints
"""
            ),

            SimpleTaskSpec(
                id="chunk_2_core_logic",
                description="Implement main functionality",
                expected_lines=40,
                context=f"Building core logic for {description}",
                exact_requirements="""
- Implement main classes and functions
- Add comprehensive error handling
- Include input validation
- Proper documentation and type hints
- Follow best practices and patterns
"""
            ),

            SimpleTaskSpec(
                id="chunk_3_main",
                description="Create main execution block",
                expected_lines=15,
                context=f"Finalizing {description}",
                exact_requirements="""
- Main execution block if appropriate
- Example usage or demonstration
- Proper error handling and cleanup
"""
            )
        ]

        return tasks

    def _create_fallback_tasks(self, file_path: str, file_spec: Dict) -> List[SimpleTaskSpec]:
        """Simple fallback when everything else fails"""

        return [
            SimpleTaskSpec(
                id="fallback_complete",
                description=f"Implement complete {file_path} functionality",
                expected_lines=50,
                context=f"Creating {file_path}",
                exact_requirements=f"""
- Implement all required functionality for {file_path}
- Include proper imports and structure
- Add error handling and documentation
- Follow Python best practices
- Make it production-ready
"""
            )
        ]