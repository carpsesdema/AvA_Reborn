# integration_helpers.py - Helper functions for error-aware integration

from pathlib import Path
from typing import Optional, Dict, List
import json


def setup_error_aware_ava():
    """
    Setup function to configure AvA for error-aware operation
    Call this in your main initialization
    """
    print("🔧 Setting up Error-Aware AvA...")
    
    # Ensure workspace directory exists
    workspace = Path("./workspace")
    workspace.mkdir(exist_ok=True)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    print("✅ Error-aware AvA setup complete!")


def create_error_analysis_shortcuts():
    """
    Create convenient shortcuts for common error analysis tasks
    Returns a dictionary of helper functions
    """
    
    def quick_error_check(project_path: str) -> Dict:
        """Quick check for common error patterns in project"""
        path = Path(project_path)
        issues = []
        
        # Check for main.py
        if not (path / "main.py").exists():
            issues.append("❌ No main.py file found")
        
        # Check for requirements.txt
        req_file = path / "requirements.txt"
        if req_file.exists():
            try:
                reqs = req_file.read_text().strip().split('\n')
                if len(reqs) > 10:
                    issues.append("⚠️ Many dependencies - potential version conflicts")
            except Exception:
                issues.append("❌ Cannot read requirements.txt")
        
        # Check for Python syntax in main files
        for py_file in path.glob("*.py"):
            try:
                content = py_file.read_text()
                # Basic syntax check
                compile(content, str(py_file), 'exec')
            except SyntaxError as e:
                issues.append(f"❌ Syntax error in {py_file.name}: {e}")
            except Exception:
                pass  # File might be binary or have encoding issues
        
        return {
            "project_path": str(path),
            "issues_found": len(issues),
            "issues": issues,
            "status": "✅ No issues found" if not issues else f"⚠️ {len(issues)} issues found"
        }
    
    def format_error_for_ava(error_context) -> str:
        """Format error context in a way that's optimal for Ava analysis"""
        if not error_context:
            return "No error context available"
        
        formatted = "🔍 ERROR ANALYSIS REQUEST\n\n"
        formatted += f"Command that failed: {error_context.last_command}\n"
        formatted += f"Exit code: {error_context.exit_code}\n"
        formatted += f"Working directory: {error_context.working_directory}\n\n"
        
        if error_context.stderr:
            formatted += "Error output:\n"
            formatted += f"```\n{error_context.stderr}\n```\n\n"
        
        if error_context.stdout and "error" in error_context.stdout.lower():
            formatted += "Standard output (contains errors):\n"
            formatted += f"```\n{error_context.stdout}\n```\n\n"
        
        formatted += "Please analyze this error and provide complete fixed files."
        return formatted
    
    return {
        "quick_check": quick_error_check,
        "format_for_ava": format_error_for_ava
    }


class ErrorAwareWorkflowManager:
    """
    Simplified workflow manager that focuses on error detection and fixing
    """
    
    def __init__(self, ava_app):
        self.ava_app = ava_app
        self.shortcuts = create_error_analysis_shortcuts()
    
    def run_and_analyze(self, project_path: Optional[str] = None) -> bool:
        """
        Run project and automatically analyze any errors that occur
        Returns True if successful, False if errors occurred
        """
        print("🚀 Running project with error analysis...")
        
        # Pre-run check
        path_to_check = project_path or str(self.ava_app.current_project_path)
        pre_check = self.shortcuts["quick_check"](path_to_check)
        
        if pre_check["issues_found"] > 0:
            print("⚠️ Pre-run issues detected:")
            for issue in pre_check["issues"]:
                print(f"  {issue}")
            print("Continuing with execution anyway...")
        
        # Run the project
        success = self.ava_app.run_project_in_terminal(project_path)
        
        # If there were errors, automatically prepare analysis
        if not success and self.ava_app.last_error_context:
            print("❌ Execution failed - preparing error analysis for Ava...")
            error_summary = self.shortcuts["format_for_ava"](self.ava_app.last_error_context)
            print("💡 You can now ask Ava: 'Can you fix the error that just occurred?'")
            return False
        
        return success
    
    def auto_fix_last_error(self):
        """Automatically send the last error to Ava for analysis"""
        if not self.ava_app.last_error_context:
            print("No recent errors to analyze")
            return
        
        print("🤖 Sending error to Ava for automatic analysis...")
        self.ava_app._send_error_to_ava("Please analyze and fix the error that just occurred.")


