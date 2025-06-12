# core/domain/structure_analyzer.py

import ast
from pathlib import Path
from typing import Dict, Any, List


class ProjectStructureAnalyzer:
    """Analyzes overall project structure, frameworks, and common patterns."""

    def __init__(self):
        self.framework_indicators = {
            'flask': ['from flask', 'Flask(__name__)', 'app.route'],
            'fastapi': ['from fastapi', 'FastAPI()', '@app.get'],
            'django': ['from django', 'DJANGO_SETTINGS_MODULE', 'urls.py'],
            'pyside6': ['from PySide6', 'QApplication', 'QWidget'],
            'tkinter': ['import tkinter', 'from tkinter', 'Tk()'],
            'pygame': ['import pygame', 'pygame.init()'],
            'streamlit': ['import streamlit', 'st.'],
            'jupyter': ['*.ipynb', '.jupyter'],
            'pytest': ['import pytest', 'def test_'],
            'sqlalchemy': ['from sqlalchemy', 'declarative_base'],
            'pandas': ['import pandas', 'pd.DataFrame'],
            'numpy': ['import numpy', 'np.array'],
            'requests': ['import requests', 'requests.get'],
            'asyncio': ['import asyncio', 'async def', 'await ']
        }

    async def analyze_project_structure(self, project_root: Path) -> Dict[str, Any]:
        """Analyze overall project structure and detect patterns"""

        structure_info = {
            'frameworks': [],
            'libraries': [],
            'patterns': [],
            'file_types': {},
            'directory_structure': {},
            'entry_points': [],
            'test_files': [],
            'config_files': [],
            'documentation': []
        }

        # Analyze file types and count
        file_counts = {}
        total_files = 0

        for file_path in project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                ext = file_path.suffix.lower()
                file_counts[ext] = file_counts.get(ext, 0) + 1
                total_files += 1
                await self._analyze_specific_file(file_path, structure_info)

        structure_info['file_types'] = file_counts
        structure_info['total_files'] = total_files
        detected_frameworks = await self._detect_frameworks(project_root)
        structure_info['frameworks'] = detected_frameworks
        structure_info['directory_structure'] = await self._analyze_directory_structure(project_root)
        structure_info['patterns'] = await self._detect_patterns(project_root, detected_frameworks)

        return structure_info

    def _should_ignore_file(self, file_path: Path) -> bool:
        ignore_patterns = {
            '__pycache__', '.git', '.venv', 'venv', 'node_modules',
            '.pytest_cache', '.mypy_cache', '.DS_Store'
        }
        return any(pattern in file_path.parts for pattern in ignore_patterns)

    async def _analyze_specific_file(self, file_path: Path, structure_info: Dict):
        try:
            if file_path.suffix == '.py':
                content = file_path.read_text(encoding='utf-8')
                if 'if __name__ == "__main__"' in content:
                    structure_info['entry_points'].append(str(file_path))
                if (file_path.name.startswith('test_') or
                        file_path.name.endswith('_test.py') or
                        'tests' in file_path.parts):
                    structure_info['test_files'].append(str(file_path))

            elif file_path.name in [
                'config.py', 'settings.py', '.env', 'requirements.txt',
                'pyproject.toml', 'setup.py', 'Pipfile'
            ]:
                structure_info['config_files'].append(str(file_path))

            elif file_path.suffix.lower() in ['.md', '.rst', '.txt']:
                if any(keyword in file_path.name.lower() for keyword in [
                    'readme', 'doc', 'changelog', 'license'
                ]):
                    structure_info['documentation'].append(str(file_path))
        except Exception:
            pass

    async def _detect_frameworks(self, project_root: Path) -> List[Dict[str, Any]]:
        detected = []
        framework_scores = {fw: 0 for fw in self.framework_indicators.keys()}

        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for framework, indicators in self.framework_indicators.items():
                    for indicator in indicators:
                        if indicator in content:
                            framework_scores[framework] += 1
            except Exception:
                continue

        for framework, score in framework_scores.items():
            if score > 0:
                confidence = min(score / 5.0, 1.0)
                detected.append({
                    'name': framework,
                    'confidence': confidence,
                    'occurrences': score
                })
        return sorted(detected, key=lambda x: x['confidence'], reverse=True)

    async def _analyze_directory_structure(self, project_root: Path) -> Dict[str, Any]:
        dirs = {}
        try:
            for item in project_root.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    dirs[item.name] = {
                        'type': self._classify_directory(item),
                        'file_count': sum(1 for _ in item.rglob("*")),
                        'python_files': sum(1 for _ in item.rglob("*.py"))
                    }
        except Exception:
            pass
        return dirs

    def _classify_directory(self, dir_path: Path) -> str:
        dir_name = dir_path.name.lower()
        classification_map = {
            'tests': 'test_directory', 'test': 'test_directory',
            'docs': 'documentation', 'documentation': 'documentation',
            'config': 'configuration', 'settings': 'configuration',
            'static': 'static_assets', 'media': 'media_files',
            'templates': 'templates', 'migrations': 'database_migrations',
            'models': 'data_models', 'views': 'view_layer',
            'controllers': 'controller_layer', 'api': 'api_layer',
            'utils': 'utilities', 'helpers': 'utilities',
            'core': 'core_logic', 'lib': 'library_code',
            'scripts': 'scripts'
        }
        return classification_map.get(dir_name, 'unknown')

    async def _detect_patterns(self, project_root: Path, frameworks: List[Dict]) -> List[str]:
        patterns = []
        try:
            dirs = [d.name.lower() for d in project_root.iterdir() if d.is_dir()]
            if 'models' in dirs and 'views' in dirs and 'controllers' in dirs: patterns.append('mvc_pattern')
            if 'api' in dirs or any('api' in d for d in dirs): patterns.append('api_architecture')
            if 'tests' in dirs or 'test' in dirs: patterns.append('test_driven_development')
            if 'migrations' in dirs: patterns.append('database_migrations')

            framework_names = [fw['name'] for fw in frameworks]
            if 'flask' in framework_names: patterns.append('flask_application')
            if 'django' in framework_names: patterns.append('django_application')
            if 'fastapi' in framework_names: patterns.append('fastapi_application')
        except Exception:
            pass
        return patterns