# core/domain/api_analyzer.py

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .models import APIDefinition


class APIDefinitionParser:
    """Discovers API definitions from code and spec files."""

    def __init__(self):
        self.supported_formats = ['openapi', 'swagger', 'flask', 'fastapi', 'django']

    async def discover_api_definition(self, project_root: Path) -> Optional[APIDefinition]:
        """Discover API structure from all known sources within the project."""
        api_sources = []

        openapi_specs = await self._discover_openapi_files(project_root)
        if openapi_specs:
            api_sources.append(('openapi', openapi_specs))

        flask_routes = await self._discover_flask_routes(project_root)
        if flask_routes:
            api_sources.append(('flask', flask_routes))

        fastapi_routes = await self._discover_fastapi_routes(project_root)
        if fastapi_routes:
            api_sources.append(('fastapi', fastapi_routes))

        if not api_sources:
            return None

        return await self._combine_api_sources(api_sources)

    async def _discover_openapi_files(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover and parse OpenAPI/Swagger specification files."""
        specs = []
        patterns = ['*.yaml', '*.yml', '*.json']
        for pattern in patterns:
            for file_path in project_root.rglob(pattern):
                if any(k in file_path.name.lower() for k in ['openapi', 'swagger', 'api']):
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if file_path.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                            spec_data = yaml.safe_load(content)
                        else:
                            spec_data = json.loads(content)

                        if 'paths' in spec_data:
                            specs.append({'file_path': str(file_path), 'spec': spec_data})
                    except Exception:
                        continue
        return specs

    async def _discover_flask_routes(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover Flask route definitions by parsing Python files."""
        routes = []
        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if '@app.route' in content or 'from flask import' in content:
                    file_routes = await self._parse_flask_routes(content, py_file)
                    routes.extend(file_routes)
            except Exception:
                continue
        return routes

    async def _parse_flask_routes(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Flask routes from a file's content using AST."""
        routes = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'route':
                            route_info = self._parse_flask_decorator(decorator, node)
                            if route_info:
                                route_info['file_path'] = str(file_path)
                                routes.append(route_info)
        except SyntaxError:
            pass
        return routes

    def _parse_flask_decorator(self, decorator: ast.Call, func_node: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Parse a Flask @app.route() decorator."""
        info = {'function_name': func_node.name, 'path': None, 'methods': ['GET']}
        if decorator.args and isinstance(decorator.args[0], ast.Constant):
            info['path'] = decorator.args[0].value
        for kw in decorator.keywords:
            if kw.arg == 'methods' and isinstance(kw.value, ast.List):
                info['methods'] = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
        return info if info['path'] else None

    async def _discover_fastapi_routes(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover FastAPI routes using regex for simplicity."""
        routes = []
        route_pattern = r'@(\w+)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if 'from fastapi import' in content:
                    for app_var, method, path in re.findall(route_pattern, content):
                        routes.append({'path': path, 'method': method.upper(), 'file_path': str(py_file)})
            except Exception:
                continue
        return routes

    async def _combine_api_sources(self, api_sources: List[Tuple[str, Any]]) -> APIDefinition:
        """Combine API information from all sources into a single object."""
        endpoints, models = [], []
        for source_type, source_data in api_sources:
            if source_type == 'openapi':
                for spec_file in source_data:
                    spec = spec_file['spec']
                    for path, methods in spec.get('paths', {}).items():
                        for method, op in methods.items():
                            endpoints.append({'path': path, 'method': method.upper(), 'summary': op.get('summary', ''), 'source': 'openapi', 'file_path': spec_file['file_path']})
                    for name, props in spec.get('components', {}).get('schemas', {}).items():
                        models.append({'name': name, 'properties': props.get('properties', {}), 'source': 'openapi'})
            elif source_type == 'flask':
                for route in source_data:
                    endpoints.append({'path': route['path'], 'methods': route['methods'], 'function_name': route['function_name'], 'source': 'flask', 'file_path': route['file_path']})
            elif source_type == 'fastapi':
                for route in source_data:
                    endpoints.append({'path': route['path'], 'method': route['method'], 'source': 'fastapi', 'file_path': route['file_path']})

        return APIDefinition(endpoints=endpoints, models=models, authentication={}, middleware=[])