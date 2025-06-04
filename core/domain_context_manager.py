# core/domain_context_manager.py - Domain-Specific Context Discovery & Management

import json
import sqlite3
import hashlib
import asyncio
import inspect
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

try:
    import sqlalchemy
    from sqlalchemy import inspect as sqlalchemy_inspect

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ContextType(Enum):
    DATABASE_SCHEMA = "database_schema"
    API_DEFINITION = "api_definition"
    PROJECT_STRUCTURE = "project_structure"
    CODE_PATTERNS = "code_patterns"
    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    TESTS = "tests"


@dataclass
class ContextItem:
    """Single piece of discovered context"""
    context_type: ContextType
    name: str
    content: Any
    metadata: Dict[str, Any]
    confidence: float
    source_file: Optional[str] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class DatabaseSchema:
    """Represents discovered database schema"""
    tables: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    orm_models: List[Dict[str, Any]]


@dataclass
class APIDefinition:
    """Represents discovered API structure"""
    endpoints: List[Dict[str, Any]]
    models: List[Dict[str, Any]]
    authentication: Dict[str, Any]
    middleware: List[str]
    base_url: Optional[str] = None


class DatabaseSchemaIntrospector:
    """Discovers database schemas from various sources"""

    def __init__(self):
        self.supported_orms = ['sqlalchemy', 'django', 'peewee']

    async def discover_schema(self, project_root: Path) -> Optional[DatabaseSchema]:
        """Discover database schema from project files"""

        schema_sources = []

        # 1. Look for SQLAlchemy models
        if SQLALCHEMY_AVAILABLE:
            sqlalchemy_models = await self._discover_sqlalchemy_models(project_root)
            if sqlalchemy_models:
                schema_sources.append(('sqlalchemy', sqlalchemy_models))

        # 2. Look for Django models
        django_models = await self._discover_django_models(project_root)
        if django_models:
            schema_sources.append(('django', django_models))

        # 3. Look for database migration files
        migrations = await self._discover_migrations(project_root)
        if migrations:
            schema_sources.append(('migrations', migrations))

        # 4. Look for SQL schema files
        sql_schemas = await self._discover_sql_files(project_root)
        if sql_schemas:
            schema_sources.append(('sql', sql_schemas))

        if not schema_sources:
            return None

        # Combine and normalize schema information
        return await self._combine_schema_sources(schema_sources)

    async def _discover_sqlalchemy_models(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover SQLAlchemy model definitions"""
        models = []

        # Find Python files that might contain models
        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')

                # Look for SQLAlchemy patterns
                if any(pattern in content for pattern in [
                    'from sqlalchemy', 'import sqlalchemy',
                    'db.Model', 'Base =', 'declarative_base'
                ]):
                    # Parse AST to extract model information
                    model_info = await self._parse_sqlalchemy_models(content, py_file)
                    models.extend(model_info)

            except Exception as e:
                print(f"Error parsing {py_file}: {e}")
                continue

        return models

    async def _parse_sqlalchemy_models(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse SQLAlchemy models from Python file content"""
        models = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if this looks like a SQLAlchemy model
                    is_model = False
                    table_name = None
                    columns = []
                    relationships = []

                    # Check base classes
                    for base in node.bases:
                        if isinstance(base, ast.Attribute) and base.attr == 'Model':
                            is_model = True
                        elif isinstance(base, ast.Name) and base.id in ['Base', 'Model']:
                            is_model = True

                    if is_model:
                        # Extract table name and columns
                        for item in node.body:
                            if isinstance(item, ast.Assign):
                                if (len(item.targets) == 1 and
                                        isinstance(item.targets[0], ast.Name)):

                                    attr_name = item.targets[0].id

                                    if attr_name == '__tablename__':
                                        if isinstance(item.value, ast.Constant):
                                            table_name = item.value.value

                                    # Look for Column definitions
                                    elif isinstance(item.value, ast.Call):
                                        if (isinstance(item.value.func, ast.Name) and
                                                item.value.func.id == 'Column'):
                                            columns.append(self._parse_column_definition(attr_name, item.value))
                                        elif (isinstance(item.value.func, ast.Name) and
                                              item.value.func.id in ['relationship', 'backref']):
                                            relationships.append(self._parse_relationship(attr_name, item.value))

                        if columns:  # Only add if we found columns
                            models.append({
                                'name': node.name,
                                'table_name': table_name or node.name.lower(),
                                'columns': columns,
                                'relationships': relationships,
                                'file_path': str(file_path),
                                'line_number': node.lineno
                            })

        except SyntaxError:
            # File has syntax errors, skip
            pass

        return models

    def _parse_column_definition(self, name: str, call_node: ast.Call) -> Dict[str, Any]:
        """Parse SQLAlchemy Column definition"""
        column_info = {
            'name': name,
            'type': 'Unknown',
            'nullable': True,
            'primary_key': False,
            'foreign_key': None,
            'default': None
        }

        # Parse arguments
        for arg in call_node.args:
            if isinstance(arg, ast.Name):
                column_info['type'] = arg.id
            elif isinstance(arg, ast.Attribute):
                column_info['type'] = f"{arg.value.id}.{arg.attr}"

        # Parse keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg == 'nullable':
                if isinstance(keyword.value, ast.Constant):
                    column_info['nullable'] = keyword.value.value
            elif keyword.arg == 'primary_key':
                if isinstance(keyword.value, ast.Constant):
                    column_info['primary_key'] = keyword.value.value
            elif keyword.arg == 'default':
                if isinstance(keyword.value, ast.Constant):
                    column_info['default'] = keyword.value.value

        return column_info

    def _parse_relationship(self, name: str, call_node: ast.Call) -> Dict[str, Any]:
        """Parse SQLAlchemy relationship definition"""
        rel_info = {
            'name': name,
            'target_model': None,
            'back_populates': None,
            'relationship_type': 'unknown'
        }

        # First argument is usually the target model
        if call_node.args:
            arg = call_node.args[0]
            if isinstance(arg, ast.Constant):
                rel_info['target_model'] = arg.value
            elif isinstance(arg, ast.Name):
                rel_info['target_model'] = arg.id

        # Parse keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg == 'back_populates':
                if isinstance(keyword.value, ast.Constant):
                    rel_info['back_populates'] = keyword.value.value
            elif keyword.arg == 'backref':
                if isinstance(keyword.value, ast.Constant):
                    rel_info['back_populates'] = keyword.value.value

        return rel_info

    async def _discover_django_models(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover Django model definitions"""
        models = []

        # Look for Django models.py files or models in apps
        for py_file in project_root.rglob("*.py"):
            if py_file.name == "models.py" or "/models/" in str(py_file):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    if 'from django.db import models' in content or 'django.db.models' in content:
                        django_models = await self._parse_django_models(content, py_file)
                        models.extend(django_models)
                except Exception as e:
                    print(f"Error parsing Django models in {py_file}: {e}")

        return models

    async def _parse_django_models(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Django models from Python file content"""
        models = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if this inherits from models.Model
                    is_django_model = False
                    for base in node.bases:
                        if (isinstance(base, ast.Attribute) and
                                base.attr == 'Model' and
                                isinstance(base.value, ast.Name) and
                                base.value.id == 'models'):
                            is_django_model = True
                            break

                    if is_django_model:
                        fields = []
                        meta_info = {}

                        for item in node.body:
                            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                                if isinstance(item.targets[0], ast.Name):
                                    field_name = item.targets[0].id

                                    if isinstance(item.value, ast.Call):
                                        # Django field definition
                                        field_info = self._parse_django_field(field_name, item.value)
                                        if field_info:
                                            fields.append(field_info)

                            elif isinstance(item, ast.ClassDef) and item.name == 'Meta':
                                # Parse Meta class
                                meta_info = self._parse_django_meta(item)

                        if fields:
                            models.append({
                                'name': node.name,
                                'table_name': meta_info.get('db_table', node.name.lower()),
                                'fields': fields,
                                'meta': meta_info,
                                'file_path': str(file_path),
                                'line_number': node.lineno
                            })

        except SyntaxError:
            pass

        return models

    def _parse_django_field(self, name: str, call_node: ast.Call) -> Optional[Dict[str, Any]]:
        """Parse Django field definition"""
        if not isinstance(call_node.func, ast.Attribute):
            return None

        field_type = call_node.func.attr

        field_info = {
            'name': name,
            'type': field_type,
            'max_length': None,
            'null': False,
            'blank': False,
            'default': None,
            'related_model': None
        }

        # Parse field arguments
        for keyword in call_node.keywords:
            if keyword.arg in ['max_length', 'null', 'blank', 'default']:
                if isinstance(keyword.value, ast.Constant):
                    field_info[keyword.arg] = keyword.value.value
            elif keyword.arg == 'to':
                if isinstance(keyword.value, ast.Constant):
                    field_info['related_model'] = keyword.value.value
                elif isinstance(keyword.value, ast.Name):
                    field_info['related_model'] = keyword.value.id

        # For ForeignKey, first arg is the related model
        if field_type in ['ForeignKey', 'OneToOneField', 'ManyToManyField'] and call_node.args:
            arg = call_node.args[0]
            if isinstance(arg, ast.Constant):
                field_info['related_model'] = arg.value
            elif isinstance(arg, ast.Name):
                field_info['related_model'] = arg.id

        return field_info

    def _parse_django_meta(self, meta_class: ast.ClassDef) -> Dict[str, Any]:
        """Parse Django model Meta class"""
        meta_info = {}

        for item in meta_class.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                if isinstance(item.targets[0], ast.Name):
                    attr_name = item.targets[0].id

                    if isinstance(item.value, ast.Constant):
                        meta_info[attr_name] = item.value.value
                    elif isinstance(item.value, ast.List):
                        # Handle lists like ordering
                        values = []
                        for elt in item.value.elts:
                            if isinstance(elt, ast.Constant):
                                values.append(elt.value)
                        meta_info[attr_name] = values

        return meta_info

    async def _discover_migrations(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover database migration files"""
        migrations = []

        # Look for migration directories
        for migration_dir in project_root.rglob("migrations"):
            if migration_dir.is_dir():
                for migration_file in migration_dir.glob("*.py"):
                    if migration_file.name != "__init__.py":
                        try:
                            content = migration_file.read_text(encoding='utf-8')
                            migration_info = await self._parse_migration_file(content, migration_file)
                            if migration_info:
                                migrations.append(migration_info)
                        except Exception as e:
                            print(f"Error parsing migration {migration_file}: {e}")

        return migrations

    async def _parse_migration_file(self, content: str, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse migration file for schema information"""
        # This would contain logic to parse Django/SQLAlchemy migration files
        # For now, return basic information
        return {
            'file_path': str(file_path),
            'type': 'migration',
            'operations': []  # Would extract actual operations
        }

    async def _discover_sql_files(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover SQL schema files"""
        sql_files = []

        for sql_file in project_root.rglob("*.sql"):
            try:
                content = sql_file.read_text(encoding='utf-8')

                # Basic SQL parsing to extract table definitions
                tables = self._extract_sql_tables(content)

                if tables:
                    sql_files.append({
                        'file_path': str(sql_file),
                        'tables': tables,
                        'type': 'sql_schema'
                    })

            except Exception as e:
                print(f"Error parsing SQL file {sql_file}: {e}")

        return sql_files

    def _extract_sql_tables(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract table definitions from SQL content"""
        tables = []

        # Simple regex to find CREATE TABLE statements
        create_table_pattern = r'CREATE\s+TABLE\s+([`"\w]+)\s*\((.*?)\);'
        matches = re.findall(create_table_pattern, sql_content, re.IGNORECASE | re.DOTALL)

        for table_name, columns_sql in matches:
            table_name = table_name.strip('`"')
            columns = self._parse_sql_columns(columns_sql)

            tables.append({
                'name': table_name,
                'columns': columns
            })

        return tables

    def _parse_sql_columns(self, columns_sql: str) -> List[Dict[str, Any]]:
        """Parse SQL column definitions"""
        columns = []

        # Split by commas, but be careful of commas in constraints
        column_lines = []
        current_line = ""
        paren_count = 0

        for char in columns_sql:
            current_line += char
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                column_lines.append(current_line[:-1].strip())
                current_line = ""

        if current_line.strip():
            column_lines.append(current_line.strip())

        for line in column_lines:
            line = line.strip()
            if line and not line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'INDEX')):
                # Parse individual column
                parts = line.split()
                if len(parts) >= 2:
                    column_name = parts[0].strip('`"')
                    column_type = parts[1]

                    column_info = {
                        'name': column_name,
                        'type': column_type,
                        'nullable': 'NOT NULL' not in line.upper(),
                        'primary_key': 'PRIMARY KEY' in line.upper(),
                        'auto_increment': 'AUTO_INCREMENT' in line.upper() or 'AUTOINCREMENT' in line.upper()
                    }

                    columns.append(column_info)

        return columns

    async def _combine_schema_sources(self, schema_sources: List[Tuple[str, Any]]) -> DatabaseSchema:
        """Combine schema information from multiple sources"""
        all_tables = []
        all_relationships = []
        all_indexes = []
        all_constraints = []
        all_orm_models = []

        for source_type, source_data in schema_sources:
            if source_type == 'sqlalchemy':
                for model in source_data:
                    all_tables.append({
                        'name': model['table_name'],
                        'columns': model['columns'],
                        'source': 'sqlalchemy',
                        'model_class': model['name']
                    })

                    all_orm_models.append({
                        'name': model['name'],
                        'type': 'sqlalchemy',
                        'table_name': model['table_name'],
                        'file_path': model['file_path']
                    })

                    # Add relationships
                    for rel in model['relationships']:
                        all_relationships.append({
                            'from_table': model['table_name'],
                            'to_table': rel.get('target_model', '').lower(),
                            'type': 'sqlalchemy_relationship',
                            'name': rel['name']
                        })

            elif source_type == 'django':
                for model in source_data:
                    all_tables.append({
                        'name': model['table_name'],
                        'columns': model['fields'],
                        'source': 'django',
                        'model_class': model['name']
                    })

                    all_orm_models.append({
                        'name': model['name'],
                        'type': 'django',
                        'table_name': model['table_name'],
                        'file_path': model['file_path']
                    })

            elif source_type == 'sql':
                for sql_file in source_data:
                    for table in sql_file['tables']:
                        all_tables.append({
                            'name': table['name'],
                            'columns': table['columns'],
                            'source': 'sql',
                            'file_path': sql_file['file_path']
                        })

        return DatabaseSchema(
            tables=all_tables,
            relationships=all_relationships,
            indexes=all_indexes,
            constraints=all_constraints,
            orm_models=all_orm_models
        )


class APIDefinitionParser:
    """Discovers API definitions from various sources"""

    def __init__(self):
        self.supported_formats = ['openapi', 'swagger', 'flask', 'fastapi', 'django']

    async def discover_api_definition(self, project_root: Path) -> Optional[APIDefinition]:
        """Discover API structure from project files"""

        api_sources = []

        # 1. Look for OpenAPI/Swagger files
        openapi_specs = await self._discover_openapi_files(project_root)
        if openapi_specs:
            api_sources.append(('openapi', openapi_specs))

        # 2. Look for Flask route definitions
        flask_routes = await self._discover_flask_routes(project_root)
        if flask_routes:
            api_sources.append(('flask', flask_routes))

        # 3. Look for FastAPI route definitions
        fastapi_routes = await self._discover_fastapi_routes(project_root)
        if fastapi_routes:
            api_sources.append(('fastapi', fastapi_routes))

        # 4. Look for Django URL patterns
        django_urls = await self._discover_django_urls(project_root)
        if django_urls:
            api_sources.append(('django', django_urls))

        if not api_sources:
            return None

        return await self._combine_api_sources(api_sources)

    async def _discover_openapi_files(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover OpenAPI/Swagger specification files"""
        specs = []

        # Look for common OpenAPI file patterns
        patterns = ['*.yaml', '*.yml', '*.json']
        potential_files = []

        for pattern in patterns:
            potential_files.extend(project_root.rglob(pattern))

        for file_path in potential_files:
            if any(keyword in file_path.name.lower() for keyword in [
                'openapi', 'swagger', 'api', 'spec'
            ]):
                try:
                    content = file_path.read_text(encoding='utf-8')

                    # Try to parse as YAML or JSON
                    if file_path.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                        spec_data = yaml.safe_load(content)
                    else:
                        spec_data = json.loads(content)

                    # Check if it looks like an OpenAPI spec
                    if ('openapi' in spec_data or 'swagger' in spec_data or
                            'paths' in spec_data):
                        specs.append({
                            'file_path': str(file_path),
                            'spec': spec_data,
                            'type': 'openapi'
                        })

                except Exception as e:
                    print(f"Error parsing potential OpenAPI file {file_path}: {e}")

        return specs

    async def _discover_flask_routes(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover Flask route definitions"""
        routes = []

        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')

                if any(pattern in content for pattern in [
                    '@app.route', '@bp.route', 'from flask import'
                ]):
                    file_routes = await self._parse_flask_routes(content, py_file)
                    routes.extend(file_routes)

            except Exception as e:
                print(f"Error parsing Flask routes in {py_file}: {e}")

        return routes

    async def _parse_flask_routes(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Flask route definitions from Python file"""
        routes = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for route decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            route_info = self._parse_flask_decorator(decorator, node)
                            if route_info:
                                route_info['file_path'] = str(file_path)
                                route_info['line_number'] = node.lineno
                                routes.append(route_info)

        except SyntaxError:
            pass

        return routes

    def _parse_flask_decorator(self, decorator: ast.Call, func_node: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Parse Flask route decorator"""
        if not isinstance(decorator.func, ast.Attribute):
            return None

        if decorator.func.attr != 'route':
            return None

        route_info = {
            'function_name': func_node.name,
            'path': None,
            'methods': ['GET'],  # Default
            'endpoint': None
        }

        # Parse route arguments
        if decorator.args:
            # First argument is usually the path
            if isinstance(decorator.args[0], ast.Constant):
                route_info['path'] = decorator.args[0].value

        # Parse keyword arguments
        for keyword in decorator.keywords:
            if keyword.arg == 'methods':
                if isinstance(keyword.value, ast.List):
                    methods = []
                    for elt in keyword.value.elts:
                        if isinstance(elt, ast.Constant):
                            methods.append(elt.value)
                    route_info['methods'] = methods
            elif keyword.arg == 'endpoint':
                if isinstance(keyword.value, ast.Constant):
                    route_info['endpoint'] = keyword.value.value

        return route_info if route_info['path'] else None

    async def _discover_fastapi_routes(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover FastAPI route definitions"""
        routes = []

        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')

                if any(pattern in content for pattern in [
                    'from fastapi', '@app.get', '@app.post', '@app.put', '@app.delete'
                ]):
                    file_routes = await self._parse_fastapi_routes(content, py_file)
                    routes.extend(file_routes)

            except Exception as e:
                print(f"Error parsing FastAPI routes in {py_file}: {e}")

        return routes

    async def _parse_fastapi_routes(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse FastAPI route definitions"""
        routes = []

        # Look for route decorators like @app.get("/path")
        route_pattern = r'@(\w+)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
        matches = re.findall(route_pattern, content)

        for app_var, method, path in matches:
            routes.append({
                'path': path,
                'method': method.upper(),
                'app_variable': app_var,
                'file_path': str(file_path),
                'type': 'fastapi'
            })

        return routes

    async def _discover_django_urls(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover Django URL patterns"""
        urls = []

        for py_file in project_root.rglob("urls.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if 'urlpatterns' in content:
                    file_urls = await self._parse_django_urls(content, py_file)
                    urls.extend(file_urls)

            except Exception as e:
                print(f"Error parsing Django URLs in {py_file}: {e}")

        return urls

    async def _parse_django_urls(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Django URL patterns"""
        urls = []

        # This would require more sophisticated parsing of Django urlpatterns
        # For now, return basic structure
        return [{
            'file_path': str(file_path),
            'type': 'django_urls',
            'patterns': []  # Would extract actual patterns
        }]

    async def _combine_api_sources(self, api_sources: List[Tuple[str, Any]]) -> APIDefinition:
        """Combine API information from multiple sources"""
        all_endpoints = []
        all_models = []
        authentication = {}
        middleware = []

        for source_type, source_data in api_sources:
            if source_type == 'openapi':
                for spec_file in source_data:
                    spec = spec_file['spec']

                    # Extract endpoints from OpenAPI spec
                    if 'paths' in spec:
                        for path, methods in spec['paths'].items():
                            for method, operation in methods.items():
                                all_endpoints.append({
                                    'path': path,
                                    'method': method.upper(),
                                    'summary': operation.get('summary', ''),
                                    'description': operation.get('description', ''),
                                    'source': 'openapi',
                                    'file_path': spec_file['file_path']
                                })

                    # Extract models from components/definitions
                    if 'components' in spec and 'schemas' in spec['components']:
                        for model_name, model_def in spec['components']['schemas'].items():
                            all_models.append({
                                'name': model_name,
                                'properties': model_def.get('properties', {}),
                                'source': 'openapi'
                            })

            elif source_type == 'flask':
                for route in source_data:
                    all_endpoints.append({
                        'path': route['path'],
                        'methods': route['methods'],
                        'function_name': route['function_name'],
                        'source': 'flask',
                        'file_path': route['file_path']
                    })

            elif source_type == 'fastapi':
                for route in source_data:
                    all_endpoints.append({
                        'path': route['path'],
                        'method': route['method'],
                        'source': 'fastapi',
                        'file_path': route['file_path']
                    })

        return APIDefinition(
            endpoints=all_endpoints,
            models=all_models,
            authentication=authentication,
            middleware=middleware
        )


class ProjectStructureAnalyzer:
    """Analyzes overall project structure and patterns"""

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

                # Analyze specific file types
                await self._analyze_specific_file(file_path, structure_info)

        structure_info['file_types'] = file_counts
        structure_info['total_files'] = total_files

        # Detect frameworks and libraries
        detected_frameworks = await self._detect_frameworks(project_root)
        structure_info['frameworks'] = detected_frameworks

        # Analyze directory structure
        structure_info['directory_structure'] = await self._analyze_directory_structure(project_root)

        # Detect common patterns
        structure_info['patterns'] = await self._detect_patterns(project_root, detected_frameworks)

        return structure_info

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis"""
        ignore_patterns = {
            '__pycache__', '.git', '.venv', 'venv', 'node_modules',
            '.pytest_cache', '.mypy_cache', '.DS_Store'
        }

        return any(pattern in file_path.parts for pattern in ignore_patterns)

    async def _analyze_specific_file(self, file_path: Path, structure_info: Dict):
        """Analyze specific file for patterns and entry points"""

        try:
            if file_path.suffix == '.py':
                content = file_path.read_text(encoding='utf-8')

                # Check for entry points
                if 'if __name__ == "__main__"' in content:
                    structure_info['entry_points'].append(str(file_path))

                # Check for test files
                if (file_path.name.startswith('test_') or
                        file_path.name.endswith('_test.py') or
                        'test' in file_path.parts):
                    structure_info['test_files'].append(str(file_path))

            # Check for config files
            elif file_path.name in [
                'config.py', 'settings.py', '.env', 'requirements.txt',
                'pyproject.toml', 'setup.py', 'Pipfile'
            ]:
                structure_info['config_files'].append(str(file_path))

            # Check for documentation
            elif file_path.suffix.lower() in ['.md', '.rst', '.txt']:
                if any(keyword in file_path.name.lower() for keyword in [
                    'readme', 'doc', 'changelog', 'license'
                ]):
                    structure_info['documentation'].append(str(file_path))

        except Exception:
            # Skip files that can't be read
            pass

    async def _detect_frameworks(self, project_root: Path) -> List[Dict[str, Any]]:
        """Detect frameworks and libraries used in the project"""
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

        # Convert scores to detected frameworks
        for framework, score in framework_scores.items():
            if score > 0:
                confidence = min(score / 5.0, 1.0)  # Normalize to 0-1
                detected.append({
                    'name': framework,
                    'confidence': confidence,
                    'occurrences': score
                })

        return sorted(detected, key=lambda x: x['confidence'], reverse=True)

    async def _analyze_directory_structure(self, project_root: Path) -> Dict[str, Any]:
        """Analyze directory structure patterns"""
        dirs = {}

        for item in project_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                dirs[item.name] = {
                    'type': self._classify_directory(item),
                    'file_count': len(list(item.rglob("*"))),
                    'python_files': len(list(item.rglob("*.py")))
                }

        return dirs

    def _classify_directory(self, dir_path: Path) -> str:
        """Classify directory type based on name and contents"""
        dir_name = dir_path.name.lower()

        classification_map = {
            'tests': 'test_directory',
            'test': 'test_directory',
            'docs': 'documentation',
            'documentation': 'documentation',
            'config': 'configuration',
            'settings': 'configuration',
            'static': 'static_assets',
            'media': 'media_files',
            'templates': 'templates',
            'migrations': 'database_migrations',
            'models': 'data_models',
            'views': 'view_layer',
            'controllers': 'controller_layer',
            'api': 'api_layer',
            'utils': 'utilities',
            'helpers': 'utilities',
            'core': 'core_logic',
            'lib': 'library_code',
            'scripts': 'scripts'
        }

        return classification_map.get(dir_name, 'unknown')

    async def _detect_patterns(self, project_root: Path, frameworks: List[Dict]) -> List[str]:
        """Detect architectural and coding patterns"""
        patterns = []

        # Check for common architectural patterns
        dirs = [d.name.lower() for d in project_root.iterdir() if d.is_dir()]

        if 'models' in dirs and 'views' in dirs and 'controllers' in dirs:
            patterns.append('mvc_pattern')

        if 'api' in dirs or any('api' in d for d in dirs):
            patterns.append('api_architecture')

        if 'tests' in dirs or 'test' in dirs:
            patterns.append('test_driven_development')

        if 'migrations' in dirs:
            patterns.append('database_migrations')

        # Check for specific framework patterns
        framework_names = [fw['name'] for fw in frameworks]

        if 'flask' in framework_names:
            patterns.append('flask_application')

        if 'django' in framework_names:
            patterns.append('django_application')

        if 'fastapi' in framework_names:
            patterns.append('fastapi_application')

        return patterns


class DomainContextManager:
    """
    🎯 Central Domain Context Discovery & Management

    Discovers and caches domain-specific context including:
    - Database schemas and ORM models
    - API definitions and route structures
    - Project architecture patterns
    - Code patterns and conventions
    - Configuration and dependencies
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.cache_db_path = self.project_root / ".ava_context_cache.db"

        # Initialize components
        self.db_introspector = DatabaseSchemaIntrospector()
        self.api_parser = APIDefinitionParser()
        self.structure_analyzer = ProjectStructureAnalyzer()

        # Cache management
        self.context_cache: Dict[str, ContextItem] = {}
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            self.cache_db = sqlite3.connect(str(self.cache_db_path))
            self.cache_db.execute("""
                CREATE TABLE IF NOT EXISTS context_cache (
                    cache_key TEXT PRIMARY KEY,
                    context_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    confidence REAL,
                    source_hash TEXT,
                    last_updated TEXT
                )
            """)
            self.cache_db.commit()
        except Exception as e:
            print(f"Error initializing context cache: {e}")
            self.cache_db = None

    async def get_comprehensive_context(self, domain: str = "auto") -> Dict[str, Any]:
        """Get comprehensive domain context for the project"""

        context = {
            'database_schema': None,
            'api_definition': None,
            'project_structure': None,
            'domain_type': domain,
            'patterns': [],
            'frameworks': [],
            'recommendations': []
        }

        # 1. Discover database schema
        try:
            schema = await self.db_introspector.discover_schema(self.project_root)
            if schema:
                context['database_schema'] = {
                    'tables': schema.tables,
                    'relationships': schema.relationships,
                    'orm_models': schema.orm_models
                }
                self._cache_context('database_schema', schema, 0.9)
        except Exception as e:
            print(f"Error discovering database schema: {e}")

        # 2. Discover API structure
        try:
            api_def = await self.api_parser.discover_api_definition(self.project_root)
            if api_def:
                context['api_definition'] = {
                    'endpoints': api_def.endpoints,
                    'models': api_def.models,
                    'authentication': api_def.authentication
                }
                self._cache_context('api_definition', api_def, 0.9)
        except Exception as e:
            print(f"Error discovering API definition: {e}")

        # 3. Analyze project structure
        try:
            structure = await self.structure_analyzer.analyze_project_structure(self.project_root)
            context['project_structure'] = structure
            context['frameworks'] = structure.get('frameworks', [])
            context['patterns'] = structure.get('patterns', [])
            self._cache_context('project_structure', structure, 0.8)
        except Exception as e:
            print(f"Error analyzing project structure: {e}")

        # 4. Generate domain-specific recommendations
        context['recommendations'] = await self._generate_recommendations(context)

        return context

    async def get_context_for_file(self, file_path: str, context_type: ContextType = None) -> Dict[str, Any]:
        """Get context specific to a particular file"""

        cache_key = f"file_context:{file_path}"
        cached = self._get_cached_context(cache_key)

        if cached:
            return cached.content

        file_context = {
            'file_path': file_path,
            'related_models': [],
            'related_endpoints': [],
            'imports_needed': [],
            'patterns_to_follow': [],
            'security_requirements': []
        }

        # Get comprehensive context first
        full_context = await self.get_comprehensive_context()

        # Find related database models
        if full_context['database_schema']:
            file_name = Path(file_path).stem.lower()
            for table in full_context['database_schema']['tables']:
                if (file_name in table['name'].lower() or
                        table['name'].lower() in file_name):
                    file_context['related_models'].append(table)

        # Find related API endpoints
        if full_context['api_definition']:
            for endpoint in full_context['api_definition']['endpoints']:
                if file_path in endpoint.get('file_path', ''):
                    file_context['related_endpoints'].append(endpoint)

        # Determine imports needed based on frameworks
        for framework in full_context['frameworks']:
            if framework['confidence'] > 0.5:
                imports = self._get_framework_imports(framework['name'], file_path)
                file_context['imports_needed'].extend(imports)

        # Cache the result
        self._cache_context(cache_key, file_context, 0.7)

        return file_context

    def _get_framework_imports(self, framework: str, file_path: str) -> List[str]:
        """Get common imports for a framework/file combination"""

        framework_imports = {
            'flask': {
                'api': ['Flask', 'request', 'jsonify', 'abort'],
                'model': ['SQLAlchemy', 'db'],
                'default': ['Flask']
            },
            'fastapi': {
                'api': ['FastAPI', 'HTTPException', 'Depends'],
                'model': ['SQLAlchemy', 'declarative_base'],
                'default': ['FastAPI']
            },
            'django': {
                'model': ['django.db.models'],
                'view': ['django.shortcuts', 'django.http'],
                'default': ['django']
            },
            'pyside6': {
                'gui': ['PySide6.QtWidgets', 'PySide6.QtCore'],
                'default': ['PySide6.QtWidgets']
            }
        }

        if framework not in framework_imports:
            return []

        file_name = Path(file_path).name.lower()

        # Determine file type
        if any(keyword in file_name for keyword in ['api', 'route', 'endpoint']):
            file_type = 'api'
        elif any(keyword in file_name for keyword in ['model', 'schema']):
            file_type = 'model'
        elif any(keyword in file_name for keyword in ['view', 'gui', 'window']):
            file_type = 'view' if framework == 'django' else 'gui'
        else:
            file_type = 'default'

        return framework_imports[framework].get(file_type, framework_imports[framework]['default'])

    async def _generate_recommendations(self, context: Dict[str, Any]) -> List[str]:
        """Generate domain-specific recommendations"""
        recommendations = []

        # Database recommendations
        if context['database_schema']:
            tables = context['database_schema']['tables']
            if len(tables) > 5:
                recommendations.append("Consider implementing database connection pooling for better performance")

            if any('password' in str(table).lower() for table in tables):
                recommendations.append("Ensure password fields are properly hashed using bcrypt or similar")

        # API recommendations
        if context['api_definition']:
            endpoints = context['api_definition']['endpoints']
            if len(endpoints) > 10:
                recommendations.append("Consider implementing API rate limiting and caching")

            if not context['api_definition']['authentication']:
                recommendations.append("Consider implementing authentication and authorization")

        # Framework-specific recommendations
        frameworks = [fw['name'] for fw in context['frameworks'] if fw['confidence'] > 0.5]

        if 'flask' in frameworks:
            recommendations.append("Use Flask-CORS for cross-origin requests")
            recommendations.append("Implement proper error handling with Flask error handlers")

        if 'fastapi' in frameworks:
            recommendations.append("Leverage FastAPI's automatic OpenAPI documentation")
            recommendations.append("Use Pydantic models for request/response validation")

        if 'pyside6' in frameworks:
            recommendations.append("Use Qt's signal-slot mechanism for event handling")
            recommendations.append("Implement proper resource management for GUI components")

        return recommendations

    def _cache_context(self, cache_key: str, content: Any, confidence: float):
        """Cache context information"""
        if not self.cache_db:
            return

        try:
            # Create context item
            context_item = ContextItem(
                context_type=ContextType.PROJECT_STRUCTURE,  # Default
                name=cache_key,
                content=content,
                metadata={},
                confidence=confidence
            )

            # Store in memory cache
            self.context_cache[cache_key] = context_item

            # Store in SQLite cache
            content_json = json.dumps(content, default=str)
            metadata_json = json.dumps(context_item.metadata)

            self.cache_db.execute("""
                INSERT OR REPLACE INTO context_cache 
                (cache_key, context_type, content, metadata, confidence, source_hash, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                context_item.context_type.value,
                content_json,
                metadata_json,
                confidence,
                hashlib.md5(content_json.encode()).hexdigest(),
                context_item.last_updated.isoformat()
            ))
            self.cache_db.commit()

        except Exception as e:
            print(f"Error caching context: {e}")

    def _get_cached_context(self, cache_key: str) -> Optional[ContextItem]:
        """Get cached context information"""

        # Check memory cache first
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]

        # Check SQLite cache
        if not self.cache_db:
            return None

        try:
            cursor = self.cache_db.execute("""
                SELECT context_type, content, metadata, confidence, last_updated
                FROM context_cache WHERE cache_key = ?
            """, (cache_key,))

            row = cursor.fetchone()
            if row:
                context_type_str, content_json, metadata_json, confidence, last_updated_str = row

                content = json.loads(content_json)
                metadata = json.loads(metadata_json)
                last_updated = datetime.fromisoformat(last_updated_str)

                # Check if cache is still fresh (24 hours)
                if (datetime.now() - last_updated).total_seconds() < 86400:
                    context_item = ContextItem(
                        context_type=ContextType(context_type_str),
                        name=cache_key,
                        content=content,
                        metadata=metadata,
                        confidence=confidence,
                        last_updated=last_updated
                    )

                    # Store in memory cache for faster access
                    self.context_cache[cache_key] = context_item
                    return context_item

        except Exception as e:
            print(f"Error retrieving cached context: {e}")

        return None

    def clear_cache(self):
        """Clear all cached context"""
        self.context_cache.clear()

        if self.cache_db:
            try:
                self.cache_db.execute("DELETE FROM context_cache")
                self.cache_db.commit()
            except Exception as e:
                print(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'memory_cache_size': len(self.context_cache),
            'db_cache_size': 0,
            'total_confidence': 0.0,
            'avg_confidence': 0.0
        }

        if self.cache_db:
            try:
                cursor = self.cache_db.execute("SELECT COUNT(*), AVG(confidence) FROM context_cache")
                row = cursor.fetchone()
                if row:
                    stats['db_cache_size'] = row[0] or 0
                    stats['avg_confidence'] = row[1] or 0.0
            except Exception:
                pass

        # Calculate total confidence from memory cache
        if self.context_cache:
            total_conf = sum(item.confidence for item in self.context_cache.values())
            stats['total_confidence'] = total_conf
            stats['avg_confidence'] = total_conf / len(self.context_cache)

        return stats