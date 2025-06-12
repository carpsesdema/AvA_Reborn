# core/domain/database_analyzer.py

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import sqlalchemy
    from sqlalchemy import inspect as sqlalchemy_inspect
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .models import DatabaseSchema


class DatabaseSchemaIntrospector:
    """Discovers database schemas from various sources like ORM models and SQL files."""

    def __init__(self):
        self.supported_orms = ['sqlalchemy', 'django', 'peewee']

    async def discover_schema(self, project_root: Path) -> Optional[DatabaseSchema]:
        """Discover database schema from all known sources within the project."""
        schema_sources = []

        if SQLALCHEMY_AVAILABLE:
            sqlalchemy_models = await self._discover_sqlalchemy_models(project_root)
            if sqlalchemy_models:
                schema_sources.append(('sqlalchemy', sqlalchemy_models))

        django_models = await self._discover_django_models(project_root)
        if django_models:
            schema_sources.append(('django', django_models))

        migrations = await self._discover_migrations(project_root)
        if migrations:
            schema_sources.append(('migrations', migrations))

        sql_schemas = await self._discover_sql_files(project_root)
        if sql_schemas:
            schema_sources.append(('sql', sql_schemas))

        if not schema_sources:
            return None

        return await self._combine_schema_sources(schema_sources)

    async def _discover_sqlalchemy_models(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover SQLAlchemy model definitions by parsing Python files."""
        models = []
        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if any(p in content for p in ['from sqlalchemy', 'declarative_base']):
                    model_info = await self._parse_sqlalchemy_models(content, py_file)
                    models.extend(model_info)
            except Exception:
                continue
        return models

    async def _parse_sqlalchemy_models(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse SQLAlchemy models from a file's content using AST."""
        models = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    is_model = any(
                        (isinstance(b, ast.Attribute) and b.attr == 'Model') or
                        (isinstance(b, ast.Name) and b.id in ['Base', 'Model'])
                        for b in node.bases
                    )
                    if is_model:
                        table_name, columns, relationships = None, [], []
                        for item in node.body:
                            if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                                attr_name = item.targets[0].id
                                if attr_name == '__tablename__':
                                    table_name = item.value.value if isinstance(item.value, ast.Constant) else None
                                elif isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name):
                                    if item.value.func.id == 'Column':
                                        columns.append(self._parse_column_definition(attr_name, item.value))
                                    elif item.value.func.id in ['relationship', 'backref']:
                                        relationships.append(self._parse_relationship(attr_name, item.value))
                        if columns:
                            models.append({'name': node.name, 'table_name': table_name or node.name.lower(),
                                           'columns': columns, 'relationships': relationships,
                                           'file_path': str(file_path), 'line_number': node.lineno})
        except SyntaxError:
            pass
        return models

    def _parse_column_definition(self, name: str, call_node: ast.Call) -> Dict[str, Any]:
        """Parse a SQLAlchemy Column() definition."""
        info = {'name': name, 'type': 'Unknown', 'nullable': True, 'primary_key': False}
        if call_node.args:
            arg = call_node.args[0]
            if isinstance(arg, ast.Name): info['type'] = arg.id
            elif isinstance(arg, ast.Attribute): info['type'] = f"{arg.value.id}.{arg.attr}"
        for kw in call_node.keywords:
            if kw.arg in ['nullable', 'primary_key'] and isinstance(kw.value, ast.Constant):
                info[kw.arg] = kw.value.value
        return info

    def _parse_relationship(self, name: str, call_node: ast.Call) -> Dict[str, Any]:
        """Parse a SQLAlchemy relationship() definition."""
        info = {'name': name, 'target_model': None, 'back_populates': None}
        if call_node.args:
            arg = call_node.args[0]
            if isinstance(arg, ast.Constant): info['target_model'] = arg.value
            elif isinstance(arg, ast.Name): info['target_model'] = arg.id
        for kw in call_node.keywords:
            if kw.arg in ['back_populates', 'backref'] and isinstance(kw.value, ast.Constant):
                info['back_populates'] = kw.value.value
        return info

    async def _discover_django_models(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover Django model definitions."""
        # This is a simplified placeholder. A full implementation would be more complex.
        return []

    async def _parse_django_models(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Django models from a file's content using AST."""
        # Placeholder for Django model parsing logic
        return []

    async def _discover_migrations(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover database migration files."""
        # Placeholder for migration discovery logic
        return []

    async def _discover_sql_files(self, project_root: Path) -> List[Dict[str, Any]]:
        """Discover and parse raw .sql files."""
        sql_files = []
        for sql_file in project_root.rglob("*.sql"):
            try:
                content = sql_file.read_text(encoding='utf-8')
                tables = self._extract_sql_tables(content)
                if tables:
                    sql_files.append({'file_path': str(sql_file), 'tables': tables})
            except Exception:
                continue
        return sql_files

    def _extract_sql_tables(self, sql_content: str) -> List[Dict[str, Any]]:
        """Extract table definitions from raw SQL using regex."""
        tables = []
        pattern = r'CREATE\s+TABLE\s+([`"\w]+)\s*\((.*?)\);'
        for name, cols_sql in re.findall(pattern, sql_content, re.IGNORECASE | re.DOTALL):
            tables.append({'name': name.strip('`"'), 'columns': self._parse_sql_columns(cols_sql)})
        return tables

    def _parse_sql_columns(self, columns_sql: str) -> List[Dict[str, Any]]:
        """Parse column definitions from a CREATE TABLE statement."""
        columns = []
        # Simplified regex, a real parser would be more robust
        col_pattern = r'([`"\w]+)\s+([\w\(\)]+)'
        for name, type in re.findall(col_pattern, columns_sql):
            if name.upper() not in ['PRIMARY', 'FOREIGN', 'CONSTRAINT', 'INDEX', 'KEY']:
                columns.append({'name': name.strip('`"'), 'type': type})
        return columns

    async def _combine_schema_sources(self, schema_sources: List[Tuple[str, Any]]) -> DatabaseSchema:
        """Combine schema information from all discovered sources into a single object."""
        tables, relationships, orm_models = [], [], []
        for source_type, source_data in schema_sources:
            if source_type == 'sqlalchemy':
                for model in source_data:
                    tables.append({'name': model['table_name'], 'columns': model['columns'], 'source': 'sqlalchemy'})
                    orm_models.append({'name': model['name'], 'type': 'sqlalchemy', 'file_path': model['file_path']})
                    for rel in model['relationships']:
                        relationships.append({'from': model['table_name'], 'to': rel.get('target_model', '').lower(), 'name': rel['name']})
            elif source_type == 'sql':
                for sql_file in source_data:
                    for table in sql_file['tables']:
                        tables.append({'name': table['name'], 'columns': table['columns'], 'source': 'sql', 'file_path': sql_file['file_path']})

        return DatabaseSchema(tables=tables, relationships=relationships, indexes=[], constraints=[], orm_models=orm_models)