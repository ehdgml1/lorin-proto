"""
Jinja2 Template Loader System for LORIN Prompts
=====================================================

Provides centralized template loading and rendering with caching for performance.
All templates are designed with Tree of Thought patterns and optimized for token efficiency.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from ..logger.logger import get_logger

logger = get_logger(__name__)

class PromptTemplateLoader:
    """
    Centralized Jinja2 template loader for LORIN prompt system.

    Features:
    - Template caching for performance
    - Tree of Thought pattern support
    - Token-efficient rendering
    - English-first design for log analysis
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template loader with specified directory.

        Args:
            template_dir: Path to template directory. Defaults to LORIN/prompt/
        """
        if template_dir is None:
            # Default to LORIN/prompt directory
            current_dir = Path(__file__).parent
            template_dir = str(current_dir)

        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,  # We're generating prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._template_cache: Dict[str, Template] = {}

        logger.info(f"[PromptTemplateLoader] Initialized with directory: {template_dir}")

    def load_template(self, template_name: str) -> Template:
        """
        Load and cache a Jinja2 template.

        Args:
            template_name: Name of template file (e.g., 'planner_tot.j2')

        Returns:
            Jinja2 Template object

        Raises:
            TemplateNotFound: If template file doesn't exist
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        try:
            template = self.env.get_template(template_name)
            self._template_cache[template_name] = template
            logger.debug(f"[PromptTemplateLoader] Loaded template: {template_name}")
            return template
        except TemplateNotFound:
            logger.error(f"[PromptTemplateLoader] Template not found: {template_name}")
            raise

    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a template with provided variables.

        Args:
            template_name: Name of template file
            **kwargs: Template variables

        Returns:
            Rendered prompt string
        """
        try:
            template = self.load_template(template_name)

            # Log decomposition value for debugging
            if 'decomposition' in kwargs:
                logger.info(f"[PromptTemplateLoader] decomposition value: {kwargs['decomposition']}")
                logger.info(f"[PromptTemplateLoader] decomposition type: {type(kwargs['decomposition'])}")
                logger.info(f"[PromptTemplateLoader] decomposition truthiness: {bool(kwargs['decomposition'])}")

            rendered = template.render(**kwargs)

            logger.debug(f"[PromptTemplateLoader] Rendered template: {template_name}")
            logger.debug(f"[PromptTemplateLoader] Template variables: {list(kwargs.keys())}")

            return rendered.strip()
        except Exception as e:
            logger.error(f"[PromptTemplateLoader] Render error for {template_name}: {e}")
            raise

    def clear_cache(self):
        """Clear template cache (useful for development/testing)."""
        self._template_cache.clear()
        logger.info("[PromptTemplateLoader] Template cache cleared")

    def list_templates(self) -> list:
        """List available template files."""
        try:
            template_files = []
            for file in os.listdir(self.template_dir):
                if file.endswith('.j2'):
                    template_files.append(file)
            return sorted(template_files)
        except Exception as e:
            logger.error(f"[PromptTemplateLoader] Error listing templates: {e}")
            return []

# Global template loader instance
_template_loader: Optional[PromptTemplateLoader] = None

def get_template_loader() -> PromptTemplateLoader:
    """
    Get global template loader instance (singleton pattern).

    Returns:
        PromptTemplateLoader instance
    """
    global _template_loader
    if _template_loader is None:
        _template_loader = PromptTemplateLoader()
    return _template_loader

def render_prompt(template_name: str, **kwargs) -> str:
    """
    Convenience function to render a prompt template.

    Args:
        template_name: Name of template file
        **kwargs: Template variables

    Returns:
        Rendered prompt string
    """
    loader = get_template_loader()
    return loader.render_template(template_name, **kwargs)

# Template name constants for type safety
class TemplateNames:
    """Constants for template file names to prevent typos."""
    PLANNER_TOT = "planner_tot.j2"
    ANSWER_SUB = "answer_sub.j2"
    ANSWER_FINAL = "answer_final.j2"
    QUALITY_EVALUATION = "quality_evaluation.j2"
    REPLANNER_ANALYSIS = "replanner_analysis.j2"
    REPLANNER_RECONSTRUCTION = "replanner_reconstruction.j2"