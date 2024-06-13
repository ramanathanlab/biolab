"""For registering components components to a global registry."""
from typing import Callable, Type, Optional, Dict, Any
import importlib
import pkgutil
import sys


class Registry:
    """
    A general registry to map names to classes. Is used to
    dynamically instantiate classes based on their name.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Type], Type]:
        """
        A decorator to register a class with an optional name.

        Parameters
        ----------
        name : Optional[str]
            The name to register the class under.
            If not provided, the class name is used.

        Returns
        -------
        Callable[[Type], Type]
            The decorator that registers the class.
        """

        def decorator(cls: Type) -> Type:
            class_name = name if name else cls.__name__
            self._registry[class_name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Optional[Type]:
        """
        Retrieve a registered class by name.

        Parameters
        ----------
        name : str
            The name of the class to retrieve.

        Returns
        -------
        Optional[Type]
            The registered class, or None if not found.
        """
        return self._registry.get(name)


class CoupledRegistry:
    """
    A registry to map names to a set of classes.

    Attributes
    ----------
    _registry : Dict[str, Dict[str, Any]]
        A dictionary storing the mappings from names to set of classes.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self, name: Optional[str] = None, **kwargs: Any
    ) -> Callable[[Type], Type]:
        """
        A decorator to register an element to the registry and set of
        associated classes.

        Parameters
        ----------
        name : Optional[str]
            The name to register the class under. Defaults to class name.
        **kwargs : Any
            Additional information to couple with the class.

        Returns
        -------
        Callable[[Type], Type]
            The decorator that registers the class.
        """

        def decorator(cls: Type) -> Type:
            class_name = name if name else cls.__name__
            if class_name in self._registry:
                self._registry[class_name].update(kwargs)
                self._registry[class_name]["class"] = cls
            else:
                self._registry[class_name] = {"class": cls, **kwargs}
            return cls

        return decorator

    def get(self, name: str, field: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        Retrieve a registered set of classes by name.

        Parameters
        ----------
        name : str
            The name of the class to retrieve associated classes for.
        field : Optional[str]
            The field to retrieve from set of associated classes.

        Returns
        -------
        Optional[dict[str, Any]]
            The registered classes or field, or None if not found.
        """
        if field is None:
            return self._registry.get(name)
        return self._registry.get(name, {}).get(field)


def import_submodules(package_name):
    """Helper function to import all submodules of a package."""
    package = sys.modules[package_name]
    for _, name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if is_pkg:
            import_submodules(name)
        else:
            importlib.import_module(name)
