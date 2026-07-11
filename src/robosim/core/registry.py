"""Decorator-based auto-registration for scenes, robots, and tasks.

Adding a new scene/robot/task is: write one file with a `@register_x("name")` class.
Nothing here changes, and nothing in core/world.py changes either.
"""

_SCENES: dict[str, type] = {}
_ROBOTS: dict[str, type] = {}
_TASKS: dict[str, type] = {}


def register_scene(name: str):
    def deco(cls):
        if name in _SCENES:
            raise ValueError(f"scene '{name}' already registered")
        cls.name = name
        _SCENES[name] = cls
        return cls

    return deco


def register_robot(name: str):
    def deco(cls):
        if name in _ROBOTS:
            raise ValueError(f"robot '{name}' already registered")
        cls.name = name
        _ROBOTS[name] = cls
        return cls

    return deco


def register_task(name: str):
    def deco(cls):
        if name in _TASKS:
            raise ValueError(f"task '{name}' already registered")
        cls.name = name
        _TASKS[name] = cls
        return cls

    return deco


def get_scene(name: str) -> type:
    _ensure_loaded()
    if name not in _SCENES:
        raise KeyError(f"unknown scene '{name}'; available: {sorted(_SCENES)}")
    return _SCENES[name]


def get_robot(name: str) -> type:
    _ensure_loaded()
    if name not in _ROBOTS:
        raise KeyError(f"unknown robot '{name}'; available: {sorted(_ROBOTS)}")
    return _ROBOTS[name]


def get_task(name: str) -> type:
    _ensure_loaded()
    if name not in _TASKS:
        raise KeyError(f"unknown task '{name}'; available: {sorted(_TASKS)}")
    return _TASKS[name]


def list_scenes() -> list[str]:
    _ensure_loaded()
    return sorted(_SCENES)


def list_robots() -> list[str]:
    _ensure_loaded()
    return sorted(_ROBOTS)


def list_tasks() -> list[str]:
    _ensure_loaded()
    return sorted(_TASKS)


_loaded = False


def _ensure_loaded() -> None:
    """Import the built-in scenes/robots/tasks packages so their decorators run.

    Plugin files only need to exist under scenes/robots/envs.tasks to self-register;
    this is the one place that knows those packages exist, so it walks and imports
    every module in them rather than hardcoding a list of filenames.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    import importlib
    import pkgutil

    import robosim.scenes as scenes_pkg
    import robosim.robots as robots_pkg
    import robosim.envs.tasks as tasks_pkg

    for pkg in (scenes_pkg, robots_pkg, tasks_pkg):
        for module_info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            if module_info.name.endswith(".base"):
                continue
            importlib.import_module(module_info.name)
