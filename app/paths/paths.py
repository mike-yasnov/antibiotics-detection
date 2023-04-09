import os
from pathlib import Path


def get_project_path() -> str:
    """
    :return:
    project path
    """
    return Path(__file__).parent.parent.parent


def get_models_path() -> str:
    """
    :return:
    models path
    """
    path_to_model = os.path.join(
        Path(__file__).parent.parent,
        'pipeline_storage'
        )
    return path_to_model


PATH_TO_UI_FILE = os.path.join(get_project_path(), 'app', 'GUI', 'GUI.ui')
PATH_TO_PROJECT = get_project_path()
PATH_TO_PIPELINE = get_models_path()
