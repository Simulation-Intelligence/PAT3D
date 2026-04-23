def main(*args, **kwargs):
    from pat3d.cli.run_pipeline import main as _impl

    return _impl(*args, **kwargs)


def run_preprocess_main(*args, **kwargs):
    from pat3d.cli.run_preprocess import main as _impl

    return _impl(*args, **kwargs)

__all__ = ["main", "run_preprocess_main"]
