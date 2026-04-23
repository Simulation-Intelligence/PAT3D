__all__ = [
    "FirstContractSlicePipeline",
    "PaperCorePipeline",
    "build_paper_core_pipeline",
    "build_prompt_backed_contract_slice_pipeline",
]


def __getattr__(name: str):
    if name in {"build_paper_core_pipeline", "build_prompt_backed_contract_slice_pipeline"}:
        from .factories import (
            build_paper_core_pipeline,
            build_prompt_backed_contract_slice_pipeline,
        )

        mapping = {
            "build_paper_core_pipeline": build_paper_core_pipeline,
            "build_prompt_backed_contract_slice_pipeline": build_prompt_backed_contract_slice_pipeline,
        }
        value = mapping[name]
        globals()[name] = value
        return value

    if name == "FirstContractSlicePipeline":
        from .first_contract_slice import FirstContractSlicePipeline

        globals()[name] = FirstContractSlicePipeline
        return FirstContractSlicePipeline

    if name == "PaperCorePipeline":
        from .paper_core_pipeline import PaperCorePipeline

        globals()[name] = PaperCorePipeline
        return PaperCorePipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
