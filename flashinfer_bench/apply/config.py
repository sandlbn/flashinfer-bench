"""Configuration classes for apply runtime behavior."""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class ApplyConfig(BaseModel):
    """Configuration for apply runtime behavior.

    Controls error tolerances, AOT compilation strategy, and miss handling policy.
    """

    max_atol: float = Field(default=1e-2, gt=0)
    """The maximum absolute difference allowed between the reference and the candidate."""
    max_rtol: float = Field(default=1e-5, gt=0)
    """The maximum relative difference allowed between the reference and the candidate."""
    aot_ratio: float = Field(default=1.0, ge=0, le=1)
    """The ratio of the top solutions to AOT build for each definition."""
    on_miss_policy: Literal["fallback_only", "use_def_best"] = "fallback_only"
    """The policy when a runtime ApplyKey misses the table."""
    min_gain_us: float = Field(default=0.0, ge=0)
    """Microseconds a solution must save over the provider baseline to be worth selecting.

    Substituting a kernel is not free: resolving the definition, building the key, checking
    dtypes and invoking the Runnable cost real time on every call, and that cost does not
    shrink with the kernel. On a part where it is comparable to an elementwise kernel's
    whole runtime, a solution can be several times faster than the provider's and still
    lose the exchange. The cost is a property of the part and the stack, so it is not
    written down anywhere: ``flashinfer_bench.device.calibration.get().dispatch_us``
    measures it once per part and caches it (``scripts/calibrate_part.py`` prints it).

    When this is above zero, a key is indexed only if the best solution beats the best
    provider baseline (``vllm-xpu``, ``sgl-kernel-xpu``) by more than this margin on the
    hardware the table is built for. Keys with no provider baseline on this part are
    **not** indexed either: with the gate on, the burden of proof is on the substitution,
    and no comparator means no proof. Generate the baseline (``add-baselines``) to have
    them judged.

    Zero disables the check, which is the default so that no existing deployment changes
    behaviour on upgrade. Set it from the calibration, not from a number read elsewhere.
    """


class ApplyConfigRegistry(BaseModel):
    """Per-definition apply configuration registry.

    If default is not set, only kernels explicitly registered in per_definition are applied.
    Set default to an ApplyConfig to apply all kernels.
    """

    default: Optional[ApplyConfig] = None
    """Fallback config for definitions not in per_definition. None means skip."""
    per_definition: Dict[str, ApplyConfig] = Field(default_factory=dict)
    """Mapping from definition name to its specific ApplyConfig."""

    def get(self, def_name: str) -> Optional[ApplyConfig]:
        """Get config for a definition, falling back to default if not registered.

        Parameters
        ----------
        def_name : str
            The definition name to look up.

        Returns
        -------
        Optional[ApplyConfig]
            The config for this definition. If the definition shouldn't be applied, return None.
        """
        return self.per_definition.get(def_name, self.default)

    def register(
        self, def_name: str, config: ApplyConfig, *, override: bool = False
    ) -> ApplyConfigRegistry:
        """Register config for a single definition.

        Parameters
        ----------
        def_name : str
            The definition name.
        config : ApplyConfig
            The config to use.
        override : bool
            If the existing config will be overridden. If False, raise an error if
            the definition already exists. Default: False.

        Returns
        -------
        ApplyConfigRegistry
            Self, for method chaining.

        Raises
        ------
        ValueError
            If def_name already exists and override is False.
        """
        if not override and def_name in self.per_definition:
            raise ValueError(
                f"Definition '{def_name}' already exists. Use override=True to replace it."
            )
        self.per_definition[def_name] = config
        return self

    def register_many(
        self, configs: Dict[str, ApplyConfig], *, override: bool = False
    ) -> "ApplyConfigRegistry":
        """Register configs for multiple definitions.

        Parameters
        ----------
        configs : Dict[str, ApplyConfig]
            Mapping from definition names to their configs.
        override : bool
            If the existing configs will be overridden. If False, raise an error if
            any key already exists. Default: False.

        Returns
        -------
        ApplyConfigRegistry
            Self, for method chaining.

        Raises
        ------
        ValueError
            If any key already exists and override is False.
        """
        if not override:
            conflicts = self.per_definition.keys() & configs.keys()
            if conflicts:
                raise ValueError(
                    "Cannot register. The following definitions already exist: "
                    f"{list(conflicts)}"
                )
        self.per_definition.update(configs)
        return self
