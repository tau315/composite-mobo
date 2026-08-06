"""Forward-only reimplementation of Summit's Reizman-Suzuki emulator.

Summit (https://github.com/sustainable-processes/summit, MIT) ships the trained
neural-network ensemble that emulates the Reizman et al. Suzuki-Miyaura
screening experiments. The package itself pins ``python = ">=3.8,<3.11"`` and an
old torch/pytorch-lightning stack, so it cannot be installed alongside this
project. Only inference is needed here, so the five predictor state dicts and
the JSON descriptor are vendored under ``data/`` and the forward pass is
reproduced directly.

Reproduced from ``summit/benchmarks/experimental_emulator.py``:

* features are ``ColumnTransformer([("num", StandardScaler(), continuous),
  ("cat", OneHotEncoder(categories=domain_levels), categorical)])``, so the
  three standardized continuous columns come first and the eight one-hot
  catalyst columns follow **in the domain's own level order**, which is not
  alphabetical;
* each predictor is ``output_layer(relu(input_layer(x)))`` with 512 hidden units;
* every predictor carries its own input and output ``StandardScaler`` statistics
  because they were fitted on different cross-validation folds;
* predictions are averaged across the ensemble and clipped to the domain bounds.

Source: Reizman, Wang, Buchwald & Jensen, "Suzuki-Miyaura cross-coupling
optimization enabled by automated feedback", React. Chem. Eng. 1, 658-666
(2016). Emulator weights: Felton, Rittig & Lapkin, Summit.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

Tensor = torch.Tensor

DATA_DIR = Path(__file__).parent / "data"
CONTINUOUS_NAMES = ("t_res", "temperature", "catalyst_loading")
N_PREDICTORS = 5


class ReizmanEmulator:
    """The Summit ensemble for one Reizman-Suzuki case, inference only."""

    def __init__(self, case: int = 4, catalyst: str = "P1-L2") -> None:
        directory = DATA_DIR / f"reizman_suzuki_case_{case}"
        descriptor = json.loads(
            (directory / f"reizman_suzuki_case_{case}.json").read_text(encoding="utf-8")
        )
        params = descriptor["experiment_params"]
        if params["output_variable_names"] != ["ton", "yld"]:
            raise ValueError("unexpected output ordering in the vendored descriptor")

        inputs = {v["name"]: v for v in descriptor["domain"] if not v["is_objective"]}
        self.levels: list[str] = inputs["catalyst"]["levels"]
        if catalyst not in self.levels:
            raise ValueError(f"unknown catalyst {catalyst!r}; expected {self.levels}")
        self.catalyst = catalyst
        self.catalyst_index = self.levels.index(catalyst)
        self.bounds = torch.tensor(
            [inputs[name]["bounds"] for name in CONTINUOUS_NAMES], dtype=torch.double
        ).T.contiguous()

        objectives = {v["name"]: v for v in descriptor["domain"] if v["is_objective"]}
        self.output_bounds = torch.tensor(
            [objectives["ton"]["bounds"], objectives["yld"]["bounds"]],
            dtype=torch.double,
        ).T.contiguous()
        self.clip = bool(params.get("clip", True))

        self.weights = []
        for index in range(N_PREDICTORS):
            state = torch.load(
                directory / f"reizman_suzuki_case_{case}_predictor_{index}.pt",
                map_location="cpu",
                weights_only=True,
            )
            preprocessor = params["predictors"][index]
            self.weights.append(
                {
                    "w1": state["input_layer.weight"].double(),
                    "b1": state["input_layer.bias"].double(),
                    "w2": state["output_layer.weight"].double(),
                    "b2": state["output_layer.bias"].double(),
                    "x_mean": torch.tensor(
                        preprocessor["input_preprocessor"]["num"]["mean_"],
                        dtype=torch.double,
                    ),
                    "x_scale": torch.tensor(
                        preprocessor["input_preprocessor"]["num"]["scale_"],
                        dtype=torch.double,
                    ),
                    "y_mean": torch.tensor(
                        preprocessor["output_preprocessor"]["mean_"], dtype=torch.double
                    ),
                    "y_scale": torch.tensor(
                        preprocessor["output_preprocessor"]["scale_"],
                        dtype=torch.double,
                    ),
                }
            )

    def physical(self, X: Tensor) -> Tensor:
        """Map the unit cube to residence time, temperature, catalyst loading."""

        lower, upper = self.bounds[0], self.bounds[1]
        return lower + X.double() * (upper - lower)

    def predict(self, X: Tensor) -> Tensor:
        """Return ensemble-mean ``(ton, yld)`` for unit-cube designs."""

        continuous = self.physical(X)
        one_hot = torch.zeros(
            *continuous.shape[:-1], len(self.levels), dtype=torch.double
        )
        one_hot[..., self.catalyst_index] = 1.0

        total = torch.zeros(*continuous.shape[:-1], 2, dtype=torch.double)
        for predictor in self.weights:
            scaled = (continuous - predictor["x_mean"]) / predictor["x_scale"]
            features = torch.cat((scaled, one_hot), dim=-1)
            hidden = torch.relu(features @ predictor["w1"].T + predictor["b1"])
            standardized = hidden @ predictor["w2"].T + predictor["b2"]
            member = standardized * predictor["y_scale"] + predictor["y_mean"]
            # Summit clips each predictor before averaging, not the mean. The
            # order matters whenever a member leaves the domain bounds: on
            # P1-L2 that shifts the ensemble yield by up to 0.22 percentage
            # points, and by 5.3 on other catalysts.
            if self.clip:
                member = member.clamp(self.output_bounds[0], self.output_bounds[1])
            total = total + member
        return total / len(self.weights)

    def yield_percent(self, X: Tensor) -> Tensor:
        """Predicted reaction yield in percent, the one simulated quantity."""

        return self.predict(X)[..., 1]
