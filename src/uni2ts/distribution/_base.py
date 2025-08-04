#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import abc
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
from einops import rearrange
from jaxtyping import Float, PyTree
from torch import nn
from torch.distributions import AffineTransform, Distribution, TransformedDistribution
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from uni2ts.common.core import abstract_class_property
from uni2ts.module.ts_embed import MultiOutSizeLinear


# TODO: Replace with tree_map when multiple trees supported
def tree_map_multi(
    func: Callable, tree: PyTree[Any, "T"], *other: PyTree[Any, "T"]
) -> PyTree[Any, "T"]:
    """
    A tree map function that applies a function to the leaves of multiple PyTrees.

    Args:
        func (Callable): The function to apply to the leaves.
        tree (PyTree[Any, "T"]): The primary PyTree.
        *other (PyTree[Any, "T"]): The other PyTrees.

    Returns:
        PyTree[Any, "T"]: A new PyTree with the function applied to the leaves.
    """
    leaves, treespec = tree_flatten(tree)
    other_leaves = [tree_flatten(o)[0] for o in other]
    return_leaves = [func(*leaf) for leaf in zip(leaves, *other_leaves)]
    return tree_unflatten(return_leaves, treespec)


def convert_to_module(tree: PyTree[nn.Module, "T"]) -> PyTree[nn.Module, "T"]:
    """
    Converts a PyTree of simple containers (dicts, lists, tuples) into a PyTree of
    `nn.Module` containers (`nn.ModuleDict`, `nn.ModuleList`).

    Args:
        tree (PyTree[nn.Module, "T"]): The PyTree to convert.

    Returns:
        PyTree[nn.Module, "T"]: The converted PyTree.
    """
    if isinstance(tree, dict):
        return nn.ModuleDict(
            {key: convert_to_module(child) for key, child in tree.items()}
        )
    if isinstance(tree, (list, tuple)):
        return nn.ModuleList([convert_to_module(child) for child in tree])
    return tree


def convert_to_container(tree: PyTree[nn.Module, "T"]) -> PyTree[nn.Module, "T"]:
    """
    Converts a PyTree of `nn.Module` containers (`nn.ModuleDict`, `nn.ModuleList`)
    into a PyTree of simple containers (dicts, lists, tuples).

    Args:
        tree (PyTree[nn.Module, "T"]): The PyTree to convert.

    Returns:
        PyTree[nn.Module, "T"]: The converted PyTree.
    """
    if isinstance(tree, nn.ModuleDict):
        return {key: convert_to_container(child) for key, child in tree.items()}
    if isinstance(tree, nn.ModuleList):
        return [convert_to_container(child) for child in tree]
    return tree


class DistrParamProj(nn.Module):
    """
    A projection layer that maps a representation to the parameters of a distribution.

    Args:
        in_features (int): The size of the input representation.
        out_features (int | tuple[int, ...] | list[int]): The size multiplier for the
            distribution parameters.
        args_dim (PyTree[int, "T"]): A PyTree specifying the dimensionality of each
            distribution parameter.
        domain_map (PyTree[Callable[[torch.Tensor], torch.Tensor], "T"]): A PyTree of
            functions that map the unbounded distribution parameters to the valid
            domain for each parameter.
        proj_layer (Callable[..., nn.Module], optional): The projection layer to use.
            Defaults to `MultiOutSizeLinear`.
        **kwargs (Any): Additional keyword arguments for the projection layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | tuple[int, ...] | list[int],
        args_dim: PyTree[int, "T"],
        domain_map: PyTree[Callable[[torch.Tensor], torch.Tensor], "T"],
        proj_layer: Callable[..., nn.Module] = MultiOutSizeLinear,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.args_dim = args_dim
        self.domain_map = domain_map

        if isinstance(out_features, int):

            def proj(dim):
                proj_layer(in_features, dim * out_features, **kwargs)

        elif isinstance(out_features, Sequence):

            def proj(dim):
                return proj_layer(
                    in_features,
                    tuple(dim * of for of in out_features),
                    dim=dim,
                    **kwargs,
                )

        else:
            raise ValueError(
                f"out_features must be int or sequence of ints, got invalid type: {type(out_features)}"
            )

        self.proj = convert_to_module(tree_map(proj, args_dim))
        self.out_size = (
            out_features if isinstance(out_features, int) else max(out_features)
        )

    def forward(self, *args) -> PyTree[Float[torch.Tensor, "*batch out dim"], "T"]:
        params_unbounded = tree_map(
            lambda proj: rearrange(
                proj(*args),
                "... (dim out_size) -> ... out_size dim",
                out_size=self.out_size,
            ),
            convert_to_container(self.proj),
        )
        params = tree_map_multi(
            lambda func, inp: func(inp), self.domain_map, params_unbounded
        )
        return params


class AffineTransformed(TransformedDistribution):
    """
    A distribution that applies an affine transformation to a base distribution.

    Args:
        base_dist (Distribution): The base distribution.
        loc (Optional[torch.Tensor | float], optional): The location parameter.
            Defaults to None.
        scale (Optional[torch.Tensor | float], optional): The scale parameter.
            Defaults to None.
        validate_args (Optional[bool], optional): Whether to validate the arguments.
            Defaults to None.
    """
    def __init__(
        self,
        base_dist: Distribution,
        loc: Optional[torch.Tensor | float] = None,
        scale: Optional[torch.Tensor | float] = None,
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc if loc is not None else 0.0
        self.scale = scale if scale is not None else 1.0
        super().__init__(
            base_dist,
            [AffineTransform(loc=self.loc, scale=self.scale)],
            validate_args=validate_args,
        )

    @property
    def mean(self) -> torch.Tensor:
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self) -> torch.Tensor:
        return self.base_dist.variance * self.scale**2


@abstract_class_property("distr_cls")
class DistributionOutput:
    """
    An abstract base class for distribution outputs. It defines the type of output
    distribution and provides helper methods for creating predictive distributions.
    """

    distr_cls: type[Distribution] = NotImplemented

    def distribution(
        self,
        distr_params: PyTree[torch.Tensor, "T"],
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> Distribution:
        """
        Creates a distribution from the given parameters.

        Args:
            distr_params (PyTree[torch.Tensor, "T"]): The parameters of the distribution.
            loc (Optional[torch.Tensor], optional): The location parameter for an
                affine transformation. Defaults to None.
            scale (Optional[torch.Tensor], optional): The scale parameter for an
                affine transformation. Defaults to None.
            validate_args (Optional[bool], optional): Whether to validate the
                arguments. Defaults to None.

        Returns:
            Distribution: The created distribution.
        """
        distr = self._distribution(distr_params, validate_args=validate_args)
        if loc is not None or scale is not None:
            distr = AffineTransformed(distr, loc=loc, scale=scale)
        return distr

    def _distribution(
        self,
        distr_params: PyTree[torch.Tensor, "T"],
        validate_args: Optional[bool] = None,
    ) -> Distribution:
        """
        Creates the base distribution from the given parameters.
        """
        return self.distr_cls(**distr_params, validate_args=validate_args)

    @property
    @abc.abstractmethod
    def args_dim(self) -> PyTree[int, "T"]:
        """
        Returns a PyTree specifying the dimensionality of each distribution parameter.
        """
        ...

    @property
    @abc.abstractmethod
    def domain_map(self) -> PyTree[Callable[[torch.Tensor], torch.Tensor], "T"]:
        """
        Returns a PyTree of functions that map the unbounded distribution parameters
        to the valid domain for each parameter.
        """
        ...

    def get_param_proj(
        self,
        in_features: int,
        out_features: int | tuple[int, ...] | list[int],
        proj_layer: Callable[..., nn.Module] = MultiOutSizeLinear,
        **kwargs: Any,
    ) -> nn.Module:
        """
        Returns a projection layer that maps a representation to the parameters of
        the distribution.

        Args:
            in_features (int): The size of the input representation.
            out_features (int | tuple[int, ...] | list[int]): The size multiplier
                for the distribution parameters.
            proj_layer (Callable[..., nn.Module], optional): The projection layer
                to use. Defaults to `MultiOutSizeLinear`.
            **kwargs (Any): Additional keyword arguments for the projection layer.

        Returns:
            nn.Module: The projection layer.
        """
        return DistrParamProj(
            in_features=in_features,
            out_features=out_features,
            args_dim=self.args_dim,
            domain_map=self.domain_map,
            proj_layer=proj_layer,
            **kwargs,
        )
