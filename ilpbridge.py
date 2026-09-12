"""Minimal CP-SAT-compatible model layer backed by the exact Z3 bridge.

Only the subset of the OR-Tools CP-SAT API used by this project is
implemented. Every constraint is translated into the JSON spec consumed by
``topoflow_native.solve_ilp_exact``, so the physical-layout search shares the
single Z3 engine embedded in the native extension.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import topoflow_native

# Mirror OR-Tools CP-SAT status constants.
UNKNOWN = 0
MODEL_INVALID = 1
FEASIBLE = 2
INFEASIBLE = 3
OPTIMAL = 4


class InfeasibleError(RuntimeError):
    """The model has no feasible solution."""


class TimeoutError(RuntimeError):
    """The solver stopped before finding a solution."""


class BridgeError(RuntimeError):
    """The bridge rejected the model specification."""


def _solve_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(topoflow_native.solve_ilp_exact(json.dumps(spec)))
    status = result["status"]
    if status == "error":
        raise BridgeError(result.get("error"))
    if status == "infeasible":
        raise InfeasibleError()
    if status == "timeout":
        raise TimeoutError()
    return result


class _Condition:
    """A boolean variable or its negation, used in only_enforce_if lists."""

    def __init__(self, var: "_Var", negated: bool = False):
        self.var = var
        self.negated = negated

    def Not(self) -> "_Condition":
        return _Condition(self.var, not self.negated)

    def token(self) -> str:
        return f"!{self.var.name}" if self.negated else self.var.name


class _Var:
    """An integer or boolean decision variable."""

    def __init__(self, model: "Model", name: str, lo: int, hi: int, kind: str):
        self._model = model
        self.name = name
        self.lo = lo
        self.hi = hi
        self.kind = kind
        model._vars[name] = self

    __hash__ = object.__hash__

    def _expr(self) -> "_LinearExpr":
        return _LinearExpr(self._model).add_var(self, 1)

    def __add__(self, other: Any) -> "_LinearExpr":
        return self._expr().add(other, 1)

    def __radd__(self, other: Any) -> "_LinearExpr":
        return self._expr().add(other, 1)

    def __sub__(self, other: Any) -> "_LinearExpr":
        return self._expr().add(other, -1)

    def __rsub__(self, other: Any) -> "_LinearExpr":
        return self._expr().negate().add(other, 1)

    def __mul__(self, factor: Any) -> "_LinearExpr":
        if not isinstance(factor, int):
            raise TypeError("only multiplication by an integer constant is supported")
        return self._expr().mul(factor)

    def __rmul__(self, factor: Any) -> "_LinearExpr":
        return self.__mul__(factor)

    def __neg__(self) -> "_LinearExpr":
        return self._expr().negate()

    def __eq__(self, other: Any) -> "_Constraint":  # type: ignore[override]
        return self._expr().__eq__(other)

    def __le__(self, other: Any) -> "_Constraint":
        return self._expr().__le__(other)

    def __ge__(self, other: Any) -> "_Constraint":
        return self._expr().__ge__(other)

    def __lt__(self, other: Any) -> "_Constraint":
        return self._expr().__lt__(other)

    def __gt__(self, other: Any) -> "_Constraint":
        return self._expr().__gt__(other)

    def __ne__(self, other: Any) -> "_Constraint":  # type: ignore[override]
        return self._expr().__ne__(other)

    def Not(self) -> _Condition:
        if self.kind != "bool":
            raise TypeError("Not() is only defined for boolean variables")
        return _Condition(self, negated=True)


class _LinearExpr:
    """A weighted sum of decision variables plus an integer constant."""

    def __init__(self, model: "Model"):
        self.model = model
        self.terms: dict[str, int] = {}
        self.constant = 0

    def add_var(self, var: _Var, coefficient: int) -> "_LinearExpr":
        self.terms[var.name] = self.terms.get(var.name, 0) + coefficient
        return self

    def add(self, other: Any, sign: int) -> "_LinearExpr":
        if isinstance(other, _Var):
            return self.add_var(other, sign)
        if isinstance(other, _LinearExpr):
            for name, coefficient in other.terms.items():
                self.terms[name] = self.terms.get(name, 0) + sign * coefficient
            self.constant += sign * other.constant
            return self
        if isinstance(other, int):
            self.constant += sign * other
            return self
        raise TypeError(f"unsupported linear expression operand: {other!r}")

    def mul(self, factor: int) -> "_LinearExpr":
        for name in list(self.terms):
            self.terms[name] *= factor
        self.constant *= factor
        return self

    def negate(self) -> "_LinearExpr":
        return self.mul(-1)

    def __add__(self, other: Any) -> "_LinearExpr":
        return _LinearExpr(self.model).add(self, 1).add(other, 1)

    def __radd__(self, other: Any) -> "_LinearExpr":
        return _LinearExpr(self.model).add(other, 1).add(self, 1)

    def __sub__(self, other: Any) -> "_LinearExpr":
        return _LinearExpr(self.model).add(self, 1).add(other, -1)

    def __rsub__(self, other: Any) -> "_LinearExpr":
        return _LinearExpr(self.model).add(other, 1).add(self, -1)

    def __mul__(self, factor: Any) -> "_LinearExpr":
        if not isinstance(factor, int):
            raise TypeError("only multiplication by an integer constant is supported")
        return _LinearExpr(self.model).add(self, 1).mul(factor)

    def __rmul__(self, factor: Any) -> "_LinearExpr":
        return self.__mul__(factor)

    def __neg__(self) -> "_LinearExpr":
        return _LinearExpr(self.model).add(self, -1)

    def _comparison(self, other: Any, op: str) -> "_Constraint":
        normalized = _LinearExpr(self.model).add(self, 1).add(other, -1)
        return _Constraint(self.model, normalized, op)

    def __eq__(self, other: Any) -> "_Constraint":  # type: ignore[override]
        return self._comparison(other, "eq")

    def __le__(self, other: Any) -> "_Constraint":
        return self._comparison(other, "le")

    def __ge__(self, other: Any) -> "_Constraint":
        return self._comparison(other, "ge")

    def __lt__(self, other: Any) -> "_Constraint":
        # a < b  <=>  a + 1 <= b  for integer variables.
        normalized = _LinearExpr(self.model).add(self, 1).add(other, -1)
        normalized.constant += 1
        return _Constraint(self.model, normalized, "le")

    def __gt__(self, other: Any) -> "_Constraint":
        normalized = _LinearExpr(self.model).add(self, 1).add(other, -1)
        normalized.constant -= 1
        return _Constraint(self.model, normalized, "ge")

    def __ne__(self, other: Any) -> "_Constraint":  # type: ignore[override]
        normalized = _LinearExpr(self.model).add(self, 1).add(other, -1)
        return _Constraint(self.model, normalized, "ne")


class _Constraint:
    """A pending constraint; ``only_enforce_if`` makes it an implication."""

    def __init__(self, model: "Model", expr: _LinearExpr, op: str):
        self.model = model
        self.terms = dict(expr.terms)
        self.constant = expr.constant
        self.op = op
        self.only_if: list[str] = []
        model._constraints.append(self)

    def only_enforce_if(self, conditions: Any) -> "_Constraint":
        if isinstance(conditions, (list, tuple)):
            for condition in conditions:
                if isinstance(condition, _Condition):
                    self.only_if.append(condition.token())
                elif isinstance(condition, _Var) and condition.kind == "bool":
                    self.only_if.append(condition.name)
                else:
                    raise TypeError(f"unsupported enforcement condition: {condition!r}")
        elif isinstance(conditions, _Condition):
            self.only_if.append(conditions.token())
        elif isinstance(conditions, _Var) and conditions.kind == "bool":
            self.only_if.append(conditions.name)
        else:
            raise TypeError(f"unsupported enforcement condition: {conditions!r}")
        return self


class Model:
    """Accumulates variables and constraints; ``solve`` executes the spec."""

    def __init__(self) -> None:
        self._vars: dict[str, _Var] = {}
        self._constraints: list[_Constraint] = []
        self._all_different: list[list[str]] = []
        self._exactly_one: list[list[str]] = []
        self._tables: list[dict[str, Any]] = []
        self._hints: list[list[Any]] = []
        self._objective: dict[str, Any] | None = None
        self._aux_count = 0

    def new_int_var(self, lo: int, hi: int, name: str) -> _Var:
        return _Var(self, name, lo, hi, "int")

    def new_real_var(self, lo: float, hi: float, name: str) -> _Var:
        return _Var(self, name, int(lo), int(hi), "real")

    def new_bool_var(self, name: str) -> _Var:
        return _Var(self, name, 0, 1, "bool")

    def add(self, constraint: Any) -> Any:
        if not isinstance(constraint, _Constraint):
            raise TypeError("add() expects a comparison expression")
        return constraint

    def add_abs_equality(self, delta: _Var, expr: _LinearExpr) -> None:
        # delta == |expr|: a fresh boolean selects which side of the absolute
        # value the expression falls on.
        self._aux_count += 1
        selector = self.new_bool_var(f"_abs_sel_{self._aux_count}")
        positive = _LinearExpr(self).add(delta, 1).add(expr, -1)
        negative = _LinearExpr(self).add(delta, 1).add(expr, 1)
        _Constraint(self, positive, "eq").only_enforce_if(selector)
        _Constraint(self, negative, "eq").only_enforce_if(selector.Not())

    def add_all_different(self, variables: Sequence[_Var]) -> None:
        self._all_different.append([variable.name for variable in variables])

    def add_exactly_one(self, variables: Sequence[_Var]) -> None:
        self._exactly_one.append([variable.name for variable in variables])

    def add_allowed_assignments(
        self, variables: Sequence[_Var], tuples: Sequence[Sequence[int]]
    ) -> None:
        self._tables.append(
            {
                "vars": [variable.name for variable in variables],
                "tuples": [list(row) for row in tuples],
            }
        )

    def add_bool_or(self, conditions: Sequence[Any]) -> None:
        expression = _LinearExpr(self)
        for condition in conditions:
            if isinstance(condition, _Condition):
                if condition.negated:
                    expression.constant += 1
                    expression.add_var(condition.var, -1)
                else:
                    expression.add_var(condition.var, 1)
            elif isinstance(condition, _Var) and condition.kind == "bool":
                expression.add_var(condition, 1)
            else:
                raise TypeError(f"unsupported boolean condition: {condition!r}")
        expression.constant -= 1
        _Constraint(self, expression, "ge")

    def add_multiplication_equality(
        self, target: _Var, factors: Sequence[_Var]
    ) -> None:
        if len(factors) != 2:
            raise ValueError("multiplication equality supports exactly two factors")
        left, right = factors
        tuples = [
            [row * column, row, column]
            for row in range(left.lo, left.hi + 1)
            for column in range(right.lo, right.hi + 1)
        ]
        self.add_allowed_assignments([target, left, right], tuples)

    def add_hint(self, variable: _Var, value: int) -> None:
        self._hints.append([variable.name, value])

    def minimize(self, expr: _LinearExpr) -> None:
        self._objective = {"dir": "min", "terms": _terms(expr)}

    def maximize(self, expr: _LinearExpr) -> None:
        self._objective = {"dir": "max", "terms": _terms(expr)}

    def spec(self, timeout_ms: int, workers: int = 16) -> dict[str, Any]:
        constraints = []
        auxiliary_count = 0
        for constraint in self._constraints:
            if constraint.op == "ne":
                # lhs != 0 <=> (lhs <= -1) or (lhs >= 1); a fresh boolean
                # selects which side holds.
                auxiliary = self.new_bool_var(f"_ne_{auxiliary_count}")
                auxiliary_count += 1
                positive: dict[str, int] = dict(constraint.terms)
                negative = {name: -coefficient for name, coefficient in positive.items()}
                constraints.append(
                    _linear_constraint(
                        positive, "le", -1 - constraint.constant,
                        constraint.only_if + [auxiliary.name],
                    )
                )
                constraints.append(
                    _linear_constraint(
                        negative, "le", -1 + constraint.constant,
                        constraint.only_if + [f"!{auxiliary.name}"],
                    )
                )
                continue
            constraints.append(
                _linear_constraint(
                    constraint.terms,
                    constraint.op,
                    -constraint.constant,
                    constraint.only_if,
                )
            )
        variables = [
            {
                "name": variable.name,
                "kind": variable.kind,
                "lo": str(variable.lo),
                "hi": str(variable.hi),
            }
            for variable in self._vars.values()
        ]
        spec: dict[str, Any] = {
            "vars": variables,
            "constraints": constraints,
            "all_different": self._all_different,
            "exactly_one": self._exactly_one,
            "table": self._tables,
            "hints": self._hints,
            "timeout_ms": timeout_ms,
            "workers": workers,
        }
        if self._objective is not None:
            spec["objective"] = self._objective
        return spec


def _terms(expr: _LinearExpr) -> list[list[Any]]:
    return [[str(coefficient), name] for name, coefficient in expr.terms.items()]


def _linear_constraint(
    terms: Mapping[str, int], op: str, rhs: int, only_if: Sequence[str]
) -> dict[str, Any]:
    return {
        "terms": [[str(coefficient), name] for name, coefficient in terms.items()],
        "op": op,
        "rhs": str(rhs),
        "only_if": list(only_if),
    }


class Parameters:
    """Accepted CP-SAT tuning knobs; the bridge honors time limits and worker count."""

    def __init__(self) -> None:
        self.max_time_in_seconds = 300.0
        self.num_search_workers = 16
        self.relative_gap_limit: float | None = None
        self.log_search_progress = False
        self.stop_after_first_solution = False


class Solver:
    """Runs a :class:`Model` through the exact bridge and reads values back."""

    def __init__(self) -> None:
        self.parameters = Parameters()
        self._result: dict[str, Any] | None = None

    def solve(self, model: Model, callback: Any = None) -> int:
        del callback  # progress reporting is not available through the bridge
        timeout_ms = max(1, int(self.parameters.max_time_in_seconds * 1000))
        try:
            self._result = _solve_spec(
                model.spec(timeout_ms, self.parameters.num_search_workers)
            )
        except InfeasibleError:
            return INFEASIBLE
        except TimeoutError:
            return UNKNOWN
        return OPTIMAL if self._result["status"] == "optimal" else FEASIBLE

    def _value(self, variable: _Var) -> str | None:
        if self._result is None:
            return None
        return self._result["values"].get(variable.name)

    def value(self, variable: Any) -> Any:
        raw = self._value(variable)
        if raw is None:
            raise RuntimeError(f"variable {variable.name} has no assigned value")
        if "/" not in raw:
            return int(raw)
        numerator, denominator = raw.split("/")
        numerator_int = int(numerator)
        denominator_int = int(denominator)
        quotient, remainder = divmod(numerator_int, denominator_int)
        if remainder == 0:
            return quotient
        return numerator_int / denominator_int

    def int_value(self, variable: Any) -> int:
        raw = self._value(variable)
        if raw is None:
            raise RuntimeError(f"variable {variable.name} has no assigned value")
        return int(float(raw))

    @property
    def objective_value(self) -> float:
        if self._result is None or self._result.get("objective") is None:
            return 0.0
        raw = self._result["objective"]
        if "/" in raw:
            numerator, denominator = raw.split("/")
            return int(numerator) / int(denominator)
        return float(raw)


CpModel = Model
CpSolver = Solver

