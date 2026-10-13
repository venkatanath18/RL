# Phase 1 benchmark results

Run date: 13 October 2026. Commands were run from the repository root with
the supplied source files.

## Python policy iteration

The Python run reports the following optimal policy and values:

| States | Action | Value range |
|---|---|---:|
| 1–13 | CONTINUE (`1`) | 1106.6478–1942.6682 |
| 14–15 | RESET (`0`) | 1982.2688 |

Selected values are:

```text
s= 1  V=1106.6478  continue
s= 7  V=1538.4991  continue
s=13  V=1942.6682  continue
s=14  V=1982.2688  reset
s=15  V=1982.2688  reset
```

## C++ policy/value iteration

The C++ program prints:

```text
optimal policy
1 1 1 1 1 1 1 1 1 1 1 1 1 0 0

optimal values
1106.65 1182.89 1256.75 1328.86 1399.68 1469.48 1538.5
1606.89 1674.76 1742.21 1809.31 1876.12 1942.67 1982.27 1982.27
```

The policy matches Python exactly. Values differ only because the C++ output
uses two decimal places.

## Transition-order diagnostic

`first1.cpp` was compiled and run. It reported:

```text
Found violation:
i1 = 1, i2 = 7, k = 1
temp1 = 1.000000000000000000000000000000
temp2 = 0.999999999999999888977697537484
Difference = 0.000000000000000111022302462516
```

This is approximately `1.11e-16`, far below the scale of the probabilities,
and is a floating-point precision artifact relative to the `1e-18` tolerance.
It should be reported as a diagnostic precision finding rather than as a
counterexample to the intended transition ordering.

## Verification

Both main programs completed successfully. The Python and C++ policies agree
for all 15 states, giving the Phase 1 benchmark replacement region `14–15`.
