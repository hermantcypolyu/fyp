# tenacity (retry conditions)

A reduced extract of [tenacity](https://github.com/jd/tenacity), covering only the
*retry conditions* - the predicates that decide whether a failed or unsatisfactory call
should be attempted again.

## The pieces

* `tenacity/retry.py` - the strategies. `retry_base` is the abstract base every strategy
  derives from; `retry_if_result`, `retry_if_exception_type` and friends are the concrete
  ones; `retry_any` and `retry_all` combine several strategies with OR and AND semantics.
* `tenacity/retry_state.py` - the objects a strategy is handed at call time. A
  `RetryCallState` carries the attempt number and the `outcome`, a `Future` that either
  holds a returned value or the exception that was raised.

## Composing conditions

A strategy is any callable taking a `RetryCallState` and returning a bool, so a plain
function is a perfectly good condition:

```python
def gave_up_early(retry_state):
    return retry_state.attempt_number < 3
```

`retry_base` overloads `|` and `&` so conditions read like boolean expressions, and the
documented promise is that either operand may be a plain callable:

```python
condition = retry_if_exception_type(IOError) | gave_up_early
condition = retry_if_result(lambda r: r is None) & gave_up_early
```

Both forms build the same objects you would get from writing `retry_any(...)` or
`retry_all(...)` by hand, with the operands in left-to-right order.
