# recordstore

A tiny in-memory record store with offset pagination.

```python
from src import Query

page = Query(rows).where(status="active").order_by("name").page(2, per_page=20)
print(page.items, page.has_next)
```

Pages are 1-indexed. An empty collection still has one (blank) page, so the UI can
always render "page 1 of 1" instead of "page 0 of 0".

Run the smoke tests with `pytest smoke/`.
