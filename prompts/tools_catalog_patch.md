# Changes to `mcp_server/tools_catalog.py` (your file; apply by hand)

The new gateway code expects three small changes here.

## 1. `search_products`: make `q` optional and fix the price units

```python
"search_products": {
    "type": "object",
    "properties": {
        "q": {"type": "string", "description": "1 to 4 plain keywords, for example 'mechanical keyboard'. Omit for browsing ('what's popular')."},
        "category_id": {"type": "string", "description": "Restrict results to a single category id."},
        "merchant_id": {"type": "string", "description": "Restrict results to a single merchant/store."},
        "price_min": {"type": "number", "description": "Lower price bound in NAIRA (major units). 50000 means fifty thousand naira. Never kobo."},
        "price_max": {"type": "number", "description": "Upper price bound in NAIRA (major units)."},
        "currency": {"type": "string", "description": "ISO currency code for the price bounds.", "default": "NGN"},
        "limit": {"type": "integer", "description": "Maximum products to return (default 20, max 48).", "default": 20},
    },
    "required": [],
},
```
(`skip` is removed: the gateway pages through the catalog itself.)

## 2. `add_to_cart`: product_id is enough

```python
"add_to_cart": {
    "type": "object",
    "properties": {
        "product_id": {"type": "string", "description": "The product id. The variant, price and stock are looked up from the catalog."},
        "variant_id": {"type": "string", "description": "Optional. The variant to add; defaults to the product's first variant."},
        "quantity": {"type": "integer", "description": "Units to add.", "default": 1},
    },
    "required": [],
},
```
Remove `product_name`, `variant_name`, `sku`, `unit_price_amount`, `unit_price_currency` and `quantity_available`: the model must not supply prices or stock. The gateway ignores them if sent.

## 3. New tool `update_cart_items`

```python
UPDATE_CART_ITEMS = ToolMeta(
    name="update_cart_items",
    method="SESSION",
    path="",
    risk_tier=RiskTier.LOW,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="Set absolute quantities for cart lines. Quantity 0 removes the line.",
)
```
Add `UPDATE_CART_ITEMS` to `ALL_TOOLS`, and to `TOOL_PARAMETERS`:

```python
"update_cart_items": {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Lines to change.",
            "items": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string"},
                    "quantity": {"type": "integer", "description": "New quantity. 0 removes the line."},
                },
                "required": ["variant_id", "quantity"],
            },
        }
    },
    "required": ["items"],
},
```
Also register it with the MCP server's cart tools in `tools/cart.py` if you expose cart tools there.
