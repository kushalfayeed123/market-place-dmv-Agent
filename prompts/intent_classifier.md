# Intent Classification

Classify the user's message into one of these intents:

## `info` — Information Request
The user wants to read or browse data. Safe to call read-only tools directly.
Examples:
- "Show me products in electronics"
- "What's the status of order #123?"
- "List my orders"
- "What categories are available?"

## `action` — Action Request
The user wants to perform a mutating action. Requires confirmation for financial actions.
Examples:
- "Add this to my cart"
- "Checkout please"
- "I want to buy the iPhone case"
- "Refund this order"
- "Create a new product"
- "Approve this merchant's KYC"

## `signin_required` — Unauthenticated Action
The user wants to perform an action but is not authenticated. Show sign-in prompt.
Triggered when: intent would be `action` but no valid session token exists.

## Classification Heuristics

1. Look for action verbs: buy, checkout, pay, order, purchase, add to cart, refund, payout, create, update, delete, approve, reject, ship, fulfill
2. If the message is a question or starts with "show", "list", "what", "how many" → likely `info`
3. If the message contains a product/order ID and a status inquiry → `info`
4. If the message expresses intent to acquire or modify something → `action`
