"""Mock order database, policy logic and tool implementations for Aria.

Eligibility is computed in code (not by the LLM) so policy answers are deterministic.
Per-session `overrides` let a demo cancellation succeed without mutating the shared
mock data, so the next evaluator still sees ORD-103 as "Processing".
"""
import copy
import re

ORDERS = {
    "ORD-101": {
        "order_id": "ORD-101",
        "customer_name": "Priya Sharma",
        "product": "Vitamin C Serum (30ml)",
        "value_inr": 699,
        "status": "Out for Delivery",
        "carrier": "BlueDart",
        "tracking_id": "BD-982103",
        "notes": "Expected by 6 PM today",
        "delivered_days_ago": None,
        "ordered_hours_ago": None,
    },
    "ORD-102": {
        "order_id": "ORD-102",
        "customer_name": "Rahul Verma",
        "product": "Hydrating Sunscreen SPF 50",
        "value_inr": 499,
        "status": "Delivered",
        "carrier": "Delhivery",
        "tracking_id": "DL-441029",
        "notes": "Delivered 14 days ago",
        "delivered_days_ago": 14,
        "ordered_hours_ago": None,
    },
    "ORD-103": {
        "order_id": "ORD-103",
        "customer_name": "Ananya Patel",
        "product": "Green Tea Face Wash + Toner",
        "value_inr": 850,
        "status": "Processing",
        "carrier": None,
        "tracking_id": None,
        "notes": "Ordered 3 hours ago. Eligible for cancellation",
        "delivered_days_ago": None,
        "ordered_hours_ago": 3,
    },
}

RETURN_WINDOW_DAYS = 7
DAMAGE_REPORT_WINDOW_HOURS = 48


def normalize_order_id(raw) -> str | None:
    """'ord 101', 'ORD101', 'o r d - 1 0 1', '101' -> 'ORD-101'. None if no digits."""
    if not raw:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw)).upper()
    cleaned = cleaned.removeprefix("ORD")
    digits = re.sub(r"\D", "", cleaned)
    return f"ORD-{digits}" if digits else None


def _lookup(order_id, overrides: dict | None):
    oid = normalize_order_id(order_id)
    if oid is None or oid not in ORDERS:
        return oid, None
    order = copy.deepcopy(ORDERS[oid])
    if overrides and oid in overrides:
        order.update(overrides[oid])
    return oid, order


def get_order_details(order_id=None, overrides: dict | None = None) -> dict:
    if not order_id:
        return {"found": False, "error": "MISSING_ORDER_ID",
                "message": "No order ID was provided. Ask the customer for it."}
    oid, order = _lookup(order_id, overrides)
    if order is None:
        return {"found": False, "error": "ORDER_NOT_FOUND", "searched_id": oid,
                "message": "No order exists with this ID. Ask the customer to repeat or verify it."}
    return {"found": True, "order": order}


def check_eligibility(order_id=None, action=None, overrides: dict | None = None) -> dict:
    """action: 'cancel' | 'return' | 'damaged_report'."""
    if action not in ("cancel", "return", "damaged_report"):
        return {"error": "INVALID_ACTION", "message": "action must be cancel, return or damaged_report"}
    lookup = get_order_details(order_id, overrides)
    if not lookup["found"]:
        return lookup
    o = lookup["order"]
    status = o["status"]
    res = {"order_id": o["order_id"], "action": action, "status": status}

    if action == "cancel":
        if status == "Processing":
            return {**res, "eligible": True,
                    "reason": "Order is still Processing, so it can be cancelled."}
        if status in ("Shipped", "Out for Delivery"):
            return {**res, "eligible": False,
                    "reason": f"Order is {status} and cannot be cancelled. "
                              "The customer may refuse delivery at the doorstep."}
        if status == "Cancelled":
            return {**res, "eligible": False, "reason": "Order is already cancelled."}
        return {**res, "eligible": False,
                "reason": "Order is already delivered, so it cannot be cancelled. "
                          "Return policy may apply."}

    if status != "Delivered":
        return {**res, "eligible": False,
                "reason": f"Order is {status}, not delivered yet. "
                          "Returns and damage reports apply only after delivery."}

    days = o["delivered_days_ago"]
    if action == "return":
        if days <= RETURN_WINDOW_DAYS:
            return {**res, "eligible": True, "days_since_delivery": days,
                    "conditions": "Only if the product is unopened, unused and in original packaging. "
                                  "Confirm this with the customer."}
        return {**res, "eligible": False, "days_since_delivery": days,
                "reason": f"Delivered {days} days ago; the return window is {RETURN_WINDOW_DAYS} days."}

    # damaged_report
    if days * 24 <= DAMAGE_REPORT_WINDOW_HOURS:
        return {**res, "eligible": True, "days_since_delivery": days,
                "conditions": "Customer must share photos of the damage for a replacement."}
    return {**res, "eligible": False, "days_since_delivery": days,
            "reason": f"Delivered {days} days ago; damage must be reported within "
                      f"{DAMAGE_REPORT_WINDOW_HOURS} hours of delivery."}


def cancel_order(order_id=None, overrides: dict | None = None) -> dict:
    """Cancels only if eligible. Writes to the per-session overrides, never to ORDERS."""
    check = check_eligibility(order_id, "cancel", overrides)
    if check.get("eligible") is not True:
        return {"success": False, **check}
    if overrides is not None:
        overrides[check["order_id"]] = {"status": "Cancelled", "notes": "Cancelled by customer on call"}
    return {"success": True, "order_id": check["order_id"], "new_status": "Cancelled"}


def public_orders() -> list[dict]:
    """Orders for the frontend 'Test Orders' helper card."""
    return [copy.deepcopy(o) for o in ORDERS.values()]


# ---------- Tool schemas (OpenAI/Groq function-calling format) ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_details",
        "description": "Look up a customer's order by ID: product, value, status, carrier, tracking ID and "
                       "delivery notes. Call this whenever the customer asks about the status, location or "
                       "details of an order. Never guess order data.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string",
                         "description": "Order ID such as ORD-101. Convert spoken digits ('one oh one') to ORD-101."}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "check_eligibility",
        "description": "Check whether an order is eligible for cancellation, return, or a damaged-product "
                       "replacement under Aura's policies. Always call this before telling a customer whether "
                       "they can cancel or return something.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "action": {"type": "string", "enum": ["cancel", "return", "damaged_report"]}},
            "required": ["order_id", "action"]}}},
    {"type": "function", "function": {
        "name": "cancel_order",
        "description": "Cancel an order. Only call after the customer has explicitly confirmed they want to "
                       "cancel. Fails automatically if the order is not in Processing status.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
]


def execute_tool(name: str, args: dict, overrides: dict) -> dict:
    try:
        if name == "get_order_details":
            return get_order_details(args.get("order_id"), overrides)
        if name == "check_eligibility":
            return check_eligibility(args.get("order_id"), args.get("action"), overrides)
        if name == "cancel_order":
            return cancel_order(args.get("order_id"), overrides)
        return {"error": "UNKNOWN_TOOL", "message": f"No tool named {name}"}
    except Exception as exc:  # never let a tool crash the call
        return {"error": "TOOL_FAILURE", "message": str(exc)}
