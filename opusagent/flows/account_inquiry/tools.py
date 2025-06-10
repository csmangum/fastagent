"""
Account Inquiry Flow Tools

OpenAI function tool definitions for the account inquiry conversation flow.
"""

from typing import Any, Dict, List

# Tool definition for customer identity verification
VERIFY_CUSTOMER_IDENTITY_TOOL = {
    "type": "function",
    "name": "verify_customer_identity",
    "description": "Verify customer identity before providing account information",
    "parameters": {
        "type": "object",
        "properties": {
            "account_number": {
                "type": "string",
                "description": "Customer's account number"
            },
            "ssn_last_four": {
                "type": "string",
                "description": "Last 4 digits of customer's SSN"
            },
            "phone_number": {
                "type": "string",
                "description": "Phone number on file"
            },
            "verification_method": {
                "type": "string",
                "enum": ["account_number", "ssn", "phone"],
                "description": "Which method is being used for verification"
            }
        },
        "required": ["verification_method"]
    }
}

# Tool definition for checking account balance
CHECK_ACCOUNT_BALANCE_TOOL = {
    "type": "function",
    "name": "check_account_balance",
    "description": "Retrieve customer's current account balance and available funds",
    "parameters": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "Verified customer account ID"
            },
            "include_pending": {
                "type": "boolean",
                "description": "Whether to include pending transactions in available balance",
                "default": True
            }
        },
        "required": ["account_id"]
    }
}

# Tool definition for getting transaction history
GET_TRANSACTION_HISTORY_TOOL = {
    "type": "function",
    "name": "get_transaction_history",
    "description": "Retrieve customer's transaction history for a specified period",
    "parameters": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "Verified customer account ID"
            },
            "time_period": {
                "type": "string",
                "enum": ["7_days", "30_days", "90_days", "custom"],
                "description": "Time period for transaction history"
            },
            "start_date": {
                "type": "string",
                "description": "Start date for custom range (YYYY-MM-DD format)"
            },
            "end_date": {
                "type": "string",
                "description": "End date for custom range (YYYY-MM-DD format)"
            },
            "transaction_limit": {
                "type": "integer",
                "description": "Maximum number of transactions to return",
                "default": 20
            }
        },
        "required": ["account_id", "time_period"]
    }
}

# Tool definition for searching specific transactions
SEARCH_SPECIFIC_TRANSACTION_TOOL = {
    "type": "function",
    "name": "search_specific_transaction",
    "description": "Search for specific transactions based on criteria provided by customer",
    "parameters": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "Verified customer account ID"
            },
            "amount": {
                "type": "number",
                "description": "Transaction amount (approximate)"
            },
            "amount_range": {
                "type": "object",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"}
                },
                "description": "Amount range for search"
            },
            "merchant_description": {
                "type": "string",
                "description": "Merchant name or transaction description"
            },
            "transaction_type": {
                "type": "string",
                "enum": ["debit", "credit", "transfer", "fee", "deposit", "withdrawal"],
                "description": "Type of transaction"
            },
            "date_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"}
                },
                "description": "Date range for search (YYYY-MM-DD format)"
            }
        },
        "required": ["account_id"]
    }
}

# Tool definition for getting account information
GET_ACCOUNT_INFORMATION_TOOL = {
    "type": "function",
    "name": "get_account_information",
    "description": "Retrieve customer's account information and details",
    "parameters": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "Verified customer account ID"
            },
            "info_type": {
                "type": "string",
                "enum": ["basic", "contact", "features", "rates", "all"],
                "description": "Type of account information requested",
                "default": "basic"
            }
        },
        "required": ["account_id"]
    }
}

# Tool definition for checking account status
CHECK_ACCOUNT_STATUS_TOOL = {
    "type": "function",
    "name": "check_account_status",
    "description": "Check customer's account status, holds, and alerts",
    "parameters": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "Verified customer account ID"
            },
            "include_security_info": {
                "type": "boolean",
                "description": "Whether to include recent security activity",
                "default": True
            }
        },
        "required": ["account_id"]
    }
}

# Tool definition for transferring to human
TRANSFER_TO_HUMAN_TOOL = {
    "type": "function",
    "name": "transfer_to_human",
    "description": "Transfer the conversation to a human agent",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "The reason for transferring to a human agent"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "The priority level of the transfer"
            },
            "context": {
                "type": "object",
                "description": "Additional context to pass to the human agent"
            }
        },
        "required": ["reason"]
    }
}


def get_account_inquiry_tools() -> List[Dict[str, Any]]:
    """
    Get all OpenAI tool definitions for the account inquiry flow.
    
    Returns:
        List of OpenAI function tool schemas
    """
    return [
        VERIFY_CUSTOMER_IDENTITY_TOOL,
        CHECK_ACCOUNT_BALANCE_TOOL,
        GET_TRANSACTION_HISTORY_TOOL,
        SEARCH_SPECIFIC_TRANSACTION_TOOL,
        GET_ACCOUNT_INFORMATION_TOOL,
        CHECK_ACCOUNT_STATUS_TOOL,
        TRANSFER_TO_HUMAN_TOOL,
    ]


def get_tool_by_name(tool_name: str) -> Dict[str, Any]:
    """
    Get a specific tool definition by name.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        Tool definition dictionary
        
    Raises:
        ValueError: If tool name is not found
    """
    tool_map = {
        "verify_customer_identity": VERIFY_CUSTOMER_IDENTITY_TOOL,
        "check_account_balance": CHECK_ACCOUNT_BALANCE_TOOL,
        "get_transaction_history": GET_TRANSACTION_HISTORY_TOOL,
        "search_specific_transaction": SEARCH_SPECIFIC_TRANSACTION_TOOL,
        "get_account_information": GET_ACCOUNT_INFORMATION_TOOL,
        "check_account_status": CHECK_ACCOUNT_STATUS_TOOL,
        "transfer_to_human": TRANSFER_TO_HUMAN_TOOL,
    }
    
    if tool_name not in tool_map:
        available_tools = list(tool_map.keys())
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
    
    return tool_map[tool_name] 