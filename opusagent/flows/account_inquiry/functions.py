"""
Account Inquiry Flow Functions

Function implementations for the account inquiry conversation flow.
"""

import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict

from .prompts import (
    ACCOUNT_VERIFICATION_PROMPT,
    BALANCE_INQUIRY_PROMPT,
    TRANSACTION_HISTORY_PROMPT,
    TRANSACTION_SEARCH_PROMPT,
    ACCOUNT_INFO_PROMPT,
    ACCOUNT_STATUS_PROMPT,
)

logger = logging.getLogger(__name__)


def verify_customer_identity(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify customer identity before providing account information.

    Args:
        arguments: Function arguments containing verification information

    Returns:
        Verification results with guidance for next actions
    """
    verification_method = arguments.get("verification_method", "")
    account_number = arguments.get("account_number", "")
    ssn_last_four = arguments.get("ssn_last_four", "")
    phone_number = arguments.get("phone_number", "")

    logger.info(f"Identity verification attempted using method: {verification_method}")

    # Simulate verification logic
    verification_successful = False
    account_id = None

    if verification_method == "account_number" and account_number:
        # Simulate account number verification
        if len(account_number) >= 6:  # Basic validation
            verification_successful = True
            account_id = f"ACC-{account_number[-6:]}"
    elif verification_method == "ssn" and ssn_last_four:
        # Simulate SSN verification
        if len(ssn_last_four) == 4 and ssn_last_four.isdigit():
            verification_successful = True
            account_id = f"ACC-SSN-{ssn_last_four}"
    elif verification_method == "phone" and phone_number:
        # Simulate phone verification
        if len(phone_number) >= 10:
            verification_successful = True
            account_id = f"ACC-PH-{phone_number[-4:]}"

    if verification_successful:
        return {
            "status": "success",
            "verified": True,
            "account_id": account_id,
            "prompt_guidance": "Thank you, your identity has been verified. How can I help you with your account today?",
            "next_action": "provide_account_services",
            "context": {
                "stage": "identity_verified",
                "account_id": account_id,
                "verification_method": verification_method,
            },
        }
    else:
        return {
            "status": "failed",
            "verified": False,
            "prompt_guidance": "I'm having trouble verifying your identity with the information provided. Let me transfer you to a specialist who can assist you further.",
            "next_action": "transfer_to_human",
            "context": {
                "stage": "verification_failed",
                "verification_method": verification_method,
            },
        }


def check_account_balance(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve customer's current account balance and available funds.

    Args:
        arguments: Function arguments containing account ID

    Returns:
        Account balance information
    """
    account_id = arguments.get("account_id", "")
    include_pending = arguments.get("include_pending", True)

    logger.info(f"Balance check requested for account: {account_id}")

    # Simulate balance lookup
    current_balance = 2547.83
    available_balance = 2425.83 if include_pending else current_balance
    pending_transactions = 122.00 if include_pending else 0

    balance_info = {
        "current_balance": current_balance,
        "available_balance": available_balance,
        "pending_amount": pending_transactions,
        "currency": "USD",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    prompt_guidance = f"""Your current account balance is ${current_balance:,.2f}.
Your available balance (including pending transactions) is ${available_balance:,.2f}."""

    if pending_transactions > 0:
        prompt_guidance += f"\nYou have ${pending_transactions:,.2f} in pending transactions."

    prompt_guidance += "\n\nIs there anything else I can help you with regarding your account?"

    return {
        "status": "success",
        "function_name": "check_account_balance",
        "prompt_guidance": prompt_guidance,
        "next_action": "offer_additional_help",
        "balance_info": balance_info,
        "context": {
            "stage": "balance_provided",
            "account_id": account_id,
            "balance_checked": True,
        },
    }


def get_transaction_history(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve customer's transaction history for a specified period.

    Args:
        arguments: Function arguments containing account ID and time period

    Returns:
        Transaction history information
    """
    account_id = arguments.get("account_id", "")
    time_period = arguments.get("time_period", "30_days")
    start_date = arguments.get("start_date", "")
    end_date = arguments.get("end_date", "")
    transaction_limit = arguments.get("transaction_limit", 20)

    logger.info(f"Transaction history requested for account: {account_id}, period: {time_period}")

    # Simulate transaction data generation
    transactions = _generate_sample_transactions(time_period, start_date, end_date, transaction_limit)

    transaction_count = len(transactions)
    
    prompt_guidance = f"Here are your recent transactions"
    
    if time_period == "7_days":
        prompt_guidance += " for the last 7 days"
    elif time_period == "30_days":
        prompt_guidance += " for the last 30 days"
    elif time_period == "90_days":
        prompt_guidance += " for the last 90 days"
    elif time_period == "custom" and start_date and end_date:
        prompt_guidance += f" from {start_date} to {end_date}"

    prompt_guidance += f" (showing {transaction_count} transactions):\n\n"

    # Format transactions for display
    for i, tx in enumerate(transactions[:5], 1):  # Show first 5 transactions
        prompt_guidance += f"{i}. {tx['date']} - {tx['description']} - ${tx['amount']:,.2f} ({tx['type']})\n"

    if transaction_count > 5:
        prompt_guidance += f"\n... and {transaction_count - 5} more transactions."

    prompt_guidance += "\n\nWould you like me to look up a specific transaction or help you with anything else?"

    return {
        "status": "success",
        "function_name": "get_transaction_history",
        "prompt_guidance": prompt_guidance,
        "next_action": "offer_transaction_details",
        "transactions": transactions,
        "transaction_count": transaction_count,
        "context": {
            "stage": "transaction_history_provided",
            "account_id": account_id,
            "time_period": time_period,
        },
    }


def search_specific_transaction(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for specific transactions based on customer criteria.

    Args:
        arguments: Function arguments containing search criteria

    Returns:
        Search results with matching transactions
    """
    account_id = arguments.get("account_id", "")
    amount = arguments.get("amount")
    amount_range = arguments.get("amount_range", {})
    merchant_description = arguments.get("merchant_description", "")
    transaction_type = arguments.get("transaction_type", "")
    date_range = arguments.get("date_range", {})

    logger.info(f"Transaction search requested for account: {account_id}")

    # Simulate transaction search
    matching_transactions = _search_transactions(
        amount=amount,
        amount_range=amount_range,
        merchant_description=merchant_description,
        transaction_type=transaction_type,
        date_range=date_range
    )

    match_count = len(matching_transactions)

    if match_count == 0:
        prompt_guidance = "I couldn't find any transactions matching your criteria. Could you provide different details or a broader search range?"
        next_action = "refine_search"
    elif match_count == 1:
        tx = matching_transactions[0]
        prompt_guidance = f"I found one transaction that matches your description:\n\n"
        prompt_guidance += f"Date: {tx['date']}\n"
        prompt_guidance += f"Description: {tx['description']}\n"
        prompt_guidance += f"Amount: ${tx['amount']:,.2f} ({tx['type']})\n"
        prompt_guidance += f"Reference: {tx['reference']}\n\n"
        prompt_guidance += "Is this the transaction you were looking for?"
        next_action = "confirm_transaction"
    else:
        prompt_guidance = f"I found {match_count} transactions that match your criteria:\n\n"
        for i, tx in enumerate(matching_transactions[:3], 1):
            prompt_guidance += f"{i}. {tx['date']} - {tx['description']} - ${tx['amount']:,.2f}\n"
        
        if match_count > 3:
            prompt_guidance += f"\n... and {match_count - 3} more matches."
        
        prompt_guidance += "\n\nCould you provide more specific details to narrow down the search?"
        next_action = "refine_search"

    return {
        "status": "success",
        "function_name": "search_specific_transaction",
        "prompt_guidance": prompt_guidance,
        "next_action": next_action,
        "matching_transactions": matching_transactions,
        "match_count": match_count,
        "context": {
            "stage": "transaction_search_complete",
            "account_id": account_id,
            "search_criteria": arguments,
        },
    }


def get_account_information(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve customer's account information and details.

    Args:
        arguments: Function arguments containing account ID and info type

    Returns:
        Account information based on requested type
    """
    account_id = arguments.get("account_id", "")
    info_type = arguments.get("info_type", "basic")

    logger.info(f"Account information requested for account: {account_id}, type: {info_type}")

    # Simulate account information retrieval
    account_info = _get_sample_account_info(account_id, info_type)

    prompt_guidance = "Here's your account information:\n\n"

    if info_type in ["basic", "all"]:
        prompt_guidance += f"Account Number: {account_info['account_number']}\n"
        prompt_guidance += f"Account Type: {account_info['account_type']}\n"
        prompt_guidance += f"Account Status: {account_info['status']}\n"
        prompt_guidance += f"Date Opened: {account_info['date_opened']}\n\n"

    if info_type in ["contact", "all"]:
        prompt_guidance += f"Contact Information:\n"
        prompt_guidance += f"Phone: {account_info['phone']}\n"
        prompt_guidance += f"Email: {account_info['email']}\n"
        prompt_guidance += f"Address: {account_info['address']}\n\n"

    if info_type in ["features", "all"]:
        prompt_guidance += f"Account Features:\n"
        for feature in account_info['features']:
            prompt_guidance += f"• {feature}\n"
        prompt_guidance += "\n"

    if info_type in ["rates", "all"]:
        prompt_guidance += f"Interest Rate: {account_info['interest_rate']}%\n"
        prompt_guidance += f"Monthly Fee: ${account_info['monthly_fee']}\n\n"

    prompt_guidance += "Is there any other account information you'd like to know about?"

    return {
        "status": "success",
        "function_name": "get_account_information",
        "prompt_guidance": prompt_guidance,
        "next_action": "offer_additional_help",
        "account_info": account_info,
        "context": {
            "stage": "account_info_provided",
            "account_id": account_id,
            "info_type": info_type,
        },
    }


def check_account_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check customer's account status, holds, and alerts.

    Args:
        arguments: Function arguments containing account ID

    Returns:
        Account status information
    """
    account_id = arguments.get("account_id", "")
    include_security_info = arguments.get("include_security_info", True)

    logger.info(f"Account status check requested for account: {account_id}")

    # Simulate account status check
    status_info = _get_sample_account_status(account_id, include_security_info)

    prompt_guidance = f"Your account status:\n\n"
    prompt_guidance += f"Account Standing: {status_info['standing']}\n"
    
    if status_info['holds']:
        prompt_guidance += f"Active Holds: {len(status_info['holds'])} hold(s)\n"
        for hold in status_info['holds']:
            prompt_guidance += f"  • {hold['type']}: ${hold['amount']:,.2f} (expires {hold['expires']})\n"
    else:
        prompt_guidance += "Active Holds: None\n"

    if status_info['alerts']:
        prompt_guidance += f"\nAccount Alerts:\n"
        for alert in status_info['alerts'][:3]:  # Show first 3 alerts
            prompt_guidance += f"  • {alert['type']}: {alert['message']}\n"
    else:
        prompt_guidance += "\nAccount Alerts: None\n"

    if include_security_info and status_info.get('security_activity'):
        prompt_guidance += f"\nRecent Security Activity:\n"
        for activity in status_info['security_activity'][:2]:  # Show first 2 activities
            prompt_guidance += f"  • {activity['date']}: {activity['description']}\n"

    prompt_guidance += "\n\nIs there anything specific about your account status you'd like me to explain?"

    return {
        "status": "success",
        "function_name": "check_account_status",
        "prompt_guidance": prompt_guidance,
        "next_action": "offer_additional_help",
        "status_info": status_info,
        "context": {
            "stage": "account_status_provided",
            "account_id": account_id,
        },
    }


def transfer_to_human(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle transfer to human agent.

    Args:
        arguments: Function arguments containing transfer context and reason

    Returns:
        Formatted prompt and guidance for human transfer
    """
    reason = arguments.get("reason", "general inquiry")
    priority = arguments.get("priority", "normal")
    context = arguments.get("context", {})

    # Log the transfer request
    logger.info(f"Transfer to human requested. Reason: {reason}, Priority: {priority}")

    # Generate a transfer reference number
    transfer_id = f"AI-{uuid.uuid4().hex[:8].upper()}"

    # Format the transfer message
    transfer_message = (
        f"I understand you'd like to speak with a human agent regarding {reason}. "
        f"I'll transfer you now. Your reference number is {transfer_id}. "
        f"Please hold while I connect you with a representative who can provide additional assistance."
    )

    return {
        "status": "success",
        "function_name": "transfer_to_human",
        "prompt_guidance": transfer_message,
        "next_action": "end_conversation",
        "transfer_id": transfer_id,
        "priority": priority,
        "context": {
            "stage": "human_transfer",
            "reason": reason,
            "priority": priority,
            "transfer_id": transfer_id,
            **context,
        },
    }


# Helper functions for simulation

def _generate_sample_transactions(time_period: str, start_date: str, end_date: str, limit: int) -> list:
    """Generate sample transaction data for testing."""
    transactions = []
    
    # Determine date range
    end_dt = datetime.now()
    if time_period == "7_days":
        start_dt = end_dt - timedelta(days=7)
    elif time_period == "30_days":
        start_dt = end_dt - timedelta(days=30)
    elif time_period == "90_days":
        start_dt = end_dt - timedelta(days=90)
    elif time_period == "custom" and start_date and end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=30)

    # Sample transaction templates
    sample_transactions = [
        {"description": "Amazon Purchase", "amount": -89.99, "type": "debit"},
        {"description": "Salary Deposit", "amount": 3500.00, "type": "credit"},
        {"description": "Grocery Store", "amount": -156.78, "type": "debit"},
        {"description": "ATM Withdrawal", "amount": -100.00, "type": "withdrawal"},
        {"description": "Electric Bill", "amount": -125.43, "type": "debit"},
        {"description": "Transfer from Savings", "amount": 500.00, "type": "transfer"},
        {"description": "Coffee Shop", "amount": -8.45, "type": "debit"},
        {"description": "Gas Station", "amount": -45.67, "type": "debit"},
        {"description": "Interest Payment", "amount": 12.34, "type": "credit"},
        {"description": "Monthly Fee", "amount": -15.00, "type": "fee"},
    ]

    # Generate transactions within date range
    for i in range(min(limit, len(sample_transactions))):
        tx_date = start_dt + timedelta(days=i * (end_dt - start_dt).days // limit)
        tx = sample_transactions[i].copy()
        tx.update({
            "date": tx_date.strftime("%Y-%m-%d"),
            "reference": f"TXN{uuid.uuid4().hex[:8].upper()}",
            "balance_after": 2547.83 + sum(t["amount"] for t in transactions),
        })
        transactions.append(tx)

    return sorted(transactions, key=lambda x: x["date"], reverse=True)


def _search_transactions(amount=None, amount_range=None, merchant_description="", transaction_type="", date_range=None):
    """Simulate transaction search logic."""
    # Generate some sample matching transactions
    matching_transactions = []
    
    if merchant_description.lower() in ["amazon", "grocery", "coffee"]:
        matching_transactions.append({
            "date": "2024-01-15",
            "description": f"{merchant_description.title()} Purchase",
            "amount": amount or 89.99,
            "type": transaction_type or "debit",
            "reference": f"TXN{uuid.uuid4().hex[:8].upper()}",
        })
    
    return matching_transactions


def _get_sample_account_info(account_id: str, info_type: str) -> dict:
    """Generate sample account information."""
    return {
        "account_number": f"****{account_id[-4:]}",
        "account_type": "Checking Account",
        "status": "Active",
        "date_opened": "2020-03-15",
        "phone": "(555) 123-4567",
        "email": "customer@email.com",
        "address": "123 Main St, Anytown, ST 12345",
        "features": [
            "Online Banking",
            "Mobile Deposit", 
            "Overdraft Protection",
            "Free ATM Access"
        ],
        "interest_rate": 0.50,
        "monthly_fee": 0.00,
    }


def _get_sample_account_status(account_id: str, include_security: bool) -> dict:
    """Generate sample account status information."""
    status = {
        "standing": "Good Standing",
        "holds": [],
        "alerts": [],
    }
    
    if include_security:
        status["security_activity"] = [
            {
                "date": "2024-01-10",
                "description": "Successful login from mobile app"
            },
            {
                "date": "2024-01-08", 
                "description": "Password changed successfully"
            }
        ]
    
    return status


# Function registry mapping for easy access
ACCOUNT_INQUIRY_FUNCTIONS = OrderedDict([
    ("verify_customer_identity", verify_customer_identity),
    ("check_account_balance", check_account_balance),
    ("get_transaction_history", get_transaction_history),
    ("search_specific_transaction", search_specific_transaction),
    ("get_account_information", get_account_information),
    ("check_account_status", check_account_status),
    ("transfer_to_human", transfer_to_human),
])


def get_account_inquiry_functions() -> Dict[str, Any]:
    """
    Get all function implementations for the account inquiry flow.

    Returns:
        Dictionary mapping function names to callable implementations
    """
    return ACCOUNT_INQUIRY_FUNCTIONS.copy()


def get_function_by_name(function_name: str):
    """
    Get a specific function implementation by name.

    Args:
        function_name: Name of the function to retrieve

    Returns:
        Function implementation

    Raises:
        ValueError: If function name is not found
    """
    if function_name not in ACCOUNT_INQUIRY_FUNCTIONS:
        available_functions = list(ACCOUNT_INQUIRY_FUNCTIONS.keys())
        raise ValueError(
            f"Function '{function_name}' not found. Available functions: {available_functions}"
        )

    return ACCOUNT_INQUIRY_FUNCTIONS[function_name] 