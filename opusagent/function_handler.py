"""
Function Call Handler for OpenAI Realtime API Integration

This module provides a comprehensive function call management system for OpenAI's Realtime API,
enabling seamless integration between AI conversations and backend function execution. The handler
manages the complete lifecycle of function calls including streaming argument accumulation,
function execution, and response delivery.

Architecture Overview:
    The FunctionHandler class serves as the central orchestrator for all function call operations,
    providing a clean separation between conversation flow and business logic. It handles both
    synchronous and asynchronous function execution while maintaining state for streaming calls.

Key Features:
    - **Streaming Support**: Accumulates function arguments delivered across multiple delta events
    - **Function Registry**: Dynamic registration/unregistration of callable functions
    - **Async/Sync Compatibility**: Supports both synchronous and asynchronous function implementations
    - **Error Handling**: Comprehensive error handling with detailed logging for debugging
    - **Response Management**: Automatic response generation triggering after function execution
    - **State Management**: Tracks active function calls and manages cleanup

Supported Function Flows:
    1. **Card Replacement Flow**: Complete end-to-end card replacement process
       - Member account confirmation
       - Replacement reason collection
       - Address verification
       - Replacement processing and completion

    2. **Loan Application Flow**: Comprehensive loan application processing
       - Loan type selection and information
       - Amount collection with type-specific guidance
       - Income and employment verification
       - Credit check consent and application submission
       - Pre-approval processing

    3. **Banking Operations**: Basic banking function simulations
       - Balance inquiries
       - Fund transfers
       - Intent classification

Usage Example:
    ```python
    # Initialize handler with WebSocket connection
    handler = FunctionHandler(realtime_websocket)

    # Register custom function
    def my_custom_function(args):
        return {"status": "success", "data": args}

    handler.register_function("my_function", my_custom_function)

    # Handle incoming function call events
    await handler.handle_function_call_arguments_delta(delta_event)
    await handler.handle_function_call_arguments_done(done_event)
    ```

Event Processing Workflow:
    1. **Delta Events**: Incremental argument data is accumulated in active_function_calls
    2. **Done Events**: Complete arguments trigger function lookup and execution
    3. **Function Execution**: Registered functions are called with parsed arguments
    4. **Result Delivery**: Function results are sent back to OpenAI via WebSocket
    5. **Response Generation**: AI response generation is automatically triggered
    6. **Cleanup**: Function call state is cleaned up after completion

Dependencies:
    - asyncio: For asynchronous function execution and task management
    - json: For argument parsing and result serialization
    - logging: For comprehensive operation logging
    - uuid: For generating unique identifiers
    - OpenAI Realtime API: For WebSocket communication

Thread Safety:
    This module is designed for use in asyncio environments and maintains thread-safe
    operation through proper async/await patterns and state management.

Note:
    Function implementations should follow the pattern of accepting a dictionary of
    arguments and returning a dictionary result. The handler automatically manages
    JSON serialization/deserialization for OpenAI API compatibility.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, Optional

from opusagent.config.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class FunctionHandler:
    """
    Handles function call events from the OpenAI Realtime API.

    This class manages:
    - A registry of available functions
    - Accumulation of streaming function call arguments
    - Execution of functions (sync/async)
    - Sending results back to the OpenAI Realtime API

    Attributes:
        function_registry: Dictionary mapping function names to callable implementations
        active_function_calls: Dictionary tracking ongoing function calls by call_id
        realtime_websocket: WebSocket connection to OpenAI Realtime API for sending responses
    """

    def __init__(self, realtime_websocket, call_recorder=None, voice="verse"):
        """
        Initialize the function handler.

        Args:
            realtime_websocket: WebSocket connection to OpenAI Realtime API
            call_recorder: Optional CallRecorder instance for logging function calls
        """
        self.realtime_websocket = realtime_websocket
        self.call_recorder = call_recorder
        self.function_registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.active_function_calls: Dict[str, Dict[str, Any]] = (
            {}
        )  # call_id -> {function_name, arguments_buffer, item_id, etc.}
        self.voice = voice
        # Register default functions
        self._register_default_functions()

    def _register_default_functions(self):
        """Register the default function implementations."""
        self.register_function("get_balance", self._func_get_balance)
        self.register_function("transfer_funds", self._func_transfer_funds)
        self.register_function("call_intent", self._func_call_intent)
        self.register_function("transfer_to_human", self._func_transfer_to_human)

        # Card replacement flow functions
        self.register_function(
            "member_account_confirmation", self._func_member_account_confirmation
        )
        self.register_function("replacement_reason", self._func_replacement_reason)
        self.register_function("confirm_address", self._func_confirm_address)
        self.register_function(
            "start_card_replacement", self._func_start_card_replacement
        )
        self.register_function(
            "finish_card_replacement", self._func_finish_card_replacement
        )
        self.register_function("wrap_up", self._func_wrap_up)

        # Loan application flow functions
        self.register_function("loan_type_selection", self._func_loan_type_selection)
        self.register_function(
            "loan_amount_collection", self._func_loan_amount_collection
        )
        self.register_function("income_verification", self._func_income_verification)
        self.register_function(
            "employment_verification", self._func_employment_verification
        )
        self.register_function("credit_check_consent", self._func_credit_check_consent)
        self.register_function(
            "submit_loan_application", self._func_submit_loan_application
        )
        self.register_function("loan_pre_approval", self._func_loan_pre_approval)
        self.register_function("process_replacement", self._func_process_replacement)

    def register_function(
        self, name: str, func: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Register a function in the function registry.

        Args:
            name: The name of the function as it will be called by OpenAI
            func: The callable function implementation (can be sync or async)
        """
        self.function_registry[name] = func
        logger.info(f"Registered function: {name}")

    def unregister_function(self, name: str) -> bool:
        """
        Unregister a function from the registry.

        Args:
            name: The name of the function to remove

        Returns:
            True if function was removed, False if it didn't exist
        """
        if name in self.function_registry:
            del self.function_registry[name]
            logger.info(f"Unregistered function: {name}")
            return True
        return False

    def get_registered_functions(self) -> list[str]:
        """
        Get a list of all registered function names.

        Returns:
            List of function names
        """
        return list(self.function_registry.keys())

    async def handle_function_call(self, response_dict: Dict[str, Any]) -> None:
        """
        Handle function call event from OpenAI (out-of-band).

        Args:
            response_dict: The function call event data from OpenAI
        """
        logger.info(f"Function call event received: {response_dict}")

        try:
            # Arguments are a JSON string
            arguments_str = response_dict.get("arguments", "{}")
            call_id = response_dict.get("call_id")
            item_id = response_dict.get("item_id")
            output_index = response_dict.get("output_index")
            response_id = response_dict.get("response_id")

            # The function name is typically in the parent response or item, but for demo, assume it's in arguments
            # You may need to adjust this based on your OpenAI payloads
            args = json.loads(arguments_str)
            function_name = (
                args.get("function_name") or args.get("name") or args.get("tool_name")
            )
            if not function_name:
                logger.error(f"No function name found in arguments: {arguments_str}")
                return

            logger.info(f"Dispatching function: {function_name} with args: {args}")

            # Run function out-of-band
            asyncio.create_task(
                self._execute_and_respond_to_function(
                    function_name, args, call_id, item_id, output_index, response_id
                )
            )
        except Exception as e:
            logger.error(f"Error parsing function call: {e}")

    async def handle_function_call_arguments_delta(
        self, response_dict: Dict[str, Any]
    ) -> None:
        """
        Handle function call arguments delta events from OpenAI.

        These events contain incremental pieces of the function call arguments
        that need to be accumulated until the function call is complete.

        Args:
            response_dict: The function call arguments delta event data
        """
        call_id = response_dict.get("call_id")
        delta = response_dict.get("delta", "")
        item_id = response_dict.get("item_id")
        output_index = response_dict.get("output_index", 0)
        response_id = response_dict.get("response_id")

        if not call_id:
            logger.warning(
                f"Function call delta received without call_id: {response_dict}"
            )
            return

        # Initialize or update the function call state
        if call_id not in self.active_function_calls:
            self.active_function_calls[call_id] = {
                "arguments_buffer": "",
                "item_id": item_id,
                "output_index": output_index,
                "response_id": response_id,
                "function_name": None,  # Will be determined from the arguments
            }

        # Accumulate the arguments
        self.active_function_calls[call_id]["arguments_buffer"] += delta

        logger.debug(
            f"Function call arguments delta for {call_id}: '{delta}' (total: {len(self.active_function_calls[call_id]['arguments_buffer'])} chars)"
        )

    async def handle_function_call_arguments_done(
        self, response_dict: Dict[str, Any]
    ) -> None:
        """
        Handle function call arguments completion events from OpenAI.

        This event indicates that all arguments for a function call have been received
        and the function can now be executed.

        Args:
            response_dict: The function call arguments done event data
        """
        logger.info(f"🔧 FUNCTION CALL ARGUMENTS DONE HANDLER CALLED")
        logger.info(f"🔧 Full response_dict: {response_dict}")

        call_id = response_dict.get("call_id")
        final_arguments = response_dict.get("arguments", "")
        item_id = response_dict.get("item_id")
        output_index = response_dict.get("output_index", 0)
        response_id = response_dict.get("response_id")

        logger.info(f"🔧 Extracted values:")
        logger.info(f"   call_id: {call_id}")
        logger.info(f"   final_arguments: {final_arguments}")
        logger.info(f"   item_id: {item_id}")
        logger.info(f"   output_index: {output_index}")
        logger.info(f"   response_id: {response_id}")

        if not call_id:
            logger.warning(
                f"🚨 Function call done received without call_id: {response_dict}"
            )
            return

        logger.info(
            f"🔧 Active function calls: {list(self.active_function_calls.keys())}"
        )

        # Use the final arguments from the done event, or fall back to accumulated buffer
        if final_arguments:
            arguments_str = final_arguments
            logger.info(f"🔧 Using final_arguments from done event: {arguments_str}")
        elif call_id in self.active_function_calls:
            arguments_str = self.active_function_calls[call_id]["arguments_buffer"]
            logger.info(f"🔧 Using accumulated buffer: {arguments_str}")
        else:
            logger.error(f"🚨 Function call done for unknown call_id: {call_id}")
            logger.error(
                f"🚨 Known call_ids: {list(self.active_function_calls.keys())}"
            )
            return

        logger.info(
            f"🔧 Function call arguments complete for {call_id}: {arguments_str}"
        )

        try:
            # Parse the JSON arguments
            args = json.loads(arguments_str) if arguments_str else {}
            logger.info(f"🔧 Parsed arguments: {args}")

            # Get the function name from the captured function call state
            function_name = None
            if call_id in self.active_function_calls:
                function_name = self.active_function_calls[call_id].get("function_name")
                logger.info(f"🔧 Found function_name in active calls: {function_name}")
            else:
                logger.error(f"🚨 call_id {call_id} not found in active_function_calls")

            if not function_name:
                logger.error(f"🚨 No function name found for call_id: {call_id}")
                logger.error(f"🚨 Arguments string: {arguments_str}")
                logger.error(
                    f"🚨 Active function calls state: {self.active_function_calls}"
                )
                # Clean up the active function call
                if call_id in self.active_function_calls:
                    del self.active_function_calls[call_id]
                return

            logger.info(f"🚀 Executing function: {function_name} with args: {args}")

            # Execute the function
            asyncio.create_task(
                self._execute_and_respond_to_function(
                    function_name, args, call_id, item_id, output_index, response_id
                )
            )
            logger.info(f"🚀 Function execution task created for {function_name}")

        except json.JSONDecodeError as e:
            logger.error(f"🚨 Failed to parse function arguments JSON: {e}")
            logger.error(f"🚨 Arguments string: {arguments_str}")
        except Exception as e:
            logger.error(f"🚨 Error processing function call: {e}")
            import traceback

            logger.error(f"🚨 Traceback: {traceback.format_exc()}")
        finally:
            # Clean up the active function call
            if call_id in self.active_function_calls:
                logger.info(f"🔧 Cleaning up active function call for {call_id}")
                del self.active_function_calls[call_id]

    async def _execute_and_respond_to_function(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        call_id: str,
        item_id: str,
        output_index: int,
        response_id: str,
    ) -> None:
        """
        Execute a function and send the result back to OpenAI.

        Args:
            function_name: Name of the function to execute
            arguments: Function arguments
            call_id: Unique call identifier
            item_id: Item identifier
            output_index: Output index
            response_id: Response identifier
        """
        logger.info(f"🔥 _execute_and_respond_to_function called:")
        logger.info(f"   function_name: {function_name}")
        logger.info(f"   arguments: {arguments}")
        logger.info(f"   call_id: {call_id}")
        logger.info(f"   item_id: {item_id}")
        logger.info(f"   output_index: {output_index}")
        logger.info(f"   response_id: {response_id}")

        # Execute the function
        try:
            logger.info(f"🔥 Looking up function '{function_name}' in registry...")
            logger.info(
                f"🔥 Available functions: {list(self.function_registry.keys())}"
            )

            func = self.function_registry.get(function_name)
            if not func:
                logger.error(f"🚨 Function '{function_name}' not implemented.")
                raise NotImplementedError(
                    f"Function '{function_name}' not implemented."
                )

            logger.info(f"🔥 Found function '{function_name}', executing...")

            result = (
                await func(arguments)
                if asyncio.iscoroutinefunction(func)
                else func(arguments)
            )
            logger.info(f"✅ Function {function_name} executed successfully: {result}")

            # Log function call to call recorder if available
            if self.call_recorder:
                try:
                    await self.call_recorder.log_function_call(
                        function_name=function_name,
                        arguments=arguments,
                        result=result,
                        call_id=call_id,
                    )
                except Exception as e:
                    logger.error(f"Error logging function call to recorder: {e}")

        except Exception as e:
            logger.error(f"🚨 Function execution failed: {e}")
            import traceback

            logger.error(f"🚨 Function execution traceback: {traceback.format_exc()}")
            result = {"error": str(e)}

            # Log failed function call to call recorder if available
            if self.call_recorder:
                try:
                    await self.call_recorder.log_function_call(
                        function_name=function_name,
                        arguments=arguments,
                        result=result,
                        call_id=call_id,
                    )
                except Exception as e:
                    logger.error(f"Error logging failed function call to recorder: {e}")

        # Send result back to OpenAI as a conversation item
        function_result_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        }
        logger.info(f"📤 Sending function result for {function_name}: {result}")
        logger.info(
            f"📤 Function result event: {json.dumps(function_result_event, indent=2)}"
        )

        try:
            await self.realtime_websocket.send(json.dumps(function_result_event))
            logger.info(f"✅ Function result sent successfully to OpenAI")

            # After sending function result, trigger response generation
            # This ensures the AI continues the conversation
            logger.info("🚀 Triggering response generation after function execution...")
            response_create = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "output_audio_format": "pcm16",
                    "temperature": 0.8,
                    "max_output_tokens": 4096,
                    "voice": self.voice,
                },
            }
            await self.realtime_websocket.send(json.dumps(response_create))
            logger.info("✅ Response generation triggered successfully")

        except Exception as e:
            logger.error(f"🚨 Failed to send function result or trigger response: {e}")
            import traceback

            logger.error(f"🚨 Send result traceback: {traceback.format_exc()}")

    def clear_active_function_calls(self) -> None:
        """Clear all active function call state."""
        self.active_function_calls.clear()
        logger.info("Cleared all active function calls")

    def get_active_function_calls(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a copy of the active function calls state.

        Returns:
            Dictionary of active function calls
        """
        return self.active_function_calls.copy()

    # Default function implementations
    def _func_get_balance(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a balance lookup.

        Args:
            arguments: Function arguments (unused for this example)

        Returns:
            Simulated balance information
        """
        return {"balance": 1234.56, "currency": "USD"}

    def _func_transfer_funds(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a fund transfer.

        Args:
            arguments: Function arguments containing amount and to_account

        Returns:
            Transfer status information
        """
        amount = arguments.get("amount", 0)
        to_account = arguments.get("to_account", "unknown")
        return {"status": "success", "amount": amount, "to_account": to_account}

    def _func_call_intent(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user intent identification.

        Args:
            arguments: Function arguments containing intent

        Returns:
            Intent processing results with guidance for next actions
        """
        intent = arguments.get("intent", "")
        if intent == "card_replacement":
            logger.info(
                f"Function call intent received: {arguments}"
            )
            # Return data that guides the AI's next response
            return {
                "status": "success",
                "intent": intent,
                "next_action": "ask_card_type",
                "available_cards": ["Gold card", "Silver card", "Basic card"],
                "prompt_guidance": "Ask the customer which type of card they need to replace: Gold card, Silver card, or Basic card.",
            }
        else:
            return {
                "status": "success",
                "intent": intent,
                "next_action": "continue_conversation",
            }

    # Card replacement flow function implementations
    def _func_member_account_confirmation(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle member account confirmation for card replacement.

        Args:
            arguments: Function arguments containing member accounts and context

        Returns:
            Formatted prompt and guidance for account confirmation
        """
        from opusagent.flows.card_replacement.prompts import BASE_PROMPT as base_prompt
        from opusagent.flows.card_replacement.prompts import (
            MEMBER_ACCOUNT_CONFIRMATION_PROMPT as member_account_confirmation_prompt,
        )

        member_accounts = arguments.get(
            "member_accounts", ["Gold card", "Silver card", "Basic card"]
        )
        organization_name = arguments.get("organization_name", "Bank of Peril")

        # Format the prompt with context
        formatted_prompt = member_account_confirmation_prompt.format(
            member_accounts=", ".join(member_accounts)
        )

        logger.info(
            f"Member account confirmation function called with accounts: {member_accounts}"
        )

        return {
            "status": "success",
            "function_name": "member_account_confirmation",
            "prompt_guidance": formatted_prompt,
            "next_action": "confirm_card_selection",
            "available_cards": member_accounts,
            "context": {
                "stage": "account_confirmation",
                "organization_name": organization_name,
            },
        }

    def _func_replacement_reason(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle replacement reason collection.

        Args:
            arguments: Function arguments containing card context and selected reason

        Returns:
            Formatted prompt and guidance for reason collection
        """
        from opusagent.flows.card_replacement.prompts import (
            REPLACEMENT_REASON_PROMPT as replacement_reason_prompt,
        )

        card_in_context = arguments.get("card_in_context", "your card")
        reason = arguments.get("reason", "")

        # Format the prompt with context
        formatted_prompt = replacement_reason_prompt.format(
            card_in_context=card_in_context
        )

        logger.info(
            f"Replacement reason function called for {card_in_context}, reason: {reason}"
        )

        return {
            "status": "success",
            "function_name": "replacement_reason",
            "prompt_guidance": formatted_prompt,
            "next_action": "collect_address" if reason else "ask_reason",
            "valid_reasons": ["Lost", "Damaged", "Stolen", "Other"],
            "selected_reason": reason,
            "context": {
                "stage": "reason_collection",
                "card_in_context": card_in_context,
                "reason": reason,
            },
        }

    def _func_confirm_address(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle address confirmation for card replacement.

        Args:
            arguments: Function arguments containing card context and address

        Returns:
            Formatted prompt and guidance for address confirmation
        """
        from opusagent.flows.card_replacement.prompts import (
            CONFIRM_ADDRESS_PROMPT as confirm_address_prompt,
        )

        card_in_context = arguments.get("card_in_context", "your card")
        address_on_file = arguments.get(
            "address_on_file", "123 Main St, Anytown, ST 12345"
        )
        confirmed_address = arguments.get("confirmed_address", "")

        # Format the prompt with context
        formatted_prompt = confirm_address_prompt.format(
            card_in_context=card_in_context, address_on_file=address_on_file
        )

        logger.info(f"Address confirmation function called for {card_in_context}")

        return {
            "status": "success",
            "function_name": "confirm_address",
            "prompt_guidance": formatted_prompt,
            "next_action": (
                "start_replacement" if confirmed_address else "confirm_address"
            ),
            "address_on_file": address_on_file,
            "confirmed_address": confirmed_address,
            "context": {
                "stage": "address_confirmation",
                "card_in_context": card_in_context,
                "address_on_file": address_on_file,
                "confirmed_address": confirmed_address,
            },
        }

    def _func_start_card_replacement(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle starting the card replacement process.

        Args:
            arguments: Function arguments containing card and address context

        Returns:
            Formatted prompt and guidance for starting replacement
        """
        from opusagent.flows.card_replacement.prompts import (
            START_CARD_REPLACEMENT_PROMPT as start_card_replacement_prompt,
        )

        card_in_context = arguments.get("card_in_context", "your card")
        address_in_context = arguments.get("address_in_context", "your address on file")

        # Format the prompt with context
        formatted_prompt = start_card_replacement_prompt.format(
            card_in_context=card_in_context, address_in_context=address_in_context
        )

        logger.info(
            f"Starting card replacement for {card_in_context} to {address_in_context}"
        )

        return {
            "status": "success",
            "function_name": "start_card_replacement",
            "prompt_guidance": formatted_prompt,
            "next_action": "finish_replacement",
            "context": {
                "stage": "replacement_started",
                "card_in_context": card_in_context,
                "address_in_context": address_in_context,
            },
        }

    def _func_finish_card_replacement(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle finishing the card replacement process.

        Args:
            arguments: Function arguments containing card and address context

        Returns:
            Formatted prompt and guidance for finishing replacement
        """
        from opusagent.flows.card_replacement.prompts import (
            FINISH_CARD_REPLACEMENT_PROMPT as finish_card_replacement_prompt,
        )

        card_in_context = arguments.get("card_in_context", "your card")
        address_in_context = arguments.get("address_in_context", "your address")
        delivery_time = arguments.get("delivery_time", "5-7 business days")

        # Format the prompt with context
        formatted_prompt = finish_card_replacement_prompt.format(
            card_in_context=card_in_context, address_in_context=address_in_context
        )

        logger.info(f"Finishing card replacement for {card_in_context}")

        return {
            "status": "success",
            "function_name": "finish_card_replacement",
            "prompt_guidance": formatted_prompt,
            "next_action": "wrap_up",
            "delivery_time": delivery_time,
            "context": {
                "stage": "replacement_complete",
                "card_in_context": card_in_context,
                "address_in_context": address_in_context,
                "delivery_time": delivery_time,
            },
        }

    def _func_wrap_up(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle wrapping up the call.

        Args:
            arguments: Function arguments containing organization context

        Returns:
            Formatted prompt and guidance for wrapping up
        """
        from opusagent.flows.card_replacement.prompts import (
            WRAP_UP_PROMPT as wrap_up_prompt,
        )

        organization_name = arguments.get("organization_name", "Bank of Peril")

        # Format the prompt with context
        formatted_prompt = wrap_up_prompt.format(organization_name=organization_name)

        logger.info(f"Wrapping up call for {organization_name}")

        return {
            "status": "success",
            "function_name": "wrap_up",
            "prompt_guidance": formatted_prompt,
            "next_action": "end_call",
            "context": {
                "stage": "call_complete",
                "organization_name": organization_name,
            },
        }

    # Loan application flow function implementations
    def _func_loan_type_selection(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle loan type selection and provide information.

        Args:
            arguments: Function arguments containing loan type preference

        Returns:
            Formatted prompt and guidance for loan type selection
        """
        from demo.demo_loan_application_prompts import loan_intent_confirmation_prompt

        selected_loan_type = arguments.get("loan_type", "")

        logger.info(
            f"Loan type selection function called with type: {selected_loan_type}"
        )

        return {
            "status": "success",
            "function_name": "loan_type_selection",
            "prompt_guidance": loan_intent_confirmation_prompt,
            "next_action": (
                "collect_loan_amount" if selected_loan_type else "ask_loan_type"
            ),
            "available_loan_types": [
                "Personal loan",
                "Auto loan",
                "Home mortgage",
                "Business loan",
            ],
            "selected_loan_type": selected_loan_type,
            "context": {
                "stage": "loan_type_selection",
                "selected_loan_type": selected_loan_type,
            },
        }

    def _func_loan_amount_collection(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle loan amount collection with type-specific information.

        Args:
            arguments: Function arguments containing loan type and amount

        Returns:
            Formatted prompt and guidance for amount collection
        """
        from demo.demo_loan_application_prompts import loan_amount_prompt

        loan_type = arguments.get("loan_type", "loan")
        loan_amount = arguments.get("loan_amount", 0)

        # Provide type-specific information
        loan_type_info = {
            "Personal loan": "Personal loans range from $1,000 to $50,000 with competitive rates.",
            "Auto loan": "We finance both new and used vehicles up to $100,000.",
            "Home mortgage": "We offer purchase and refinance mortgages with various term options.",
            "Business loan": "Business loans range from $5,000 to $500,000 for qualified businesses.",
        }.get(loan_type, "Please let me know the amount you're looking for.")

        formatted_prompt = loan_amount_prompt.format(
            loan_type=loan_type, loan_type_info=loan_type_info
        )

        logger.info(f"Loan amount collection for {loan_type}, amount: ${loan_amount}")

        return {
            "status": "success",
            "function_name": "loan_amount_collection",
            "prompt_guidance": formatted_prompt,
            "next_action": "verify_income" if loan_amount > 0 else "collect_amount",
            "loan_type": loan_type,
            "loan_amount": loan_amount,
            "context": {
                "stage": "amount_collection",
                "loan_type": loan_type,
                "loan_amount": loan_amount,
            },
        }

    def _func_income_verification(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle income verification for loan application.

        Args:
            arguments: Function arguments containing loan context and income

        Returns:
            Formatted prompt and guidance for income verification
        """
        from demo.demo_loan_application_prompts import income_verification_prompt

        loan_type = arguments.get("loan_type", "loan")
        loan_amount = arguments.get("loan_amount", 0)
        annual_income = arguments.get("annual_income", 0)

        formatted_prompt = income_verification_prompt.format(
            loan_type=loan_type, loan_amount=loan_amount
        )

        logger.info(f"Income verification for {loan_type}: ${annual_income}")

        return {
            "status": "success",
            "function_name": "income_verification",
            "prompt_guidance": formatted_prompt,
            "next_action": (
                "verify_employment" if annual_income > 0 else "collect_income"
            ),
            "context": {
                "stage": "income_verification",
                "loan_type": loan_type,
                "loan_amount": loan_amount,
                "annual_income": annual_income,
            },
        }

    def _func_employment_verification(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle employment verification for loan application.

        Args:
            arguments: Function arguments containing employment details

        Returns:
            Formatted prompt and guidance for employment verification
        """
        from demo.demo_loan_application_prompts import employment_verification_prompt

        employer = arguments.get("employer", "")
        employment_duration = arguments.get("employment_duration", "")
        job_title = arguments.get("job_title", "")

        logger.info(
            f"Employment verification: {employer}, {employment_duration}, {job_title}"
        )

        return {
            "status": "success",
            "function_name": "employment_verification",
            "prompt_guidance": employment_verification_prompt,
            "next_action": (
                "get_credit_consent"
                if all([employer, employment_duration, job_title])
                else "collect_employment"
            ),
            "context": {
                "stage": "employment_verification",
                "employer": employer,
                "employment_duration": employment_duration,
                "job_title": job_title,
            },
        }

    def _func_credit_check_consent(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle credit check consent for loan application.

        Args:
            arguments: Function arguments containing loan context and consent

        Returns:
            Formatted prompt and guidance for credit check consent
        """
        from demo.demo_loan_application_prompts import credit_check_consent_prompt

        loan_type = arguments.get("loan_type", "loan")
        consent_given = arguments.get("consent_given", False)

        formatted_prompt = credit_check_consent_prompt.format(loan_type=loan_type)

        logger.info(f"Credit check consent for {loan_type}: {consent_given}")

        return {
            "status": "success",
            "function_name": "credit_check_consent",
            "prompt_guidance": formatted_prompt,
            "next_action": "submit_application" if consent_given else "request_consent",
            "consent_given": consent_given,
            "context": {
                "stage": "credit_consent",
                "loan_type": loan_type,
                "consent_given": consent_given,
            },
        }

    def _func_submit_loan_application(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle loan application submission and summary.

        Args:
            arguments: Function arguments containing complete application data

        Returns:
            Formatted prompt and guidance for application submission
        """
        from demo.demo_loan_application_prompts import application_summary_prompt

        loan_type = arguments.get("loan_type", "loan")
        loan_amount = arguments.get("loan_amount", 0)
        annual_income = arguments.get("annual_income", 0)
        employer = arguments.get("employer", "")
        employment_duration = arguments.get("employment_duration", "")

        formatted_prompt = application_summary_prompt.format(
            loan_type=loan_type,
            loan_amount=loan_amount,
            annual_income=annual_income,
            employer=employer,
            employment_duration=employment_duration,
        )

        logger.info(f"Submitting loan application: {loan_type} for ${loan_amount}")

        return {
            "status": "success",
            "function_name": "submit_loan_application",
            "prompt_guidance": formatted_prompt,
            "next_action": "check_pre_approval",
            "application_submitted": True,
            "context": {
                "stage": "application_submitted",
                "loan_type": loan_type,
                "loan_amount": loan_amount,
                "annual_income": annual_income,
                "employer": employer,
                "employment_duration": employment_duration,
            },
        }

    def _func_loan_pre_approval(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle loan pre-approval notification.

        Args:
            arguments: Function arguments containing loan context

        Returns:
            Formatted prompt and guidance for pre-approval
        """
        from demo.demo_loan_application_prompts import loan_approval_prompt

        loan_type = arguments.get("loan_type", "loan")
        loan_amount = arguments.get("loan_amount", 0)
        reference_number = arguments.get(
            "reference_number", f"LA-{uuid.uuid4().hex[:8].upper()}"
        )

        formatted_prompt = loan_approval_prompt.format(
            loan_type=loan_type,
            loan_amount=loan_amount,
            reference_number=reference_number,
        )

        logger.info(f"Loan pre-approval for {loan_type}: {reference_number}")

        return {
            "status": "success",
            "function_name": "loan_pre_approval",
            "prompt_guidance": formatted_prompt,
            "next_action": "wrap_up",
            "pre_approved": True,
            "reference_number": reference_number,
            "context": {
                "stage": "pre_approved",
                "loan_type": loan_type,
                "loan_amount": loan_amount,
                "reference_number": reference_number,
            },
        }

    def _func_transfer_to_human(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
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
        logger.info(
            f"Transfer to human requested. Reason: {reason}, Priority: {priority}"
        )

        # Generate a transfer reference number
        transfer_id = f"TR-{uuid.uuid4().hex[:8].upper()}"

        # Format the transfer message
        transfer_message = (
            f"I understand you'd like to speak with a human agent regarding {reason}. "
            f"I'll transfer you now. Your reference number is {transfer_id}. "
            f"Please hold while I connect you with a representative."
        )

        return {
            "status": "success",
            "function_name": "transfer_to_human",
            "prompt_guidance": transfer_message,
            "next_action": "end_call",
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

    def _func_process_replacement(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle card replacement processing.

        Args:
            arguments: Function arguments containing card replacement details

        Returns:
            Formatted prompt and guidance for card replacement
        """
        card = arguments.get("card", "")
        reason = arguments.get("reason", "")
        address = arguments.get("address", "")

        logger.info(f"Processing card replacement for {card} with reason {reason} to address {address}")

        return {
            "status": "success",
            "function_name": "process_replacement",
            "prompt_guidance": f"Processing card replacement for {card} with reason {reason} to address {address}",
        }