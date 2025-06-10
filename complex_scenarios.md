# Complex Card Replacement Flow Test Scenarios

## Test Cases for Enhanced Context Handling

### Scenario 1: Full Context Upfront
**Customer Input:** "Hi, I lost my gold card and need a replacement"
**Expected Behavior:**
- LLM calls `call_intent` with: `intent="card_replacement", card_type="Gold card", replacement_reason="Lost"`
- Should skip to address confirmation step
- Should directly ask: "Could you please confirm the address where you would like the new Gold card to be delivered?"

### Scenario 2: Partial Context - Card Type Only
**Customer Input:** "I need to replace my silver card"
**Expected Behavior:**
- LLM calls `call_intent` with: `intent="card_replacement", card_type="Silver card"`
- Should ask for reason directly: "Could you please let me know the reason for replacing your Silver card? Is it Lost, Damaged, Stolen, or Other?"
- Should NOT ask which card type

### Scenario 3: Partial Context - Reason Only
**Customer Input:** "My card was stolen and I need a new one"
**Expected Behavior:**
- LLM calls `call_intent` with: `intent="card_replacement", replacement_reason="Stolen"`
- Should ask for card type directly: "Which type of card do you need to replace? Your options are Gold card, Silver card, or Basic card."
- Should NOT ask for reason again

### Scenario 4: Complex Context with Extra Details
**Customer Input:** "Hi, my gold card was damaged yesterday when I dropped it in water, can you help me get a new one sent to my home address?"
**Expected Behavior:**
- LLM calls `call_intent` with: `intent="card_replacement", card_type="Gold card", replacement_reason="Damaged", additional_context="dropped in water yesterday, wants it sent to home"`
- Should proceed directly to address confirmation
- Should ask: "Could you please confirm the address where you would like the new Gold card to be delivered?"

### Scenario 5: Multiple Cards Mentioned
**Customer Input:** "I lost both my gold and silver cards, but I only need to replace the gold one right now"
**Expected Behavior:**
- LLM calls `call_intent` with: `intent="card_replacement", card_type="Gold card", replacement_reason="Lost", additional_context="also lost silver card but only replacing gold"`
- Should focus on gold card replacement
- Should proceed directly to address confirmation

## How to Test

1. **Run the enhanced flow** with these customer inputs
2. **Monitor the logs** to see which functions are called and with what parameters
3. **Check the conversation flow** to ensure no redundant questions are asked
4. **Verify the LLM follows** the `prompt_guidance` from function responses
5. **Ensure no verbal acknowledgments** - LLM should move directly to next required step

## Expected Improvements

- **Reduced conversation turns** for customers who provide complete information upfront
- **Direct, efficient conversations** that proceed to next required step
- **Intelligent flow skipping** based on captured information
- **Better customer experience** with streamlined flow and no redundant acknowledgments

## Potential Issues to Watch For

- **Over-extraction:** LLM might extract information that wasn't clearly stated
- **Under-extraction:** LLM might miss obvious context
- **Flow confusion:** LLM might get confused about which step to take next
- **Context loss:** Later functions might not have access to earlier captured context
- **Unexpected acknowledgments:** LLM might still acknowledge despite instructions not to 