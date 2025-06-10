```mermaid
sequenceDiagram
    participant AC as Telephony Server
    participant TB as OpusBridge Server
    participant OAI as Realtime Voice Server

    %% Session Initiation
    AC->>TB: session.initiate (conversationId, mediaFormat)
    TB->>TB: Initialize CallRecorder
    TB->>AC: session.accepted (conversationId, mediaFormat)
    TB->>OAI: session.update (config, tools, instructions)
    OAI->>TB: session.updated
    TB->>OAI: conversation.item.create (initial greeting)
    OAI->>TB: conversation.item.created
    TB->>OAI: response.create
    OAI->>TB: response.created (response_active = true)
    
    %% Audio Streaming Flow
    AC->>TB: userStream.start
    TB->>AC: userStream.started
    loop Audio Chunks
        AC->>TB: userStream.chunk (audioChunk base64)
        TB->>TB: Validate chunk size (min 3200 bytes)
        TB->>TB: Record caller audio
        TB->>OAI: input_audio_buffer.append (audio)
    end
    AC->>TB: userStream.stop
    TB->>OAI: input_audio_buffer.commit
    TB->>AC: userStream.stopped
    
    %% Speech Detection & Transcription
    OAI->>TB: input_audio_buffer.speech_started
    OAI->>TB: input_audio_buffer.speech_stopped
    OAI->>TB: conversation.item.input_audio_transcription.delta
    OAI->>TB: conversation.item.input_audio_transcription.completed
    TB->>TB: Record caller transcript
    
    %% AI Response Generation
    TB->>OAI: response.create (if not response_active)
    OAI->>TB: response.created
    OAI->>TB: response.output_item.added
    
    %% Function Call Handling (if triggered)
    opt Function Call Flow
        OAI->>TB: response.function_call_arguments.delta
        TB->>TB: Accumulate function arguments
        OAI->>TB: response.function_call_arguments.done
        TB->>TB: Execute function via FunctionHandler
        TB->>OAI: conversation.item.create (function_call_output)
        TB->>OAI: response.create
    end
    
    %% Audio Response Flow
    TB->>AC: playStream.start (streamId, mediaFormat)
    loop Audio Response
        OAI->>TB: response.audio.delta
        TB->>TB: Record bot audio
        TB->>AC: playStream.chunk (audioChunk base64)
    end
    OAI->>TB: response.audio.done
    TB->>AC: playStream.stop
    
    %% Transcript Handling
    OAI->>TB: response.audio_transcript.delta
    OAI->>TB: response.audio_transcript.done
    TB->>TB: Record bot transcript
    
    %% Response Completion
    OAI->>TB: response.done (response_active = false)
    TB->>TB: Check for pending user input
    opt Pending Input Exists
        TB->>OAI: response.create
    end
    
    %% Session Termination
    AC->>TB: session.end
    TB->>TB: Stop CallRecorder and finalize
    TB->>TB: Close connections
```

---

## Additional Technical Details

### Audio Specifications
- **Format**: PCM16, 16kHz, mono
- **Minimum chunk**: 3200 bytes (100ms)
- **Encoding**: Base64 for transport
- **Validation**: Auto-padding for undersized chunks

### Function Registry (12 Banking Functions)
- `get_balance` - Account balance lookup
- `transfer_funds` - Money transfers  
- `call_intent` - Intent classification
- `member_account_confirmation` - Card account selection
- `replacement_reason` - Card replacement reasons
- `confirm_address` - Address verification
- `start_card_replacement` - Begin replacement process
- `finish_card_replacement` - Complete replacement
- `wrap_up` - Call conclusion
- `loan_type_selection` - Loan product selection
- `income_verification` - Income validation
- `transfer_to_human` - Escalation to human agent

### State Management
- **Response Race Prevention**: `response_active` flag + `pending_user_input` queue
- **Stream Tracking**: `active_stream_id` for audio flow control
- **Buffer Management**: Separate transcript buffers for input/output
- **Function Call State**: `active_function_calls` dictionary with incremental argument building

### Recording & Monitoring
- **CallRecorder**: Persistent audio + transcript storage
- **Channels**: CALLER vs BOT audio separation
- **Formats**: Both raw audio (base64) and text transcripts
- **Metadata**: Timestamps, conversation_id, session_id

### Error Recovery
- **Quota Exhaustion**: Graceful session termination with detailed logging
- **Connection Loss**: Automatic cleanup and resource deallocation  
- **Invalid Audio**: Chunk validation and padding
- **Function Failures**: Error logging while maintaining conversation flow
