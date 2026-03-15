# Biomni Stream Client

Frontend integration guide for Biomni SSE streaming and WebSocket.

**Aligned with agent/run:** Biomni uses the same input pattern: `query`, `session_id`, `history`, `options`. `user_id` is always from the Bearer token (never from request body).

## SSE Event Protocol

**Endpoint:** `POST /biomni/stream`  
**Headers:** `Authorization: Bearer <token>`, `Content-Type: application/json`  
**Body:** `{ "query": "...", "session_id": "...", "history": [], "options": {} }` (or use `prompt` instead of `query`)

### Event Types

| Event | Payload | Description |
|-------|---------|-------------|
| `started` | `{ prompt, session_id, run_id }` | Task started |
| `step` | `{ index, content }` | Intermediate execution step |
| `clarify` | `{ needs_clarification, question, options }` | Agent asks for clarification |
| `result` | `{ answer, steps, success, artifact_url, pdf_url, error? }` | Final result |
| `error` | `{ error }` | Error occurred |
| `ping` | `{ ts }` | Keep-alive |
| `done` | `{}` | Stream complete |

### Request Schema (aligned with agent/run)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes* | User query (same as agent/run) |
| `prompt` | string | Yes* | Alias for query (backwards compat). Use query or prompt. |
| `user_id` | string | No | **Ignored.** Always from Bearer token (auth). Do not send. |
| `session_id` | string | No | Session ID for tracking |
| `history` | array | No | Previous messages: `[{role, content}]` |
| `options` | object | No | `skip_clarify`, `max_clarify_rounds`, `generate_pdf` |

*Either `query` or `prompt` is required. `user_id` is always fetched from auth (JWT), never from request body.

### Example: Fetch with ReadableStream

```javascript
async function streamBiomni(query, token) {
  const res = await fetch('/biomni/stream', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, session_id: crypto.randomUUID() }),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';
    for (const block of lines) {
      const match = block.match(/^event: (\w+)\ndata: (.+)$/s);
      if (match) {
        const [, event, data] = match;
        const payload = JSON.parse(data);
        if (event === 'clarify') {
          // Show clarification UI, collect answer, retry with history
          return { type: 'clarify', ...payload };
        }
        if (event === 'result') {
          return { type: 'result', ...payload };
        }
      }
    }
  }
}
```

### React Hook: useBiomniStream

```tsx
import { useState, useCallback } from 'react';

type BiomniEvent =
  | { type: 'started'; prompt: string; run_id: string }
  | { type: 'step'; index: number; content: string }
  | { type: 'clarify'; question: string; options?: string[] }
  | { type: 'result'; answer: string; steps: string[]; success: boolean; pdf_url?: string }
  | { type: 'error'; error: string }
  | { type: 'done' };

export function useBiomniStream(token: string) {
  const [status, setStatus] = useState<'idle' | 'streaming' | 'done' | 'error'>('idle');
  const [steps, setSteps] = useState<string[]>([]);
  const [result, setResult] = useState<string | null>(null);
  const [clarify, setClarify] = useState<{ question: string; options?: string[] } | null>(null);

  const run = useCallback(async (query: string, history: Array<{role: string; content: string}> = []) => {
    setStatus('streaming');
    setSteps([]);
    setResult(null);
    setClarify(null);
    const res = await fetch('/biomni/stream', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, history }),
    });
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';
      for (const block of blocks) {
        const m = block.match(/^event: (\w+)\ndata: (.+)$/s);
        if (m) {
          const [, ev, data] = m;
          const payload = JSON.parse(data);
          if (ev === 'step') setSteps((s) => [...s, payload.content]);
          if (ev === 'clarify') setClarify({ question: payload.question, options: payload.options });
          if (ev === 'result') setResult(payload.answer);
          if (ev === 'error') setStatus('error');
        }
      }
    }
    setStatus('done');
  }, [token]);

  return { run, status, steps, result, clarify };
}
```

## WebSocket Protocol

**Endpoint:** `WS /biomni/ws?token=JWT` or connect then send `{"type": "auth", "token": "..."}`

### Client Messages

| Type | Payload | Description |
|------|---------|-------------|
| `auth` | `{ token }` | First message if not using query param |
| `run` | `{ query, prompt?, session_id?, max_retries?, history?, options? }` | Execute task. Use `query` or `prompt` (aligned with agent/run). `user_id` from token. |
| `clarify_response` | `{ answer }` | Response to clarify (after receiving clarify) |
| `ping` | `{}` | Keep-alive |

### Server Messages

| Type | Payload | Description |
|------|---------|-------------|
| `connected` | `{ session_id, user_id }` | Connection accepted |
| `started` | `{ run_id }` | Task started |
| `step` | `{ index, content }` | Execution step |
| `clarify` | `{ question, options? }` | Needs clarification |
| `result` | `{ answer, steps, success, artifact_url? }` | Final result |
| `done` | `{}` | Task complete |
| `error` | `{ error }` | Error |
| `pong` | `{}` | Response to ping |

### Example: WebSocket Client

```javascript
const ws = new WebSocket(`wss://api.example.com/biomni/ws?token=${token}`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'clarify') {
    const answer = prompt(msg.question);
    ws.send(JSON.stringify({ type: 'clarify_response', answer }));
  }
  if (msg.type === 'result') {
    console.log('Result:', msg.answer);
  }
};
ws.onopen = () => {
  ws.send(JSON.stringify({ type: 'run', query: 'Predict ADMET for aspirin' }));
};
```

## Clarifying Questions Flow

1. User sends query (or prompt).
2. If agent needs clarification, you receive `clarify` (SSE) or `{"type": "clarify"}` (WS).
3. **SSE:** Show the question in UI, user answers, make a new `/biomni/stream` request with `history: [{role: "assistant", content: question}, {role: "user", content: answer}]`.
4. **WS:** Show the question, user answers, send `{"type": "clarify_response", "answer": "..."}` on the same connection. Agent continues automatically.
