# Chat Stream Client

Frontend integration guide for the chat API with **Claude-style inline citations**.

## Endpoints

| Endpoint        | Type        | Description                          |
|-----------------|-------------|--------------------------------------|
| **POST** `/chat`       | JSON        | Non-streaming; returns full response |
| **POST** `/chat/stream`| SSE stream  | Streaming tokens, sources, done      |

**Headers:** `Authorization: Bearer <token>`, `Content-Type: application/json`  
**Body:** `{ "content": "...", "session_id": "...", "history": [] }`

### Non-streaming response (POST /chat)

Same structure as `done` event:

```json
{
  "content": "Here's a roundup... Ganaplacide-Lumefantrine (GanLum) [1] announced...",
  "role": "assistant",
  "sources": [{ "url": "https://...", "title": "Malaria facts & statistics 2025 | MMV" }],
  "searched_web": true
}
```

## SSE Event Protocol

| Event     | Payload                                      | Description                          |
|-----------|-----------------------------------------------|--------------------------------------|
| `thinking`| `{ node, status, message }`                   | Progress (chat, browsing)             |
| `token`   | `{ text }`                                   | Streaming text chunk                 |
| `source`  | `{ url, title }`                             | Web source (emitted when discovered) |
| `ping`    | `{ ts }`                                     | Keep-alive (every ~15s when idle)    |
| `done`    | `{ content, role, sources?, searched_web? }` | Final response                      |
| `error`   | `{ error }`                                  | Error occurred                       |

### Done Event (with citations)

When web search was used, `done` includes:

```json
{
  "content": "Here's a roundup... Ganaplacide-Lumefantrine (GanLum) [1] announced...",
  "role": "assistant",
  "sources": [
    { "url": "https://...", "title": "Malaria facts & statistics 2025 | Medicines for Malaria Venture" }
  ],
  "searched_web": true
}
```

- **content**: Full text. May contain numbered markers `[1]`, `[2]`, `[3]` where each number refers to `sources[index]` (1-based: `[1]` → `sources[0]`).
- **sources**: Ordered list of `{ url, title }` in citation order.
- **searched_web**: `true` when sources exist; use to show "Searched the web >".

---

## Claude-Style Citation Display

Display links like Claude: inline blue clickable buttons with hover popup.

### 1. "Searched the web" Section

When `searched_web === true`, show an expandable section above the answer:

```
Searched the web >
```

### 2. Inline Citation Buttons

Replace each `[1]`, `[2]`, `[3]` in the content with a **CitationLink** component:

- **Visual**: Blue pill/button, truncated title (e.g. "Medicines for Malaria Vent."), external link icon (↗)
- **Hover**: Popup with full title and source name
- **Click**: Open URL in new tab

### 3. Parsing Content with Citations

Split content by citation markers and render segments + links:

```tsx
// Parse "text [1] more text [2] end" into segments
function parseContentWithCitations(
  content: string,
  sources: Array<{ url: string; title: string }>
): Array<{ type: 'text' | 'citation'; value: string; url?: string; title?: string }> {
  const segments: Array<{ type: 'text' | 'citation'; value: string; url?: string; title?: string }> = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(content)) !== null) {
    const num = parseInt(match[1], 10);
    if (num >= 1 && num <= sources.length) {
      segments.push({ type: 'text', value: content.slice(lastIndex, match.index) });
      const src = sources[num - 1];
      segments.push({
        type: 'citation',
        value: truncateTitle(src.title, 25),
        url: src.url,
        title: src.title,
      });
      lastIndex = regex.lastIndex;
    }
  }
  if (lastIndex < content.length) {
    segments.push({ type: 'text', value: content.slice(lastIndex) });
  }
  return segments;
}

function truncateTitle(title: string, maxLen: number): string {
  if (title.length <= maxLen) return title;
  return title.slice(0, maxLen - 3).trim() + '...';
}
```

### 4. CitationLink Component (React)

```tsx
interface CitationLinkProps {
  url: string;
  title: string;
  displayLabel: string;
}

function CitationLink({ url, title, displayLabel }: CitationLinkProps) {
  const [showPopup, setShowPopup] = useState(false);

  return (
    <span className="citation-wrapper" style={{ position: 'relative', display: 'inline' }}>
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="citation-link"
        onMouseEnter={() => setShowPopup(true)}
        onMouseLeave={() => setShowPopup(false)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '4px',
          padding: '2px 8px',
          borderRadius: '9999px',
          backgroundColor: '#2563eb',
          color: 'white',
          textDecoration: 'none',
          fontSize: '0.875rem',
          fontWeight: 500,
        }}
      >
        {displayLabel}
        <span aria-hidden>↗</span>
      </a>
      {showPopup && (
        <div
          className="citation-popup"
          style={{
            position: 'absolute',
            left: 0,
            bottom: '100%',
            marginBottom: '4px',
            padding: '8px 12px',
            backgroundColor: '#374151',
            color: 'white',
            borderRadius: '8px',
            fontSize: '0.8125rem',
            maxWidth: '320px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            zIndex: 10,
          }}
        >
          <div style={{ fontWeight: 500 }}>{title}</div>
          <div style={{ fontSize: '0.75rem', opacity: 0.9, marginTop: '2px' }}>
            {new URL(url).hostname}
          </div>
        </div>
      )}
    </span>
  );
}
```

### 5. Rendering the Answer

```tsx
function ChatAnswer({ content, sources }: { content: string; sources?: Array<{ url: string; title: string }> }) {
  const segments = sources?.length
    ? parseContentWithCitations(content, sources)
    : [{ type: 'text' as const, value: content }];

  return (
    <div className="chat-answer">
      {sources?.length ? (
        <details className="searched-web" style={{ marginBottom: '8px' }}>
          <summary style={{ cursor: 'pointer', color: '#6b7280', fontSize: '0.875rem' }}>
            Searched the web &gt;
          </summary>
          {/* Optional: list of sources */}
        </details>
      ) : null}
      <div className="answer-content">
        {segments.map((seg, i) =>
          seg.type === 'text' ? (
            <span key={i}>{seg.value}</span>
          ) : (
            <CitationLink
              key={i}
              url={seg.url!}
              title={seg.title!}
              displayLabel={seg.value}
            />
          )
        )}
      </div>
    </div>
  );
}
```

---

## Example: Fetch with ReadableStream

```javascript
async function streamChat(content, token, onEvent) {
  const res = await fetch('/chat/stream', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ content, session_id: crypto.randomUUID() }),
  });
  const reader = res.body.getReader();
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
        const [, event, data] = m;
        const payload = JSON.parse(data);
        onEvent(event, payload);
      }
    }
  }
}

// Usage
let fullContent = '';
const sources = [];
streamChat('Recent malaria treatment research', token, (event, data) => {
  if (event === 'token') fullContent += data.text;
  if (event === 'source') sources.push(data);
  if (event === 'done') {
    // Render with ChatAnswer(fullContent, data.sources)
  }
});
```

---

## Summary

| Backend provides        | Frontend renders                          |
|-------------------------|-------------------------------------------|
| `content` with `[1]`…   | Inline blue citation buttons              |
| `sources` array          | Link URL, full title in hover popup       |
| `searched_web: true`     | "Searched the web >" section              |
