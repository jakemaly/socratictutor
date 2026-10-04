import { useState } from "react";
import {
  AuiIf,
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";

export default function App() {
  const [session, setSession] = useState(0);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="SocraticTutor home">
          SocraticTutor
        </a>
        <button className="new-chat-button" onClick={() => setSession((value) => value + 1)}>
          <span aria-hidden="true">＋</span> New chat
        </button>
      </header>
      <ChatSession key={session} />
    </main>
  );
}

function ChatSession() {
  const [error, setError] = useState<string | null>(null);
  const runtime = useDataStreamRuntime({
    api: "/api/chat",
    protocol: "data-stream",
    onResponse: (response) => {
      if (response.ok) setError(null);
    },
    onError: (cause) => setError(cause.message),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatContent error={error} clearError={() => setError(null)} />
    </AssistantRuntimeProvider>
  );
}

function ChatContent({ error, clearError }: { error: string | null; clearError: () => void }) {
  const aui = useAui();
  const isRunning = useAuiState((state) => state.thread.isRunning);

  return (
    <section className="chat-panel" aria-label="Chat">
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button className="error-dismiss" onClick={clearError} aria-label="Dismiss error">×</button>
        </div>
      )}

      <ThreadPrimitive.Root className="thread-root">
        <AuiIf condition={(state) => state.thread.isEmpty}>
          <div className="empty-state">
            <div className="empty-state-content">
              <h1>Where should we begin?</h1>
              <Composer isRunning={isRunning} stop={() => aui.thread().cancelRun()} />
              <p className="composer-disclaimer">SocraticTutor can make mistakes. Check important information.</p>
            </div>
          </div>
        </AuiIf>

        <AuiIf condition={(state) => !state.thread.isEmpty}>
          <ThreadPrimitive.Viewport className="thread-viewport" autoScroll>
            <div className="message-list">
              <ThreadPrimitive.Messages>
                {({ message }) => message.role === "user" ? <UserMessage /> : <AssistantMessage />}
              </ThreadPrimitive.Messages>
            </div>

            <ThreadPrimitive.ViewportFooter className="thread-footer">
              <Composer isRunning={isRunning} stop={() => aui.thread().cancelRun()} />
              <p className="composer-disclaimer">SocraticTutor can make mistakes. Check important information.</p>
            </ThreadPrimitive.ViewportFooter>
          </ThreadPrimitive.Viewport>
        </AuiIf>
      </ThreadPrimitive.Root>
    </section>
  );
}

function Composer({ isRunning, stop }: { isRunning: boolean; stop: () => void }) {
  return (
    <ComposerPrimitive.Root className="composer">
      <ComposerPrimitive.Input
        aria-label="Message"
        placeholder="Message SocraticTutor"
        className="composer-input"
        rows={1}
        submitMode="enter"
      />
      <div className="composer-actions">
        {isRunning ? (
          <button
            className="send-button stop-button"
            type="button"
            onClick={stop}
            aria-label="Stop response"
          >
            <span className="stop-icon" aria-hidden="true" />
          </button>
        ) : (
          <ComposerPrimitive.Send className="send-button" aria-label="Send message">
            <span aria-hidden="true">↑</span>
          </ComposerPrimitive.Send>
        )}
      </div>
    </ComposerPrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="message-row user-row">
      <div className="user-bubble"><MessagePrimitive.Parts /></div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="message-row assistant-row">
      <div className="assistant-content">
        <MessagePrimitive.Parts
          components={{ Text: () => <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} /> }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}
