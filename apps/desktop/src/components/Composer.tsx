import { useEffect, useRef, useState } from "react";

interface Props {
  disabled: boolean;
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ disabled, busy, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || busy || disabled) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <div className="composer">
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder={disabled ? "Select a model to start chatting" : "Message Higgs…  (Enter to send, Shift+Enter for newline)"}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
      />
      {busy ? (
        <button type="button" className="btn stop" onClick={onStop} title="Stop generation">
          Stop
        </button>
      ) : (
        <button type="button" className="btn send" onClick={submit} disabled={disabled || !text.trim()}>
          Send
        </button>
      )}
    </div>
  );
}
