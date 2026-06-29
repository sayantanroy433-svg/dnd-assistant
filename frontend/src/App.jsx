import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "👋 Hi! Ask me anything about Dungeons & Dragons."
    }
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // 👇 scroll anchor
  const messagesEndRef = useRef(null);

  // 👇 auto scroll on every update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage() {
    if (!input.trim() || loading) return;

    const question = input;

    setInput("");
    setLoading(true);

    // add user message
    setMessages((prev) => [
      ...prev,
      { role: "user", text: question }
    ]);

    try {
      const res = await axios.post(
        "https://dnd-assistant-nmxn.onrender.com/chat",
        { message: question }
      );

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: res.data.answer }
      ]);

    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "❌ Unable to reach the backend."
        }
      ]);
    }

    setLoading(false);
  }

  return (
    <div className="app">

      <div className="header">
        🐉 D&D Assistant
      </div>

      <div className="chat">

        {messages.map((m, i) => (
          <div
            key={i}
            className={m.role === "user" ? "user" : "assistant"}
          >
            <ReactMarkdown>{m.text}</ReactMarkdown>
          </div>
        ))}

        {loading && (
          <div className="assistant">
            Thinking...
          </div>
        )}

        {/* 👇 scroll target */}
        <div ref={messagesEndRef} />
      </div>

      <div className="inputBar">

        <input
          value={input}
          placeholder="Ask about D&D..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") sendMessage();
          }}
        />

        <button onClick={sendMessage}>
          Send
        </button>

      </div>

    </div>
  );
}

export default App;
