import { useState } from "react";
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

async function sendMessage() {
  if (!input.trim() || loading) return;

  setLoading(true); // lock immediately

  const question = input;
  setInput("");

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

      </div>

      <div className="inputBar">

        <input
          value={input}
          placeholder="Ask about D&D..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter")
              sendMessage();
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
