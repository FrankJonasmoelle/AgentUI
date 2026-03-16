import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import './index.css'


type Message = {
  role: 'user' | 'agent'
  text: string
}

function App() {

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage() {

    if (!input.trim()) return // if the user typed nothing (or only spaces), stop here and don't send anything.

    // add message to the list
    const userMsg: Message = { role: 'user', text: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')

    // add empty agent bubble
    setMessages(prev => [...prev, { role: 'agent', text: '' }])

    // call backend
    const response = await fetch('http://localhost:8080/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input, thread_id: 'my-thread' }),
    })

    // read stream chunk by chunk 
    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = decoder.decode(value)
      // Each chunk looks like:  data: {"text": "Hello"}\n\n
      const lines = chunk.split('\n').filter(l => l.startsWith('data: '))
      for (const line of lines) {
        const json = JSON.parse(line.replace('data: ', ''))
        // Append the new text to the last message (the agent bubble)
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1].text += json.text
          return updated
        })
      }
    }
  }

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`bubble ${msg.role}`}>
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {msg.text}
            </ReactMarkdown>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="input-bar">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') sendMessage()
          }}
          placeholder="How can the agent help you? :)"
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div >
  )
}


export default App