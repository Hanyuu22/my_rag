import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Send, BookOpen, ChevronDown, ChevronUp, Trash2, Eraser } from 'lucide-react'
import type { Settings } from '../hooks/useSettings'

interface Source { index: number; score: number; page: string | number; content: string }
export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  route?: string
  fallbackType?: { type: 'llm' | 'web'; web_sources?: { title: string; url: string }[] }
  loading?: boolean
}

interface Props { collection: string; extraCollections?: string[]; settings: Settings }

/* ── 历史记录工具函数 ── */
const HISTORY_KEY = 'rag_chat_history'

function loadHistory(collection: string): Message[] {
  if (!collection) return []
  try {
    const all = JSON.parse(localStorage.getItem(HISTORY_KEY) || '{}')
    return (all[collection] || []).map((m: Message) => ({ ...m, loading: false }))
  } catch { return [] }
}

function saveHistory(collection: string, messages: Message[]) {
  if (!collection) return
  try {
    const all = JSON.parse(localStorage.getItem(HISTORY_KEY) || '{}')
    all[collection] = messages.filter(m => !m.loading)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(all))
  } catch {}
}

function clearHistory(collection: string) {
  try {
    const all = JSON.parse(localStorage.getItem(HISTORY_KEY) || '{}')
    delete all[collection]
    localStorage.setItem(HISTORY_KEY, JSON.stringify(all))
  } catch {}
}

/* ── 主组件 ── */
export default function ChatWindow({ collection, extraCollections = [], settings }: Props) {
  const [messages, setMessages] = useState<Message[]>(() => loadHistory(collection))
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const [hoveredMsg, setHoveredMsg] = useState<number | null>(null)
  const [crossLingualEnabled, setCrossLingualEnabled] = useState(
    () => localStorage.getItem('cross_lingual_enabled') === 'true'
  )
  const bottomRef = useRef<HTMLDivElement>(null)
  const msgId = useRef(0)

  /* 切换知识库：加载对应知识库的历史（不在这里保存，避免竞争） */
  useEffect(() => {
    const hist = loadHistory(collection)
    setMessages(hist)
    msgId.current = hist.length > 0 ? Math.max(...hist.map(m => m.id)) + 1 : 0
    setExpandedSources(new Set())
  }, [collection])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || loading || !collection) return
    const question = input.trim()
    setInput('')
    setLoading(true)

    const userMsg: Message = { id: ++msgId.current, role: 'user', content: question }
    const botId = ++msgId.current
    const botMsg: Message = { id: botId, role: 'assistant', content: '', loading: true }
    setMessages(prev => [...prev, userMsg, botMsg])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question, collection,
          extra_collections: extraCollections,
          retriever_top_k: settings.retrieverTopK,
          reranker_top_n: settings.rerankerTopN,
          score_threshold: settings.scoreThreshold,
          max_retry: settings.maxRetry,
          temperature: settings.temperature,
          reranker_enabled: settings.rerankerEnabled,
          fallback_enabled: settings.fallbackEnabled,
          fallback_method: settings.fallbackMethod,
          cross_lingual_enabled: crossLingualEnabled,
        }),
      })

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let hasStreamedTokens = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            if (event.type === 'answer_token') {
              hasStreamedTokens = true
              setMessages(prev => prev.map(m =>
                m.id === botId
                  ? { ...m, content: (m.loading ? '' : m.content) + event.content, loading: false }
                  : m
              ))
            } else if (event.type === 'answer') {
              // 非流式客户端兜底：只在没有收到 token 时使用
              if (!hasStreamedTokens) {
                setMessages(prev => prev.map(m =>
                  m.id === botId ? { ...m, content: event.content, loading: false } : m
                ))
              }
            } else if (event.type === 'fallback') {
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, fallbackType: event.content } : m
              ))
            } else if (event.type === 'sources') {
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, sources: event.content } : m
              ))
            } else if (event.type === 'status') {
              // 仅在没有真实内容时才显示状态文字（避免覆盖已流式输出的答案）
              setMessages(prev => prev.map(m =>
                m.id === botId && !hasStreamedTokens
                  ? { ...m, content: m.content || `_${event.content}_`, loading: true }
                  : m
              ))
            } else if (event.type === 'done') {
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, loading: false } : m
              ))
            } else if (event.type === 'error') {
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, content: `⚠️ ${event.content}`, loading: false } : m
              ))
            }
          } catch {}
        }
      }
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === botId ? { ...m, content: `⚠️ 请求失败：${e}`, loading: false } : m
      ))
    }

    /* 显式保存：stream 结束后保存最终消息（此时 collection 仍是本次对话的知识库） */
    setMessages(prev => {
      const final = prev.filter(m => !m.loading)
      saveHistory(collection, final)
      return prev
    })
    setLoading(false)
  }

  const deleteMessage = useCallback((id: number) => {
    setMessages(prev => {
      // 删除该消息及其配对消息（用户问题配 bot 回答）
      const idx = prev.findIndex(m => m.id === id)
      if (idx === -1) return prev
      const msg = prev[idx]
      let toRemove = new Set([id])
      // 如果删的是 user 消息，同时删下一条 assistant；反之亦然
      if (msg.role === 'user' && prev[idx + 1]?.role === 'assistant') {
        toRemove.add(prev[idx + 1].id)
      } else if (msg.role === 'assistant' && prev[idx - 1]?.role === 'user') {
        toRemove.add(prev[idx - 1].id)
      }
      const next = prev.filter(m => !toRemove.has(m.id))
      saveHistory(collection, next)
      return next
    })
  }, [collection])

  const handleClearAll = () => {
    if (!confirm(`确认清空「${collection}」的全部对话记录？`)) return
    setMessages([])
    clearHistory(collection)
  }

  const toggleSources = (id: number) => {
    setExpandedSources(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  if (!collection) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: 'var(--text-dim)' }}>
          <BookOpen size={40} style={{ margin: '0 auto 12px', opacity: 0.35 }} />
          <div style={{ fontSize: 14 }}>请先从左侧选择一个知识库</div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>

      {/* 消息区顶部工具栏 */}
      {messages.length > 0 && (
        <div style={{
          display: 'flex', justifyContent: 'flex-end', marginBottom: 6, flexShrink: 0,
        }}>
          <button
            onClick={handleClearAll}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-dim)', fontSize: 11, padding: '2px 6px',
              borderRadius: 6, transition: 'color 0.15s',
            }}
            onMouseEnter={e => (e.currentTarget.style.color = 'var(--danger)')}
            onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
          >
            <Eraser size={12} /> 清空记录
          </button>
        </div>
      )}

      {/* 消息区 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 4px 16px', display: 'flex', flexDirection: 'column', gap: 20 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-dim)', paddingTop: 60 }}>
            <div style={{
              width: 76, height: 76, borderRadius: '50%', margin: '0 auto 16px',
              overflow: 'hidden', opacity: 0.75,
              border: '2px solid rgba(100,180,80,0.25)',
              boxShadow: '0 4px 16px rgba(50,120,40,0.1)',
            }}>
              {settings.botAvatar.startsWith('/')
                ? <img src={settings.botAvatar} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                : <div style={{ width: '100%', height: '100%', display: 'flex',
                    alignItems: 'center', justifyContent: 'center', fontSize: 42 }}>{settings.botAvatar}</div>}
            </div>
            <div style={{ fontSize: 14, fontWeight: 500 }}>
              当前知识库：<span style={{ color: 'var(--green-mid)' }}>{collection}</span>
            </div>
            <div style={{ fontSize: 12, marginTop: 6 }}>输入问题开始对话</div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{ display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}
            onMouseEnter={() => setHoveredMsg(msg.id)}
            onMouseLeave={() => setHoveredMsg(null)}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10,
              flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', maxWidth: '86%' }}>

              {/* 头像 */}
              <div style={{
                width: 46, height: 46, borderRadius: '50%', flexShrink: 0,
                border: '1.5px solid rgba(100,180,80,0.25)',
                overflow: 'hidden', background: 'rgba(255,255,255,0.7)',
                boxShadow: '0 2px 8px rgba(50,120,40,0.1)',
              }}>
                {(() => {
                  const av = msg.role === 'user' ? settings.userAvatar : settings.botAvatar
                  return av.startsWith('/')
                    ? <img src={av} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    : <div style={{ width: '100%', height: '100%', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>{av}</div>
                })()}
              </div>

              {/* 气泡 */}
              <div style={{
                padding: '11px 15px',
                borderRadius: msg.role === 'user' ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
                background: msg.role === 'user'
                  ? 'linear-gradient(135deg, rgba(90,170,76,0.2), rgba(60,140,50,0.13))'
                  : 'rgba(255,255,255,0.88)',
                border: msg.role === 'user'
                  ? '1px solid rgba(90,170,76,0.32)'
                  : '1px solid rgba(100,180,80,0.18)',
                fontSize: 14, lineHeight: 1.7, color: 'var(--text-primary)',
                backdropFilter: 'blur(8px)',
                wordBreak: 'break-word',
                boxShadow: '0 1px 4px rgba(50,120,40,0.06)',
              }}>
                {msg.loading && !msg.content
                  ? <LoadingDots />
                  : msg.role === 'assistant'
                    ? <div className="md-content">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >{msg.content}</ReactMarkdown>
                      </div>
                    : <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
                }
              </div>

              {/* 删除按钮（hover 显示） */}
              {!msg.loading && hoveredMsg === msg.id && (
                <button
                  onClick={() => deleteMessage(msg.id)}
                  title="删除这对对话"
                  style={{
                    alignSelf: 'center', background: 'none', border: 'none',
                    cursor: 'pointer', color: 'var(--text-dim)', padding: 4,
                    borderRadius: 6, flexShrink: 0, transition: 'color 0.15s',
                    opacity: 0.7,
                  }}
                  onMouseEnter={e => (e.currentTarget.style.color = 'var(--danger)')}
                  onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
                >
                  <Trash2 size={13} />
                </button>
              )}
            </div>

            {/* Fallback 来源标注 */}
            {msg.fallbackType && (
              <div style={{
                maxWidth: '86%', marginLeft: 56, marginTop: 5,
                padding: '6px 12px', borderRadius: 8,
                background: msg.fallbackType.type === 'web'
                  ? 'rgba(50,150,255,0.07)' : 'rgba(255,200,50,0.10)',
                border: `1px solid ${msg.fallbackType.type === 'web'
                  ? 'rgba(50,130,220,0.25)' : 'rgba(200,150,30,0.25)'}`,
                fontSize: 11,
                color: msg.fallbackType.type === 'web'
                  ? 'rgba(30,100,200,0.9)' : 'rgba(160,110,20,0.9)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span>{msg.fallbackType.type === 'web' ? '🌐' : '⚠️'}</span>
                  <span>
                    {msg.fallbackType.type === 'web'
                      ? <>以下内容来自<strong>网络搜索</strong>，并非知识库文档内容，请注意甄别</>
                      : <>以下内容来自<strong>大语言模型自身知识</strong>，并非知识库文档内容，仅供参考</>
                    }
                  </span>
                </div>
                {msg.fallbackType.type === 'web' && msg.fallbackType.web_sources?.length ? (
                  <div style={{ marginTop: 5, display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {msg.fallbackType.web_sources.map((s, i) => (
                      <a key={i} href={s.url} target="_blank" rel="noopener noreferrer"
                        style={{ color: 'rgba(30,100,200,0.85)', textDecoration: 'none',
                          display: 'flex', alignItems: 'center', gap: 4 }}
                        onMouseEnter={e => (e.currentTarget.style.textDecoration = 'underline')}
                        onMouseLeave={e => (e.currentTarget.style.textDecoration = 'none')}
                      >
                        <span style={{ opacity: 0.6 }}>[{i+1}]</span>
                        <span>{s.title || s.url}</span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            )}

            {/* 来源 */}
            {msg.sources && msg.sources.length > 0 && (
              <div style={{ maxWidth: '86%', marginLeft: 56, marginTop: 6 }}>
                <button
                  onClick={() => toggleSources(msg.id)}
                  style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'none',
                    border: 'none', cursor: 'pointer', color: 'var(--text-dim)', fontSize: 12, padding: 0 }}>
                  <BookOpen size={11} />
                  来源 ({msg.sources.length})
                  {expandedSources.has(msg.id) ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                </button>
                {expandedSources.has(msg.id) && (
                  <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {msg.sources.map(src => (
                      <div key={src.index} style={{
                        padding: '7px 11px', borderRadius: 9,
                        background: 'rgba(240,250,236,0.9)', border: '1px solid rgba(100,180,80,0.18)',
                        fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5,
                      }}>
                        <span className="badge badge-green" style={{ marginRight: 6 }}>
                          p.{src.page}  {(src.score * 100).toFixed(0)}%
                        </span>
                        {src.content}...
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* 输入区 */}
      <div style={{ padding: '14px 0 0', borderTop: '1px solid rgba(100,180,80,0.12)', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={`向「${collection}」提问...`}
            rows={2}
            style={{ resize: 'none', lineHeight: 1.55, flex: 1 }}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
            }}
          />
          <button
            className="btn-primary"
            style={{ padding: '10px 18px', flexShrink: 0, height: 58 }}
            onClick={sendMessage}
            disabled={loading || !input.trim()}
          >
            <Send size={16} />
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 5 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            Enter 发送 · Shift+Enter 换行
          </div>
          <button
            onClick={() => setCrossLingualEnabled(v => {
              const next = !v
              localStorage.setItem('cross_lingual_enabled', String(next))
              return next
            })}
            title="开启后将同时检索配对的中/英文知识库（如 energy_zh ↔ energy_en）"
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              background: crossLingualEnabled ? 'rgba(90,170,76,0.15)' : 'none',
              border: crossLingualEnabled ? '1px solid rgba(90,170,76,0.45)' : '1px solid rgba(100,180,80,0.2)',
              borderRadius: 8, padding: '3px 9px', cursor: 'pointer',
              fontSize: 11, color: crossLingualEnabled ? 'var(--green-mid)' : 'var(--text-dim)',
              transition: 'all 0.15s', flexShrink: 0,
            }}
          >
            <span style={{ fontSize: 13 }}>🌐</span>
            双语检索 {crossLingualEnabled ? '已开启' : '已关闭'}
          </button>
        </div>
      </div>
    </div>
  )
}

function LoadingDots() {
  return (
    <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center', padding: '2px 0' }}>
      {[0, 1, 2].map(i => (
        <span key={i} style={{
          width: 6, height: 6, borderRadius: '50%', background: 'var(--green-soft)',
          animation: `dotBounce 1.2s ${i * 0.2}s infinite ease-in-out`,
          display: 'inline-block',
        }} />
      ))}
      <style>{`
        @keyframes dotBounce {
          0%,80%,100% { transform: scale(0.6); opacity: 0.4; }
          40% { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </span>
  )
}
