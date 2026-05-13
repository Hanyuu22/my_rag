import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Send, BookOpen, ChevronDown, ChevronUp, Trash2, Eraser } from 'lucide-react'
import type { Settings } from '../hooks/useSettings'
import type { ChatSession } from './ChatSessionList'
import KBSelector from './KBSelector'

interface Source { index: number; score: number; page: string | number; content: string }
export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  route?: string
  fallbackType?: { type: 'llm' | 'web'; web_sources?: { title: string; url: string }[] }
  subQueries?: string[]
  loading?: boolean
}

interface Props {
  session: ChatSession
  settings: Settings
  onSessionUpdate: (updates: Partial<ChatSession>) => void
  refreshTrigger?: number
}

/* ── 历史记录（按 sessionId 存储）── */
function loadHistory(sessionId: string): Message[] {
  if (!sessionId) return []
  try {
    const raw = localStorage.getItem(`rag_chat_history_${sessionId}`)
    return raw ? JSON.parse(raw).map((m: Message) => ({ ...m, loading: false })) : []
  } catch { return [] }
}

function saveHistory(sessionId: string, messages: Message[]) {
  if (!sessionId) return
  try {
    localStorage.setItem(`rag_chat_history_${sessionId}`, JSON.stringify(messages.filter(m => !m.loading)))
  } catch {}
}

function clearHistory(sessionId: string) {
  localStorage.removeItem(`rag_chat_history_${sessionId}`)
}

/* ── 主组件 ── */
export default function ChatWindow({ session, settings, onSessionUpdate, refreshTrigger }: Props) {
  const [messages, setMessages] = useState<Message[]>(() => loadHistory(session.id))
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const [hoveredMsg, setHoveredMsg] = useState<number | null>(null)
  const [crossLingualEnabled, setCrossLingualEnabled] = useState(
    () => localStorage.getItem('cross_lingual_enabled') === 'true'
  )
  const [decomposeEnabled, setDecomposeEnabled] = useState(
    () => localStorage.getItem('decompose_enabled') !== 'false'
  )
  const [raptorEnabled, setRaptorEnabled] = useState(
    () => localStorage.getItem('raptor_enabled') === 'true'
  )
  const [kgEnabled, setKgEnabled] = useState(
    () => localStorage.getItem('kg_enabled') === 'true'
  )
  const [graphragEnabled, setGraphragEnabled] = useState(
    () => localStorage.getItem('graphrag_enabled') === 'true'
  )
  const [parentChildEnabled, setParentChildEnabled] = useState(
    () => localStorage.getItem('parent_child_enabled') === 'true'
  )
  const bottomRef = useRef<HTMLDivElement>(null)
  const msgId = useRef(0)

  /* 切换 session 时加载对应历史 */
  useEffect(() => {
    const hist = loadHistory(session.id)
    setMessages(hist)
    msgId.current = hist.length > 0 ? Math.max(...hist.map(m => m.id)) + 1 : 0
    setExpandedSources(new Set())
  }, [session.id])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const collection = session.collection
  const extraCollections = session.extraCollections

  const sendMessage = async () => {
    if (!input.trim() || loading || !collection) return
    const question = input.trim()
    setInput('')
    setLoading(true)

    // 首条消息时自动设置会话标题
    if (session.title === '新对话' && messages.length === 0) {
      onSessionUpdate({ title: question.length > 22 ? question.slice(0, 22) + '…' : question })
    }

    const userMsg: Message = { id: ++msgId.current, role: 'user', content: question }
    const botId = ++msgId.current
    const botMsg: Message = { id: botId, role: 'assistant', content: '', loading: true }
    setMessages(prev => [...prev, userMsg, botMsg])

    try {
      const historyToSend = messages
        .filter(m => !m.loading && m.content)
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content }))

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question, collection,
          history: historyToSend,
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
          decompose_enabled: decomposeEnabled,
          raptor_enabled: raptorEnabled,
          kg_enabled: kgEnabled,
          graphrag_enabled: graphragEnabled,
          parent_child_enabled: parentChildEnabled,
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
              if (!hasStreamedTokens) {
                setMessages(prev => prev.map(m =>
                  m.id === botId ? { ...m, content: event.content, loading: false } : m
                ))
              }
            } else if (event.type === 'fallback') {
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, fallbackType: event.content } : m
              ))
            } else if (event.type === 'sub_queries') {
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, subQueries: event.content } : m
              ))
            } else if (event.type === 'sources') {
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, sources: event.content } : m
              ))
            } else if (event.type === 'status') {
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

    setMessages(prev => {
      const final = prev.filter(m => !m.loading)
      saveHistory(session.id, final)
      return prev
    })
    setLoading(false)
  }

  const deleteMessage = useCallback((id: number) => {
    setMessages(prev => {
      const idx = prev.findIndex(m => m.id === id)
      if (idx === -1) return prev
      const msg = prev[idx]
      const toRemove = new Set([id])
      if (msg.role === 'user' && prev[idx + 1]?.role === 'assistant') toRemove.add(prev[idx + 1].id)
      else if (msg.role === 'assistant' && prev[idx - 1]?.role === 'user') toRemove.add(prev[idx - 1].id)
      const next = prev.filter(m => !toRemove.has(m.id))
      saveHistory(session.id, next)
      return next
    })
  }, [session.id])

  const handleClearAll = () => {
    if (!confirm('确认清空本对话的全部记录？')) return
    setMessages([])
    clearHistory(session.id)
  }

  const toggleSources = (id: number) => {
    setExpandedSources(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>

      {/* 顶部工具栏：KB 选择器 + 清空按钮 */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 12, flexShrink: 0, gap: 10,
      }}>
        <KBSelector
          collection={collection}
          extraCollections={extraCollections}
          onChange={(col, extra) => onSessionUpdate({ collection: col, extraCollections: extra })}
          refreshTrigger={refreshTrigger}
        />
        {messages.length > 0 && (
          <button
            onClick={handleClearAll}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-dim)', fontSize: 11, padding: '2px 6px',
              borderRadius: 6, transition: 'color 0.15s', flexShrink: 0,
            }}
            onMouseEnter={e => (e.currentTarget.style.color = 'var(--danger)')}
            onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
          >
            <Eraser size={12} /> 清空记录
          </button>
        )}
      </div>

      {/* 消息区 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 4px 16px', display: 'flex', flexDirection: 'column', gap: 20 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-dim)', paddingTop: 50 }}>
            <div style={{
              width: 72, height: 72, borderRadius: '50%', margin: '0 auto 14px',
              overflow: 'hidden', opacity: 0.72,
              border: '2px solid rgba(100,180,80,0.25)',
              boxShadow: '0 4px 16px rgba(50,120,40,0.1)',
            }}>
              {settings.botAvatar.startsWith('/')
                ? <img src={settings.botAvatar} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                : <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 40 }}>{settings.botAvatar}</div>}
            </div>
            {collection
              ? <div style={{ fontSize: 14, fontWeight: 500 }}>
                  知识库：<span style={{ color: 'var(--green-mid)' }}>{collection}</span>
                  {extraCollections.length > 0 && <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}> +{extraCollections.length}</span>}
                  <div style={{ fontSize: 12, marginTop: 6, fontWeight: 400 }}>输入问题开始对话</div>
                </div>
              : <div style={{ fontSize: 13 }}>点击上方选择知识库，然后开始提问</div>
            }
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
                    : <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>{av}</div>
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
                    borderRadius: 6, flexShrink: 0, transition: 'color 0.15s', opacity: 0.7,
                  }}
                  onMouseEnter={e => (e.currentTarget.style.color = 'var(--danger)')}
                  onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
                >
                  <Trash2 size={13} />
                </button>
              )}
            </div>

            {/* Query 拆解子问题 */}
            {msg.subQueries && msg.subQueries.length > 0 && (
              <div style={{
                maxWidth: '86%', marginLeft: 56, marginTop: 5,
                padding: '6px 12px', borderRadius: 8,
                background: 'rgba(100,160,255,0.07)',
                border: '1px solid rgba(100,160,255,0.25)',
                fontSize: 11, color: 'var(--text-dim)',
              }}>
                <span style={{ marginRight: 5 }}>🔀</span>
                <span style={{ opacity: 0.7 }}>拆解为 {msg.subQueries.length} 个子问题：</span>
                {msg.subQueries.map((q, i) => (
                  <div key={i} style={{ marginTop: 3, paddingLeft: 8, opacity: 0.85 }}>{i + 1}. {q}</div>
                ))}
              </div>
            )}

            {/* Fallback 标注 */}
            {msg.fallbackType && (
              <div style={{
                maxWidth: '86%', marginLeft: 56, marginTop: 5,
                padding: '6px 12px', borderRadius: 8,
                background: msg.fallbackType.type === 'web' ? 'rgba(50,150,255,0.07)' : 'rgba(255,200,50,0.10)',
                border: `1px solid ${msg.fallbackType.type === 'web' ? 'rgba(50,130,220,0.25)' : 'rgba(200,150,30,0.25)'}`,
                fontSize: 11,
                color: msg.fallbackType.type === 'web' ? 'rgba(30,100,200,0.9)' : 'rgba(160,110,20,0.9)',
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
                        style={{ color: 'rgba(30,100,200,0.85)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 4 }}
                        onMouseEnter={e => (e.currentTarget.style.textDecoration = 'underline')}
                        onMouseLeave={e => (e.currentTarget.style.textDecoration = 'none')}
                      >
                        <span style={{ opacity: 0.6 }}>[{i + 1}]</span>
                        <span>{s.title || s.url}</span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            )}

            {/* 来源引用 */}
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
            placeholder={collection ? `向「${collection}」提问...` : '请先选择知识库'}
            rows={2}
            disabled={!collection}
            style={{ resize: 'none', lineHeight: 1.55, flex: 1, opacity: collection ? 1 : 0.5 }}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
            }}
          />
          <button
            className="btn-primary"
            style={{ padding: '10px 18px', flexShrink: 0, height: 58 }}
            onClick={sendMessage}
            disabled={loading || !input.trim() || !collection}
          >
            <Send size={16} />
          </button>
        </div>

        {/* 功能开关 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Enter 发送 · Shift+Enter 换行</div>

          {[
            {
              key: 'decompose', label: '多跳拆解', emoji: '🔀',
              value: decomposeEnabled, color: '#38b89a', bg: 'rgba(60,180,140,0.15)', border: 'rgba(60,180,140,0.45)',
              toggle: () => setDecomposeEnabled(v => { const n = !v; localStorage.setItem('decompose_enabled', String(n)); return n }),
            },
            {
              key: 'cross', label: '双语检索', emoji: '🌐',
              value: crossLingualEnabled, color: 'var(--green-mid)', bg: 'rgba(90,170,76,0.15)', border: 'rgba(90,170,76,0.45)',
              toggle: () => setCrossLingualEnabled(v => { const n = !v; localStorage.setItem('cross_lingual_enabled', String(n)); return n }),
            },
            {
              key: 'raptor', label: 'RAPTOR', emoji: '🌲',
              value: raptorEnabled, color: '#a07ee0', bg: 'rgba(120,90,200,0.15)', border: 'rgba(120,90,200,0.45)',
              toggle: () => setRaptorEnabled(v => {
                const n = !v
                if (n) { setKgEnabled(false); setGraphragEnabled(false); localStorage.setItem('kg_enabled','false'); localStorage.setItem('graphrag_enabled','false') }
                localStorage.setItem('raptor_enabled', String(n)); return n
              }),
            },
            {
              key: 'kg', label: '知识图谱', emoji: '🕸️',
              value: kgEnabled, color: '#c89040', bg: 'rgba(200,140,50,0.15)', border: 'rgba(200,140,50,0.45)',
              toggle: () => setKgEnabled(v => {
                const n = !v
                if (n) { setRaptorEnabled(false); setGraphragEnabled(false); localStorage.setItem('raptor_enabled','false'); localStorage.setItem('graphrag_enabled','false') }
                localStorage.setItem('kg_enabled', String(n)); return n
              }),
            },
            {
              key: 'graphrag', label: 'GraphRAG', emoji: '🔭',
              value: graphragEnabled, color: '#30a0c8', bg: 'rgba(50,160,200,0.15)', border: 'rgba(50,160,200,0.45)',
              toggle: () => setGraphragEnabled(v => {
                const n = !v
                if (n) { setRaptorEnabled(false); setKgEnabled(false); localStorage.setItem('raptor_enabled','false'); localStorage.setItem('kg_enabled','false') }
                localStorage.setItem('graphrag_enabled', String(n)); return n
              }),
            },
            {
              key: 'parent_child', label: '父子检索', emoji: '📄',
              value: parentChildEnabled, color: '#8b6fa0', bg: 'rgba(139,111,160,0.15)', border: 'rgba(139,111,160,0.45)',
              toggle: () => setParentChildEnabled(v => {
                const n = !v; localStorage.setItem('parent_child_enabled', String(n)); return n
              }),
            },
          ].map(({ key, label, emoji, value, color, bg, border, toggle }) => (
            <button
              key={key}
              onClick={toggle}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                background: value ? bg : 'none',
                border: `1px solid ${value ? border : border.replace(/[\d.]+\)$/, '0.2)')}`,
                borderRadius: 8, padding: '3px 8px', cursor: 'pointer',
                fontSize: 11, color: value ? color : 'var(--text-dim)',
                transition: 'all 0.15s', flexShrink: 0,
              }}
            >
              <span style={{ fontSize: 12 }}>{emoji}</span>
              {label}
            </button>
          ))}
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
