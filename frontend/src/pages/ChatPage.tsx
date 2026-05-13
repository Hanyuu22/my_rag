import { useState, useEffect, useCallback } from 'react'
import ChatSessionList, { type ChatSession } from '../components/ChatSessionList'
import ChatWindow from '../components/ChatWindow'
import type { Settings } from '../hooks/useSettings'

const SESSIONS_KEY = 'rag_sessions'

function loadSessions(): ChatSession[] {
  try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || '[]') }
  catch { return [] }
}

function saveSessions(sessions: ChatSession[]) {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions))
}

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

interface Props {
  settings: Settings
  refreshTrigger: number
}

export default function ChatPage({ settings, refreshTrigger }: Props) {
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions)
  const [activeId, setActiveId] = useState<string | null>(() => {
    const s = loadSessions()
    if (s.length === 0) return null
    return [...s].sort((a, b) => b.updatedAt - a.updatedAt)[0].id
  })

  const activeSession = sessions.find(s => s.id === activeId) ?? null

  const createSession = useCallback(() => {
    const now = Date.now()
    const s: ChatSession = {
      id: genId(), title: '新对话',
      collection: '', extraCollections: [],
      createdAt: now, updatedAt: now,
    }
    setSessions(prev => {
      const next = [...prev, s]
      saveSessions(next)
      return next
    })
    setActiveId(s.id)
  }, [])

  const deleteSession = useCallback((id: string) => {
    if (!confirm('确认删除这条对话？')) return
    localStorage.removeItem(`rag_chat_history_${id}`)
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id)
      saveSessions(next)
      return next
    })
    setActiveId(prev => {
      if (prev !== id) return prev
      const remaining = loadSessions().filter(s => s.id !== id)
      return remaining.length > 0
        ? [...remaining].sort((a, b) => b.updatedAt - a.updatedAt)[0].id
        : null
    })
  }, [])

  const updateSession = useCallback((id: string, updates: Partial<ChatSession>) => {
    setSessions(prev => {
      const next = prev.map(s => s.id === id ? { ...s, ...updates, updatedAt: Date.now() } : s)
      saveSessions(next)
      return next
    })
  }, [])

  // 首次加载没有会话时自动新建一个
  useEffect(() => {
    if (sessions.length === 0) createSession()
  }, [])

  return (
    <div style={{ display: 'flex', height: '100%', gap: 16 }}>
      {/* 左侧会话列表 */}
      <div className="glass-card" style={{
        width: 210, flexShrink: 0, padding: '16px 12px',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <ChatSessionList
          sessions={sessions}
          activeId={activeId}
          onSelect={setActiveId}
          onNew={createSession}
          onDelete={deleteSession}
        />
      </div>

      {/* 右侧聊天区 */}
      <div style={{ flex: 1, display: 'flex', justifyContent: 'center', minWidth: 0 }}>
        <div className="glass-card" style={{
          width: '100%', maxWidth: 820,
          padding: '20px 24px',
          display: 'flex', flexDirection: 'column', minHeight: 0,
        }}>
          {activeSession ? (
            <ChatWindow
              session={activeSession}
              settings={settings}
              onSessionUpdate={(updates) => updateSession(activeSession.id, updates)}
              refreshTrigger={refreshTrigger}
            />
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)' }}>
              <div style={{ textAlign: 'center', fontSize: 14 }}>点击左侧「新建对话」开始</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
