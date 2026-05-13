import { useState } from 'react'
import { MessageSquare, Plus, Trash2, Database } from 'lucide-react'

export interface ChatSession {
  id: string
  title: string
  collection: string
  extraCollections: string[]
  createdAt: number
  updatedAt: number
}

interface Props {
  sessions: ChatSession[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
}

export default function ChatSessionList({ sessions, activeId, onSelect, onNew, onDelete }: Props) {
  const [hovered, setHovered] = useState<string | null>(null)

  const sorted = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt)

  const formatTime = (ts: number) => {
    const d = new Date(ts)
    const diff = Date.now() - ts
    if (diff < 86400000) return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    if (diff < 604800000) return ['日','一','二','三','四','五','六'][d.getDay()] ? `周${'日一二三四五六'[d.getDay()]}` : ''
    return `${d.getMonth() + 1}/${d.getDate()}`
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 8 }}>
      {/* 标题 */}
      <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em', paddingBottom: 4 }}>
        对话记录
      </div>

      {/* 新建按钮 */}
      <button
        onClick={onNew}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
          padding: '8px 0', borderRadius: 10, flexShrink: 0,
          background: 'rgba(90,170,76,0.1)',
          border: '1px dashed rgba(90,170,76,0.4)',
          cursor: 'pointer', color: 'var(--green-mid)', fontSize: 12.5, fontWeight: 500,
          transition: 'all 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(90,170,76,0.18)' }}
        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(90,170,76,0.1)' }}
      >
        <Plus size={14} /> 新建对话
      </button>

      {/* 会话列表 */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 3 }}>
        {sorted.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-dim)', fontSize: 12, paddingTop: 20 }}>
            点击上方新建对话
          </div>
        ) : sorted.map(s => {
          const isActive = s.id === activeId
          return (
            <div
              key={s.id}
              onClick={() => onSelect(s.id)}
              onMouseEnter={() => setHovered(s.id)}
              onMouseLeave={() => setHovered(null)}
              style={{
                padding: '8px 9px', borderRadius: 9, cursor: 'pointer',
                background: isActive ? 'rgba(90,170,76,0.14)' : hovered === s.id ? 'rgba(0,0,0,0.035)' : 'transparent',
                border: isActive ? '1px solid rgba(90,170,76,0.32)' : '1px solid transparent',
                transition: 'all 0.15s',
              }}
            >
              {/* 标题行 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 0, flex: 1 }}>
                  <MessageSquare size={11} color={isActive ? 'var(--green-mid)' : 'var(--text-dim)'} style={{ flexShrink: 0 }} />
                  <span style={{
                    fontSize: 12.5, fontWeight: isActive ? 500 : 400,
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {s.title || '新对话'}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexShrink: 0, marginLeft: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-dim)', opacity: 0.7 }}>{formatTime(s.updatedAt)}</span>
                  {hovered === s.id && (
                    <button
                      onClick={e => { e.stopPropagation(); onDelete(s.id) }}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--text-dim)', padding: '1px 2px', borderRadius: 4,
                        display: 'flex', transition: 'color 0.15s',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.color = 'var(--danger)')}
                      onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
                </div>
              </div>

              {/* 知识库标签 */}
              {s.collection ? (
                <div style={{ marginTop: 4, display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                  <span style={{
                    fontSize: 9.5, padding: '1px 5px', borderRadius: 4,
                    background: 'rgba(90,170,76,0.1)', color: 'var(--green-mid)',
                    border: '1px solid rgba(90,170,76,0.2)',
                    display: 'flex', alignItems: 'center', gap: 3, maxWidth: 120,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    <Database size={8} style={{ flexShrink: 0 }} />
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.collection}</span>
                  </span>
                  {s.extraCollections.map(ec => (
                    <span key={ec} style={{
                      fontSize: 9.5, padding: '1px 5px', borderRadius: 4,
                      background: 'rgba(90,170,76,0.05)', color: 'var(--text-dim)',
                      border: '1px solid rgba(90,170,76,0.12)',
                      maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      +{ec}
                    </span>
                  ))}
                </div>
              ) : (
                <div style={{ marginTop: 3, fontSize: 9.5, color: 'rgba(200,150,30,0.8)', opacity: 0.8 }}>
                  未选择知识库
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
