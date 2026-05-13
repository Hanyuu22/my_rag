import { useState, useEffect, useRef } from 'react'
import { Database, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'

interface Collection { name: string; count: number; embedding_model?: string }

interface Props {
  collection: string
  extraCollections: string[]
  onChange: (collection: string, extraCollections: string[]) => void
  refreshTrigger?: number
}

export default function KBSelector({ collection, extraCollections, onChange, refreshTrigger }: Props) {
  const [collections, setCollections] = useState<Collection[]>([])
  const [expanded, setExpanded] = useState(false)
  const [loading, setLoading] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const load = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/collections')
      const data = await res.json()
      setCollections(data.collections || [])
    } catch {}
    setLoading(false)
  }

  useEffect(() => { load() }, [refreshTrigger])

  useEffect(() => {
    if (!expanded) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setExpanded(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [expanded])

  const selectPrimary = (name: string) => {
    onChange(name, extraCollections.filter(n => n !== name))
  }

  const toggleExtra = (e: React.MouseEvent, name: string) => {
    e.stopPropagation()
    if (name === collection) return
    const next = extraCollections.includes(name)
      ? extraCollections.filter(n => n !== name)
      : [...extraCollections, name]
    onChange(collection, next)
  }

  const totalCount = 1 + extraCollections.length

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={() => { if (collections.length === 0) load(); setExpanded(v => !v) }}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: collection ? 'rgba(90,170,76,0.08)' : 'rgba(255,200,50,0.1)',
          border: collection
            ? '1px solid rgba(90,170,76,0.28)'
            : '1px dashed rgba(200,150,30,0.5)',
          borderRadius: 8, padding: '5px 11px', cursor: 'pointer',
          fontSize: 12, transition: 'all 0.15s', color: 'var(--text-primary)',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = collection ? 'rgba(90,170,76,0.14)' : 'rgba(255,200,50,0.16)' }}
        onMouseLeave={e => { e.currentTarget.style.background = collection ? 'rgba(90,170,76,0.08)' : 'rgba(255,200,50,0.1)' }}
      >
        <Database size={12} color={collection ? 'var(--green-mid)' : '#c89040'} />
        {collection ? (
          <>
            <span style={{ color: 'var(--green-mid)', fontWeight: 600, maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {collection}
            </span>
            {extraCollections.length > 0 && (
              <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}>+{extraCollections.length}</span>
            )}
          </>
        ) : (
          <span style={{ color: '#c89040' }}>选择知识库</span>
        )}
        {expanded ? <ChevronUp size={11} color="var(--text-dim)" /> : <ChevronDown size={11} color="var(--text-dim)" />}
      </button>

      {expanded && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 6px)', left: 0, zIndex: 200,
          background: 'rgba(255,255,255,0.98)', backdropFilter: 'blur(20px)',
          border: '1px solid rgba(100,180,80,0.22)', borderRadius: 10,
          boxShadow: '0 8px 28px rgba(0,0,0,0.13)',
          minWidth: 280, maxWidth: 360,
        }}>
          {/* 头部 */}
          <div style={{
            padding: '9px 12px 8px', borderBottom: '1px solid rgba(100,180,80,0.12)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              点击选为<strong>主库</strong>，勾选加入<strong>并行检索</strong>
              {totalCount > 1 && <span style={{ color: 'var(--green-mid)', marginLeft: 6 }}>已选 {totalCount} 个库</span>}
            </span>
            <button
              onClick={e => { e.stopPropagation(); load() }}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dim)', padding: 2, display: 'flex' }}
            >
              <RefreshCw size={11} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
            </button>
          </div>

          {/* 列表 */}
          <div style={{ maxHeight: 260, overflowY: 'auto', padding: '4px 0' }}>
            {collections.length === 0 ? (
              <div style={{ padding: '16px', fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>
                {loading ? '加载中...' : '暂无知识库，请先上传文件'}
              </div>
            ) : collections.map(col => {
              const isPrimary = col.name === collection
              const isExtra = extraCollections.includes(col.name)
              return (
                <div
                  key={col.name}
                  onClick={() => selectPrimary(col.name)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 9,
                    padding: '8px 13px', cursor: 'pointer',
                    background: isPrimary ? 'rgba(90,170,76,0.08)' : 'transparent',
                    transition: 'background 0.1s',
                  }}
                  onMouseEnter={e => { if (!isPrimary) e.currentTarget.style.background = 'rgba(0,0,0,0.03)' }}
                  onMouseLeave={e => { if (!isPrimary) e.currentTarget.style.background = 'transparent' }}
                >
                  <Database size={12} color={isPrimary ? 'var(--accent)' : 'var(--text-dim)'} style={{ flexShrink: 0 }} />
                  <span style={{
                    flex: 1, fontSize: 13, fontWeight: isPrimary ? 500 : 400,
                    color: isPrimary ? 'var(--text-primary)' : 'var(--text-secondary)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {col.name}
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--text-dim)', flexShrink: 0 }}>
                    {col.embedding_model?.includes('bge-m3') ? 'm3' : 'zh'} · {col.count}
                  </span>
                  <input
                    type="checkbox"
                    checked={isExtra}
                    disabled={isPrimary}
                    title={isPrimary ? '当前主库' : isExtra ? '取消并行检索' : '加入并行检索'}
                    onClick={e => toggleExtra(e, col.name)}
                    onChange={() => {}}
                    style={{ cursor: isPrimary ? 'not-allowed' : 'pointer', opacity: isPrimary ? 0.25 : 0.85, flexShrink: 0 }}
                  />
                </div>
              )
            })}
          </div>

          {collection && (
            <div style={{
              padding: '7px 13px', borderTop: '1px solid rgba(100,180,80,0.1)',
              fontSize: 11, color: 'var(--text-dim)',
            }}>
              主库：<strong style={{ color: 'var(--green-mid)' }}>{collection}</strong>
              {extraCollections.length > 0 && <>
                <span style={{ margin: '0 4px' }}>·</span>
                附加：{extraCollections.join('、')}
              </>}
            </div>
          )}
        </div>
      )}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
