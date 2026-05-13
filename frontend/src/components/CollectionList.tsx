import { useEffect, useState } from 'react'
import { Database, Trash2, RefreshCw } from 'lucide-react'

interface Collection { name: string; count: number; embedding_model?: string }
interface Props {
  selected: string
  onSelect: (name: string) => void
  extraSelected?: string[]
  onExtraSelect?: (names: string[]) => void
  refreshTrigger: number
}

export default function CollectionList({ selected, onSelect, extraSelected = [], onExtraSelect, refreshTrigger }: Props) {
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)

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

  const handleDelete = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm(`确认删除知识库「${name}」？此操作不可撤销。`)) return
    setDeleting(name)
    try {
      await fetch(`/api/collections/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (selected === name) onSelect('')
      await load()
    } catch {}
    setDeleting(null)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: 'var(--text-dim)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          知识库列表
        </span>
        <button className="btn-ghost" style={{ padding: '3px 8px', fontSize: 11 }} onClick={load}>
          <RefreshCw size={11} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
        </button>
      </div>

      {collections.length === 0 ? (
        <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
          暂无知识库，请先上传文件
        </div>
      ) : (
        <>
          {/* 附加库提示 */}
          {extraSelected.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-dim)', padding: '2px 2px 4px', opacity: 0.8 }}>
              勾选 {extraSelected.length} 个附加库将与主库并行检索
            </div>
          )}
          {collections.map(col => {
            const isPrimary = selected === col.name
            const isExtra = extraSelected.includes(col.name)

            const toggleExtra = (e: React.MouseEvent) => {
              e.stopPropagation()
              if (isPrimary || !onExtraSelect) return
              onExtraSelect(
                isExtra
                  ? extraSelected.filter(n => n !== col.name)
                  : [...extraSelected, col.name]
              )
            }

            return (
              <div
                key={col.name}
                className="glass-card"
                style={{
                  padding: '10px 14px', cursor: 'pointer',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  border: isPrimary
                    ? '1px solid var(--green-muted)'
                    : isExtra
                      ? '1px solid rgba(90,170,76,0.35)'
                      : '1px solid var(--glass-border)',
                  background: isPrimary
                    ? 'rgba(122,158,114,0.15)'
                    : isExtra
                      ? 'rgba(90,170,76,0.06)'
                      : undefined,
                }}
                onClick={() => onSelect(col.name)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                  <Database size={14} color={isPrimary ? 'var(--accent)' : 'var(--text-dim)'} />
                  <span style={{
                    fontSize: 13, color: isPrimary ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontWeight: isPrimary ? 500 : 400,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {col.name}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                  {col.embedding_model && (
                    <span className="badge badge-dim" style={{ fontSize: 10, opacity: 0.75 }}>
                      {col.embedding_model.includes('bge-m3') ? 'm3' : 'zh'}
                    </span>
                  )}
                  <span className="badge badge-dim">{col.count} chunks</span>
                  {/* 附加库勾选框（仅在有 onExtraSelect 时显示，即问答页） */}
                  {onExtraSelect && (
                    <input
                      type="checkbox"
                      checked={isExtra}
                      disabled={isPrimary}
                      title={isPrimary ? '当前主库' : (isExtra ? '取消并行检索' : '加入并行检索')}
                      onChange={() => {}}
                      onClick={toggleExtra}
                      style={{ cursor: isPrimary ? 'not-allowed' : 'pointer', opacity: isPrimary ? 0.3 : 0.8 }}
                    />
                  )}
                  <button
                    className="btn-danger"
                    style={{ padding: '2px 6px' }}
                    disabled={deleting === col.name}
                    onClick={e => handleDelete(col.name, e)}
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              </div>
            )
          })}
        </>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
