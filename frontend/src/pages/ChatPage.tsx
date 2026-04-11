import { useState } from 'react'
import CollectionList from '../components/CollectionList'
import ChatWindow from '../components/ChatWindow'
import type { Settings } from '../hooks/useSettings'

interface Props {
  settings: Settings
  refreshTrigger: number
}

export default function ChatPage({ settings, refreshTrigger }: Props) {
  const [selectedCollection, setSelectedCollection] = useState('')
  const [extraCollections, setExtraCollections] = useState<string[]>([])

  // 切换主库时，把它从附加库里移除（避免重复）
  const handleSelectPrimary = (name: string) => {
    setSelectedCollection(name)
    setExtraCollections(prev => prev.filter(n => n !== name))
  }

  const headerLabel = extraCollections.length > 0
    ? <>{selectedCollection} <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}>+ {extraCollections.length} 个附加库</span></>
    : selectedCollection

  return (
    <div style={{ display: 'flex', height: '100%', gap: 16 }}>
      {/* 左侧知识库选择 */}
      <div className="glass-card" style={{
        width: 230, flexShrink: 0, padding: '18px 14px', overflowY: 'auto',
      }}>
        <CollectionList
          selected={selectedCollection}
          onSelect={handleSelectPrimary}
          extraSelected={extraCollections}
          onExtraSelect={setExtraCollections}
          refreshTrigger={refreshTrigger}
        />
      </div>

      {/* 右侧聊天，限制最大宽度避免过宽 */}
      <div style={{ flex: 1, display: 'flex', justifyContent: 'center', minWidth: 0 }}>
        <div className="glass-card" style={{
          width: '100%', maxWidth: 820,
          padding: '20px 24px',
          display: 'flex', flexDirection: 'column', minHeight: 0,
        }}>
          <div style={{
            fontSize: 12, color: 'var(--text-dim)', fontWeight: 500,
            textTransform: 'uppercase', letterSpacing: '0.06em',
            marginBottom: 14, flexShrink: 0,
          }}>
            {selectedCollection
              ? <>问答 · <span style={{ color: 'var(--green-mid)', textTransform: 'none', fontWeight: 600 }}>{headerLabel}</span></>
              : '选择知识库开始问答'}
          </div>
          <ChatWindow
            collection={selectedCollection}
            extraCollections={extraCollections}
            settings={settings}
          />
        </div>
      </div>
    </div>
  )
}
