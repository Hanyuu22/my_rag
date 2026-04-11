import UploadZone from '../components/UploadZone'
import CollectionList from '../components/CollectionList'

interface Props {
  refreshTrigger: number
  onUploaded: () => void
}

export default function KnowledgePage({ refreshTrigger, onUploaded }: Props) {
  return (
    <div style={{ display: 'flex', gap: 16, height: '100%', alignItems: 'flex-start' }}>
      {/* 上传区 */}
      <div style={{ flex: 1, overflowY: 'auto', maxWidth: 620 }}>
        <div className="glass-card" style={{ padding: 28 }}>
          <div style={{
            fontSize: 13, fontWeight: 600, color: 'var(--green-deep)',
            marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{ fontSize: 16 }}>📁</span> 上传文件到知识库
          </div>
          <UploadZone onUploaded={onUploaded} />
        </div>
      </div>

      {/* 知识库列表 */}
      <div className="glass-card" style={{ width: 280, flexShrink: 0, padding: 18, overflowY: 'auto' }}>
        <CollectionList selected="" onSelect={() => {}} refreshTrigger={refreshTrigger} />
      </div>
    </div>
  )
}
